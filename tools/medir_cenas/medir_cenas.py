#!/usr/bin/env python3
"""Mede o ritmo de um vídeo de referência e gera um storyboard.md.

Para cada cena, detecta o corte, o tipo de transição (corte seco, suave ou
passagem pelo preto) e o movimento de câmera (parada, panorâmica, zoom).
O resumo serve para passar a um LLM junto com a transcrição e pedir um
roteiro novo "no mesmo ritmo".

Uso:
    python medir_cenas.py video.mp4
    python medir_cenas.py video.mp4 -o storyboard.md --json cenas.json --thumbs miniaturas/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, detect

# Lado maior dos quadros usados na análise de movimento (mais rápido, mesmo resultado).
ANALYSIS_WIDTH = 320
# Amostras por segundo para estimar o movimento de câmera.
MOTION_SAMPLES_PER_SEC = 6
# Deslocamento médio (em % da largura por segundo) abaixo do qual a câmera é "parada".
STATIC_PAN_LIMIT = 2.0
# Variação de escala por segundo abaixo da qual não há zoom.
STATIC_ZOOM_LIMIT = 0.03
# Brilho médio (0-255) considerado "tela preta".
BLACK_LEVEL = 18


@dataclass
class Scene:
    index: int
    start: float
    end: float
    duration: float
    transition_in: str = "início"
    camera: str = "parada"
    pan_speed: float = 0.0  # % da largura por segundo
    zoom_speed: float = 0.0  # variação de escala por segundo (+ entra, - sai)


def seconds(tc) -> float:
    """Compatível com versões novas (.seconds) e antigas (.get_seconds()) do PySceneDetect."""
    value = getattr(tc, "seconds", None)
    return float(value) if value is not None else tc.get_seconds()


def fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def small_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = ANALYSIS_WIDTH / max(w, h)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def load_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Lê o vídeo uma vez, do início ao fim, guardando só os quadros pedidos (reduzidos)."""
    cap = cv2.VideoCapture(str(video))
    frames: dict[int, np.ndarray] = {}
    last = max(wanted, default=-1)
    n = 0
    while n <= last and cap.grab():
        if n in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames[n] = small_gray(frame)
        n += 1
    cap.release()
    return frames


def transition_window(cut_frame: int, fps: float) -> range:
    w = max(2, round(fps * 0.4))
    return range(max(0, cut_frame - w), cut_frame + w + 1)


def motion_pairs(scene: Scene, fps: float) -> list[tuple[int, int]]:
    """Pares de quadros (a, b) usados para medir o movimento de câmera da cena."""
    # Ignora as bordas da cena para não confundir a transição com movimento.
    margin = min(0.25, scene.duration * 0.15)
    t0, t1 = scene.start + margin, scene.end - margin
    gap = max(1, round(fps / MOTION_SAMPLES_PER_SEC))
    if t1 - t0 - gap / fps < 0.2:
        return []
    n = min(12, max(1, int((t1 - t0) * MOTION_SAMPLES_PER_SEC / 2)))
    return [(int(t * fps), int(t * fps) + gap) for t in np.linspace(t0, t1 - gap / fps, n)]


def classify_transition(cache: dict[int, np.ndarray], cut_frame: int, fps: float) -> str:
    """Olha os quadros em volta do corte para separar corte seco de transição suave."""
    frames = [cache[f] for f in transition_window(cut_frame, fps) if f in cache]
    if len(frames) < 3:
        return "corte"

    if min(float(f.mean()) for f in frames) < BLACK_LEVEL:
        return "preto"

    diffs = sorted(float(cv2.absdiff(a, b).mean()) for a, b in pairwise(frames))
    # Num corte seco a mudança entre dois quadros vizinhos é muito maior que todas as outras;
    # numa transição suave ela se espalha por vários quadros parecidos.
    return "corte" if diffs[-1] >= 2.5 * max(diffs[-2], 0.5) else "suave"


def global_motion(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float] | None:
    """Movimento do quadro inteiro entre a e b: (dx, dy, escala), ou None se não der para medir.

    Rastreia pontos de textura e ajusta uma transformação (translação + escala) com RANSAC,
    então um personagem se mexendo vira "ponto fora da curva" e não conta como câmera.
    """
    pts = cv2.goodFeaturesToTrack(a, maxCorners=300, qualityLevel=0.01, minDistance=6)
    if pts is None or len(pts) < 8:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < 8:
        return None
    m, inliers = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC,
                                             ransacReprojThreshold=1.5)
    if m is None or inliers is None or inliers.mean() < 0.5:
        return None
    h, w = a.shape
    center = np.array([w / 2, h / 2, 1.0])
    dx, dy = m @ center - center[:2]
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    return float(dx), float(dy), scale


def measure_camera(cache: dict[int, np.ndarray], scene: Scene, fps: float) -> None:
    """Estima panorâmica e zoom pela mediana do movimento global ao longo da cena."""
    pans, zooms = [], []
    for fa, fb in motion_pairs(scene, fps):
        a, b = cache.get(fa), cache.get(fb)
        if a is None or b is None:
            continue
        motion = global_motion(a, b)
        if motion is None:
            continue
        dx, dy, scale = motion
        dt = (fb - fa) / fps
        pans.append(np.hypot(dx, dy) / a.shape[1] * 100 / dt)
        zooms.append((scale - 1) / dt)

    if not pans:
        return
    scene.pan_speed = round(float(np.median(pans)), 2)
    scene.zoom_speed = round(float(np.median(zooms)), 3)

    if abs(scene.zoom_speed) >= STATIC_ZOOM_LIMIT:
        scene.camera = "zoom-in" if scene.zoom_speed > 0 else "zoom-out"
    elif scene.pan_speed >= STATIC_PAN_LIMIT:
        scene.camera = "pan"


def analyze(video: Path, threshold: float, min_scene: float) -> tuple[dict, list[Scene]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir o vídeo: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    print("Detectando cortes...", file=sys.stderr)
    scene_list = detect(
        str(video),
        ContentDetector(threshold=threshold, min_scene_len=max(1, int(min_scene * fps))),
    )
    if not scene_list:
        scene_list = [(None, None)]

    scenes: list[Scene] = []
    for i, (s, e) in enumerate(scene_list, 1):
        start = seconds(s) if s is not None else 0.0
        end = seconds(e) if e is not None else duration
        scenes.append(Scene(i, round(start, 2), round(end, 2), round(end - start, 2)))

    cap.release()

    print(f"{len(scenes)} cenas. Medindo transições e câmera...", file=sys.stderr)
    wanted: set[int] = set()
    for sc in scenes:
        if sc.index > 1:
            wanted.update(transition_window(int(sc.start * fps), fps))
        for a, b in motion_pairs(sc, fps):
            wanted.update((a, b))
    cache = load_frames(video, wanted)

    for sc in scenes:
        if sc.index > 1:
            sc.transition_in = classify_transition(cache, int(sc.start * fps), fps)
        measure_camera(cache, sc, fps)

    info = {"arquivo": video.name, "largura": width, "altura": height,
            "fps": round(fps, 2), "duracao": round(duration, 2)}
    return info, scenes


def build_summary(info: dict, scenes: list[Scene]) -> dict:
    durs = [s.duration for s in scenes]
    transitions: dict[str, int] = {}
    cameras: dict[str, int] = {}
    for s in scenes[1:]:
        transitions[s.transition_in] = transitions.get(s.transition_in, 0) + 1
    for s in scenes:
        cameras[s.camera] = cameras.get(s.camera, 0) + 1

    per_minute = [0] * (int(info["duracao"] // 60) + 1)
    for s in scenes:
        per_minute[min(int(s.start // 60), len(per_minute) - 1)] += 1

    return {
        "cenas": len(scenes),
        "segundos_por_cena_media": round(statistics.mean(durs), 2),
        "segundos_por_cena_mediana": round(statistics.median(durs), 2),
        "cena_mais_curta": round(min(durs), 2),
        "cena_mais_longa": round(max(durs), 2),
        "transicoes": transitions,
        "camera": cameras,
        "cenas_por_minuto": per_minute,
    }


def to_markdown(info: dict, summary: dict, scenes: list[Scene]) -> str:
    def counts(d: dict) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "-"

    lines = [
        f"# Storyboard: {info['arquivo']}",
        "",
        (
            f"{info['largura']}x{info['altura']} @ {info['fps']}fps · {fmt_time(info['duracao'])} · "
            f"**{summary['cenas']} cenas** · média **{summary['segundos_por_cena_media']}s/cena** "
            f"(mediana {summary['segundos_por_cena_mediana']}s)"
        ),
        "",
        f"- Transições: {counts(summary['transicoes'])}",
        f"- Câmera: {counts(summary['camera'])}",
        f"- Cena mais curta: {summary['cena_mais_curta']}s · mais longa: {summary['cena_mais_longa']}s",
        "- Cenas por minuto: " + " | ".join(
            f"{i}' {n}" for i, n in enumerate(summary["cenas_por_minuto"])),
        "",
        "| # | início | duração | entra com | câmera |",
        "|---|--------|---------|-----------|--------|",
    ]
    for s in scenes:
        lines.append(f"| {s.index} | {fmt_time(s.start)} | {s.duration:.2f}s | "
                     f"{s.transition_in} | {s.camera} |")
    return "\n".join(lines) + "\n"


def save_thumbs(video: Path, scenes: list[Scene], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    for s in scenes:
        cap.set(cv2.CAP_PROP_POS_MSEC, (s.start + s.duration / 2) * 1000)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out_dir / f"cena_{s.index:03d}.jpg"), frame)
    cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description="Mede cortes, ritmo e câmera de um vídeo.")
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--out", type=Path, help="storyboard.md (padrão: <video>.storyboard.md)")
    p.add_argument("--json", type=Path, help="também salva os dados em JSON")
    p.add_argument("--thumbs", type=Path, help="pasta para salvar um quadro de cada cena")
    p.add_argument("--threshold", type=float, default=20.0,
                   help="sensibilidade do corte (menor = mais cortes; padrão 20)")
    p.add_argument("--min-scene", type=float, default=0.25,
                   help="duração mínima de uma cena em segundos (padrão 0.25)")
    args = p.parse_args()

    info, scenes = analyze(args.video, args.threshold, args.min_scene)
    summary = build_summary(info, scenes)

    out = args.out or args.video.with_suffix(".storyboard.md")
    out.write_text(to_markdown(info, summary, scenes), encoding="utf-8")
    print(f"Storyboard salvo em {out}", file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps(
            {"video": info, "resumo": summary, "cenas": [asdict(s) for s in scenes]},
            ensure_ascii=False, indent=2), encoding="utf-8")
    if args.thumbs:
        save_thumbs(args.video, scenes, args.thumbs)


if __name__ == "__main__":
    main()
