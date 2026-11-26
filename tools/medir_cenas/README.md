# medir_cenas

Mede o **ritmo** de um vídeo de referência sem precisar assisti-lo inteiro.
Detecta cada corte, a duração de cada cena, o tipo de transição e o movimento de
câmera, e escreve tudo num `storyboard.md` pequeno, pronto para colar num LLM junto
com a transcrição:

> "Esse storyboard.md e o transcript.txt são o resumo de um vídeo que fez dez vezes o
> normal do canal dele. Faz um roteiro no mesmo ritmo, mas com uma história brasileira."

## Instalação

```bash
pip install -r requirements.txt
```

Precisa de Python 3.10+. Não precisa de GPU.

## Uso

```bash
python medir_cenas.py referencia.mp4
# -> referencia.storyboard.md

python medir_cenas.py referencia.mp4 -o storyboard.md --json cenas.json --thumbs miniaturas/
```

| Opção | O que faz |
|---|---|
| `-o, --out` | Caminho do storyboard (padrão: `<video>.storyboard.md`) |
| `--json` | Salva também os dados brutos em JSON |
| `--thumbs` | Salva um quadro do meio de cada cena numa pasta |
| `--threshold` | Sensibilidade do corte. Menor = mais cortes. Padrão `20` |
| `--min-scene` | Duração mínima de uma cena, em segundos. Padrão `0.25` |

Um vídeo de 9 minutos em 480p leva cerca de 1 minuto num CPU comum.

## Exemplo de saída

```markdown
854x480 @ 30.0fps · 09:07.63 · **101 cenas** · média **5.42s/cena** (mediana 5.07s)

- Transições: corte=54, suave=46
- Câmera: parada=89, zoom-out=6, pan=6
- Cena mais curta: 0.63s · mais longa: 13.8s
- Cenas por minuto: 0' 18 | 1' 10 | 2' 11 | 3' 11 | ...

| # | início | duração | entra com | câmera |
|---|--------|---------|-----------|--------|
| 1 | 00:00.00 | 5.23s | início | parada |
| 2 | 00:05.23 | 2.97s | suave | parada |
| 3 | 00:08.20 | 1.37s | suave | zoom-out |
```

## Como ler o resultado

- **Segundos por cena**: o ritmo do nicho. Desenhos de "legumes que falam" ficam
  perto de 2 s/cena; vídeos explicativos, 4–6 s/cena.
- **Entra com**: `corte` (corte seco), `suave` (dissolve, wipe, slide) ou `preto`
  (passa por tela preta).
- **Câmera**: `parada`, `pan` (a imagem desliza), `zoom-in` ou `zoom-out`. Muitas cenas
  paradas indicam que quem "atua" é a expressão do personagem, não a câmera.
- **Cenas por minuto**: mostra se o vídeo acelera no começo (gancho) e desacelera depois.

Para descobrir o que um nicho espera, meça **duas referências do mesmo nicho** e
compare: o que as duas fazem igual é o padrão daquele público.

## Como funciona

1. **Cortes**: [PySceneDetect](https://www.scenedetect.com/) (`ContentDetector`) compara
   cor e brilho entre quadros vizinhos.
2. **Transição**: em volta de cada corte, mede a diferença entre quadros. Se a mudança
   está concentrada num único par de quadros, é corte seco; se está espalhada, é
   transição suave. Brilho médio perto de zero indica passagem pelo preto.
3. **Câmera**: rastreia pontos de textura entre quadros e ajusta uma transformação
   (translação + escala) com RANSAC. Personagens se mexendo viram "pontos fora da curva"
   e não contam como movimento de câmera.

## Limitações

- Dissolves muito lentos (mais de ~1 s) entre imagens parecidas podem não ser detectados
  como corte. Baixe o `--threshold` (ex.: `12`) se o vídeo usa muitas transições suaves.
- Em motion graphics, um painel deslizando na tela aparece como `pan`, mesmo sem
  câmera "de verdade". Para medir ritmo, isso costuma ser o que interessa.
- Texto aparecendo aos poucos num mesmo slide não conta como corte.
