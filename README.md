# Aisthesis v4.2

**Sonificazione semantica** — progetto universitario che trasforma testo italiano in palette cromatiche (Goethe), paesaggi sonori e spiegazioni XAI.

## Cosa fa

- Classifica le emozioni con **FEEL-IT** (Transformers, italiano)
- Costruisce un **mix top-k** (fino a 4 emozioni), non una sola etichetta
- Corregge errori del modello su testi brevi con un **lessico affettivo italiano** (ibrido 50/50, fino al 72% su frasi corte)
- Fonde i colori Goethe in una **tavolozza composita** (media circolare HSV)
- Deriva **valence / arousal** (circumplex di Russell) per scala, tempo e timbro
- Sintetizza un **profilo sonoro multi-voce** (pentatonica, arpeggio, riverbero)
- Spiega le decisioni con **Integrated Gradients**, analisi contrastiva e allineamento lessico

## Pipeline

```
Testo
  → FEEL-IT (top-k classi)
  → lessico italiano (arrabiato, paura, gioia…)
  → mix ibrido modello + lessico
  → blend cromatico Goethe (top-k layer)
  → spazio affettivo (valence / arousal)
  → sonic_profile (voci, scala, arpeggio)
  → explainability (IG + contrastivo + lexicon_alignment)
```

## Interfaccia

Layout a schermata unica:

- **Sinistra:** area di scrittura (ampia, con padding generoso)
- **Destra:** risultati scrollabili (palette, mix, Goethe, XAI, riascolto)
- **Caricamento:** overlay «L'animo pensa…» con barra di progresso e timer

## Avvio

```bash
pip install -r requirements.txt
python main.py
```

Apri [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API

`POST /process` con `{ "text": "..." }`

Risposta principale:

| Campo | Contenuto |
|-------|-----------|
| `semantic_analysis.emotion_mix` | Mix pesato top-k (guida colore e suono) |
| `semantic_analysis.affective_space` | Valence, arousal, descrizione quadrante |
| `visual_state.layers` | Colori Goethe con peso, HSV, citazione |
| `sonic_profile` | Voci, scala, tempo BPM, arpeggio, riverbero |
| `explainability.lexicon_alignment` | Confronto modello vs parole esplicite nel testo |
| `explainability.influential_terms` | Parole salienti (Integrated Gradients) |

## Lessico ibrido

Su parole affettive esplicite (`arrabiato`, `rabbia`, `piango`, `paura`, `gioia`…) il sistema **non si fida ciecamente del top-1 del modello**. Se c'è disallineamento (es. modello → gioia, testo → rabbia), l'UI mostra un avviso e il mix viene corretto.

## Riferimenti

- Russell — circumplex (valence / arousal)
- Goethe — *Zur Farbenlehre* (analogia affettiva, non mapping 1:1 colore↔suono)
- Ferguson & Brewster — parametri psicoacustici per sonificazione
- Sundararajan et al. — Integrated Gradients
- Winters & Wanderley — sonificazione continua di emozioni

## Stack

- **Backend:** FastAPI, PyTorch, Transformers (`MilaNLProc/feel-it-italian-emotion`)
- **Frontend:** HTML/CSS/JS, Web Audio API
- Nessun LLM generativo né RAG
