# Aisthesis v5.0

**Sonificazione semantica goethiana** — testo italiano → palette cromatica (Goethe) → paesaggio sonoro → spiegazioni XAI.

## Cosa fa

- Traduce il segnale emotivo (FEEL-IT + lessico) in **11 colori**: 7 canonici da *Zur Farbenlehre* + 4 estensioni documentate
- Classifica con **lessico cromatico italiano** (ibrido 50/50 con il modello, fino al 72% su frasi corte)
- Fonde i colori in una **tavolozza composita** (media circolare HSV)
- Deriva il **suono dal carattere cromatico** goethiano (scala, pitch, timbro, riverbero per colore)
- Calcola il **polo attivo/passivo** goethiano e lo spazio affettivo Russell (ausiliario)
- Spiega con **Integrated Gradients** e allineamento lessicale

## Pipeline

```
Testo
  → FEEL-IT (joy, anger, sadness, fear)
  → traduzione emozione → colore Goethe
  → lessico cromatico italiano (parole → colore diretto)
  → mix ibrido → blend top-k (fino a 4 colori)
  → sonificazione da profilo sonoro del colore dominante
  → explainability (IG + lexicon_alignment)
```

## Colori

| Tipo | Colori |
|------|--------|
| **Goethe (7)** | Giallo, Arancione, Vermiglio, Blu, Violetto, Porpora, Verde |
| **Estensioni (4)** | Rosa (amore), Azzurro (fiducia), Verde Marcio (disgusto), Cremisi (attesa/sorpresa) |

Le estensioni citano Palmer et al., Marks (luminosità↔pitch), Ferguson & Brewster (roughness/sharpness).

## Avvio

```bash
pip install -r requirements.txt
python main.py
```

Apri [http://127.0.0.1:8000](http://127.0.0.1:8000)

## API

`POST /process` con `{ "text": "..." }`

| Campo | Contenuto |
|-------|-----------|
| `goethe_analysis` | Mix cromatico, polo attivo/passivo, conteggio canonici/estensioni |
| `visual_state.layers` | Layer con `goethe_color`, HSV, citazione, `source`, `pole` |
| `sonic_profile` | Suono derivato dal blend cromatico |
| `semantic_analysis` | Compatibilità: emozioni proxy + spazio Russell |
| `explainability` | IG, allineamento lessico cromatico |

## Riferimenti

- Goethe — *Zur Farbenlehre* (1810)
- Russell — circumplex (valence / arousal, ausiliario)
- Marks — corrispondenze luminosità / pitch / loudness
- Palmer et al. — associazioni emozione–colore
- Ferguson & Brewster — parametri psicoacustici
- Sundararajan et al. — Integrated Gradients

## Stack

- **Backend:** FastAPI, PyTorch, Transformers, `goethe.py` (ontologia cromatica)
- **Frontend:** HTML/CSS/JS, Web Audio API
