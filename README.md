# Aisthesis v5

**Architettura ibrida per la classificazione emotiva e la mappatura sinestetica in tempo reale.**

Aisthesis traduce testo italiano in una **palette cromatica** ispirata alla *Teoria dei Colori* di Goethe e in un **paesaggio sonoro** sintetizzato nel browser. Il sistema combina il classificatore neurale [FEEL-IT](https://huggingface.co/MilaNLProc/feel-it-italian-emotion) (UmBERTo, 4 emozioni) con un **lessico cromatico** italiano (11 colori) e restituisce spiegazioni tramite Integrated Gradients e allineamento lessicale.

Progetto di laurea magistrale — Università degli Studi Roma Tre.

---

## Caratteristiche principali

- **11 colori Goethe** — 7 canonici (*Zur Farbenlehre*) + 4 estensioni documentate (rosa, azzurro, verde marcio, cremisi)
- **Fusione ibrida** neurale + lessicale con *boost* lessicale su frasi brevi (fino al 72% sul canale simbolico)
- **Blend cromatico** multi-colore (top-4, smoothing) con media circolare HSV
- **Sonificazione deterministica** da profili `SonicColorProfile` per colore (Web Audio API)
- **Spazio affettivo Russell** (ausiliario, per visualizzazione)
- **Explainability** — Integrated Gradients su FEEL-IT + traccia esplicativa delle regole goethiane

---

## Pipeline

```
Testo italiano
    │
    ├─► FEEL-IT ──► emozioni (joy, anger, sadness, fear) ──► mappa ──► colori Goethe (m)
    │
    └─► GOETHE_LEXICON ──► parole → colori (ℓ)
                │
                ▼
         fusione ibrida (α·m + β·ℓ)  ──►  top-k + smoothing  ──►  blend ẇ
                │                                              │
                ├─► HSV visivo                                   ├─► profilo sonoro
                ├─► polo attivo/passivo                          └─► Web Audio API
                ├─► spazio Russell (ausiliario)
                └─► IG + rule trace
```

**Pesi di default:** α = β = 0,5. Su frasi ≤ 6 parole con picco lessicale ≥ 0,4: β = 0,72, α = 0,28.

---

## Requisiti

- Python 3.10+
- ~2 GB di spazio per il download del modello Hugging Face (primo avvio)
- Browser moderno con supporto Web Audio API

Opzionale: GPU CUDA (l'inferenza funziona anche su CPU).

---

## Installazione

```bash
git clone https://github.com/Alessio-Pecilli/Aisthesis.git
cd Aisthesis
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Al primo avvio viene scaricato automaticamente `MilaNLProc/feel-it-italian-emotion`. Per limiti di rate più alti su Hugging Face, opzionale:

```bash
cp .env.example .env
# imposta HF_TOKEN=hf_...
```

---

## Avvio

```bash
python main.py
```

Apri [http://127.0.0.1:8000](http://127.0.0.1:8000) nel browser, inserisci un testo e osserva colore, suono e spiegazioni in tempo reale.

---

## API

### `POST /process`

**Request**

```json
{ "text": "sono felice sotto il sole" }
```

**Response** (campi principali)

| Campo | Descrizione |
|-------|-------------|
| `goethe_analysis` | Blend cromatico, colore dominante, polo attivo/passivo |
| `visual_state.layers` | Layer HSV con citazione goethiana, `pole`, `source` |
| `sonic_profile` | Pitch, scala, BPM, timbro, riverbero (da blend cromatico) |
| `semantic_analysis` | Emozioni proxy, spazio Russell (valenza / arousal) |
| `explainability` | Termini salienti (IG), allineamento lessicale, `contradiction` |

**Esempio con curl**

```bash
curl -s -X POST http://127.0.0.1:8000/process \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"gioia\"}" | python -m json.tool
```

---

## Struttura del repository

```
Aisthesis/
├── main.py              # Backend FastAPI, pipeline end-to-end, IG
├── goethe.py            # Ontologia 11 colori, lessico, fusione, sonificazione
├── static/
│   └── index.html       # Frontend + Web Audio API
├── scripts/
│   ├── eval_corpus.json       # Corpus di controllo (N=44)
│   ├── ablation_study.py      # Ablation study riproducibile
│   └── fear_bias_metrics.py   # Analisi bias verso fear
└── docs/                # Relazione tecnica, risultati, figure
```

---

## Valutazione (studio preliminare)

Su un corpus di controllo di **44 frasi italiane** (`scripts/eval_corpus.json`, gold label redatte dagli autori):

| Configurazione | Accuratezza cromatica |
|----------------|----------------------|
| Solo FEEL-IT | 31,8% |
| Solo lessico | 86,4% |
| **Ibrido completo (v5)** | **81,8%** |

L'ibrido recupera il 100% sulle etichette affettive isolate nel corpus, dove FEEL-IT da solo è al 18,8%. I risultati sono **preliminari**: calibrazione e valutazione usano lo stesso insieme di esempi.

**Rigenerare i risultati:**

```bash
python scripts/ablation_study.py
```

Output: `docs/ablation_results.json`, grafici in `docs/figures/`.

---

## Palette cromatica

| Tipo | Colori |
|------|--------|
| **Canonici (7)** | Giallo, Arancione, Vermiglio, Blu, Violetto, Porpora, Verde |
| **Estensioni (4)** | Rosa (amore), Azzurro (fiducia), Verde Marcio (disgusto), Cremisi (attesa/sorpresa) |

Le estensioni sono motivate da letteratura su emozione–colore e corrispondenze cross-modali (Palmer et al., Marks, Ferguson & Brewster).

---

## Stack tecnologico

| Layer | Tecnologie |
|-------|------------|
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| NLP | PyTorch, Hugging Face Transformers, FEEL-IT / UmBERTo |
| Frontend | HTML, CSS, JavaScript vanilla, Web Audio API |
| Ontologia | `goethe.py` — lessico, blend HSV, profili sonori |

---

## Limiti noti

- FEEL-IT è addestrato su tweet: errori su input minimali e bias verso *fear*
- Il lessico non gestisce la negazione in modo semantico completo (attenuazione 0,25, senza inversione)
- Corpus di valutazione piccolo e autoprodotto, senza annotazione inter-rater
- Latenza dipende da CPU/GPU; IG aggiunge overhead computazionale

---

## Documentazione

La relazione tecnica completa è in [`relazione.pdf`](relazione.pdf).

---

## Riferimenti

- Bianchi et al. — [FEEL-IT](https://aclanthology.org/2021.wassa-1.10/) (2021)
- Goethe — *Zur Farbenlehre* (1810)
- Russell — circumplex model of affect (1980)
- Sundararajan et al. — Integrated Gradients (2017)
- Palmer et al. — emozione e preferenza cromatica (2010)
- Marks — corrispondenze luminosità / pitch (1974)
- Ferguson & Brewster — psicoacustica (2017)

---

## Autori

**Alessio Pecilli** · **Matteo Cerretani**  
Ingegneria Informatica, Laurea Magistrale — Università degli Studi Roma Tre

---

## Licenza

Uso accademico / progetto universitario. Verificare le licenze dei modelli Hugging Face e delle dipendenze prima di un deploy in produzione.
