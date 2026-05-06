import os
import re
import chromadb
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from goetheColor import parse_emotion

# --- 1. SETUP DEL DATABASE (Percorsi Assoluti per evitare bug di Uvicorn) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

PLUS_SECTIONS = (
    "== GIALLO ",
    "== ROSSO-GIALLO",
    "== GIALLO-ROSSO",
    "== BLU-ROSSO",
)
MINUS_SECTIONS = (
    "== BLU ",
    "== ROSSO-BLU",
)
NEUTRAL_SECTIONS = (
    "== VERDE ",
    "== BLU-ROSSO",
)

MIN_CONTEXT_DOCS = 4
MAX_GOETHE_COLORS = 7
VALID_WAVES = {"sine", "square", "sawtooth", "triangle"}

HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")

def rank_docs_by_polarity(docs: List[str], polarity: str) -> List[str]:
    allowed_sections = {
        "plus": PLUS_SECTIONS,
        "minus": MINUS_SECTIONS,
        "neutral": NEUTRAL_SECTIONS,
    }.get(polarity)

    if not allowed_sections:
        return docs

    matching_docs = [
        doc for doc in docs
        if doc.strip().startswith(allowed_sections)
    ]
    other_docs = [doc for doc in docs if doc not in matching_docs]
    return matching_docs + other_docs

def is_goethe_color_doc(doc: str) -> bool:
    return doc.strip().startswith("==")

def extract_goethe_heading(doc: str) -> str:
    return doc.strip().splitlines()[0].replace("=", "").strip()

def limit_context_docs(docs: List[str], requested_k: int) -> List[str]:
    context_size = min(len(docs), max(requested_k, MIN_CONTEXT_DOCS))
    return docs[:context_size]

def llm_num_predict(requested_k: int, retry: int = 0) -> int:
    return min(2600, 550 + (requested_k * 230) + (retry * 700))

def freq_range_for_polarity(polarity: str) -> tuple[float, float]:
    if polarity == "minus":
        return 100.0, 300.0
    if polarity == "plus":
        return 400.0, 800.0
    return 250.0, 500.0

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

def normalize_wave(wave: str) -> str:
    normalized = (wave or "sine").lower()
    return normalized if normalized in VALID_WAVES else "sine"

def normalize_frequency(value: Any, polarity: str) -> float:
    minimum, maximum = freq_range_for_polarity(polarity)
    try:
        numeric = abs(float(value))
    except (TypeError, ValueError):
        numeric = (minimum + maximum) / 2
    return clamp(numeric, minimum, maximum)

def normalize_melody(melody: List[Dict[str, Any]], polarity: str, fallback_wave: str) -> List[Dict[str, Any]]:
    normalized_notes: List[Dict[str, Any]] = []
    for note in melody[:5]:
        freq = normalize_frequency(note.get("freq"), polarity)
        try:
            duration = clamp(abs(float(note.get("duration", 0.45))), 0.1, 3.0)
        except (TypeError, ValueError):
            duration = 0.45
        normalized_notes.append(
            {
                "freq": freq,
                "duration": duration,
                "wave": normalize_wave(note.get("wave", fallback_wave)),
            }
        )
    return normalized_notes

def normalize_hex(hex_value: str) -> str:
    if not HEX_RE.match(hex_value or ""):
        raise ValueError(f"Codice HEX non valido: {hex_value}")
    normalized = hex_value if hex_value.startswith("#") else f"#{hex_value}"
    return normalized.upper()

def hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    clean = normalize_hex(hex_value).lstrip("#")
    return (
        int(clean[0:2], 16),
        int(clean[2:4], 16),
        int(clean[4:6], 16),
    )

def mix_hex_colors(hex_values: List[str]) -> str:
    rgbs = [hex_to_rgb(hex_value) for hex_value in hex_values]
    red = round(sum(rgb[0] for rgb in rgbs) / len(rgbs))
    green = round(sum(rgb[1] for rgb in rgbs) / len(rgbs))
    blue = round(sum(rgb[2] for rgb in rgbs) / len(rgbs))
    return f"#{red:02X}{green:02X}{blue:02X}"

def extract_goethe_quote(color_name: str, docs: List[str]) -> str | None:
    if not docs:
        return None

    target = color_name.lower()
    selected_doc = docs[0]
    for doc in docs:
        heading = doc.strip().splitlines()[0].replace("=", "").strip().lower()
        if target in heading or heading in target:
            selected_doc = doc
            break

    quote_lines = [
        line.strip()
        for line in selected_doc.splitlines()
        if line.strip() and not line.strip().startswith(("==", "#"))
    ]
    quote_text = " ".join(quote_lines).split(".")[0].strip()
    if not quote_text:
        return None
    return f"{quote_text}. - Goethe"

def get_collection():
    """Funzione che recupera il database in modo sicuro al momento del bisogno"""
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        return chroma_client.get_collection(name="goethe_colors")
    except Exception as e:
        print(f"\n[!] ERRORE INTERNO CHROMA DB: {e}\n")
        return None

# --- 2. INIZIALIZZAZIONE APP E CORS ---
app = FastAPI(
    title="Aisthesis API",
    version="1.0.0",
)

# IL CORS E' FONDAMENTALE PER NON AVERE "NETWORK ERROR" NEL FRONTEND
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.svg")

class SonifyRequest(BaseModel):
    text: str
    k: int = Field(default=1, ge=1, le=MAX_GOETHE_COLORS)

class RankedColor(BaseModel):
    rank: int
    name: str
    hex: str
    motivation: str

class SonifyLLMResponse(BaseModel):
    colors: List[RankedColor]
    freq: float
    wave: str
    quote: str
    artistic_description: str
    melody: List[Dict[str, Any]]

class SonifyResponse(BaseModel):
    hex: str
    mix_hex: str
    colors: List[RankedColor]
    freq: float
    wave: str
    quote: str
    artistic_description: str
    melody: List[Dict[str, Any]]

def generate_sonify_llm_response(prompt: str, requested_k: int) -> SonifyLLMResponse:
    last_error: Exception | None = None

    for retry in range(2):
        response_llm = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sei l'intelligenza artistica di Aisthesis. "
                        "Rispondi solo con JSON valido, senza markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "temperature": 0.2 if retry == 0 else 0.1,
                "num_predict": llm_num_predict(requested_k, retry),
                "num_ctx": 4096,
            },
        )

        json_finale = response_llm["message"]["content"]
        try:
            parsed = SonifyLLMResponse.model_validate_json(json_finale)
            if len(parsed.colors) < requested_k:
                raise ValueError(
                    f"Il motore LLM ha restituito {len(parsed.colors)} colori invece di {requested_k}."
                )
            return parsed
        except Exception as e:
            last_error = e
            print(f"\n[!] JSON LLM non valido, retry {retry + 1}/2: {e}", flush=True)

    raise last_error or ValueError("Il motore LLM non ha restituito JSON valido.")

@app.post("/sonify", response_model=SonifyResponse)
async def sonify_emotion(request: SonifyRequest) -> SonifyResponse:
    
    # PERCORSO 1: Il database non è raggiungibile (solleva eccezione, quindi non serve return)
    collection = get_collection()
    if collection is None:
         raise HTTPException(status_code=500, detail="Database vettoriale non trovato o bloccato.")

    stage = "inizializzazione"
    try:
        # 1. Estrazione Emozione
        stage = "estrazione emozione"
        risultato_nlp = parse_emotion(request.text)
        print(f"\n--- NUOVA RICHIESTA ---", flush=True)
        print(f"Emozione: {risultato_nlp.emozione} | Polarità: {risultato_nlp.polarita.value}", flush=True)

        # 2. Embedding della query
        stage = "embedding query"
        query_embedding = ollama.embeddings(
            model="nomic-embed-text", 
            prompt=risultato_nlp.search_query
        )["embedding"]

        # 3. Interrogazione di ChromaDB
        stage = "query ChromaDB"
        db_size = collection.count()
        print(f"Documenti totali presenti nel DB: {db_size}", flush=True)
        if db_size <= 0:
            raise HTTPException(status_code=500, detail="Database vettoriale vuoto.")

        n_results = db_size

        # Chiediamo a Chroma di restituirci anche le distanze (minore = più simile)
        risultati_ricerca = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances"] # <-- Aggiungi "distances" qui
        )
        
        # Stampiamo i punteggi per capire cosa sta succedendo
        print(f"\n--- DEBUG CHROMA DB ---", flush=True)
        print(f"Distanze calcolate: {risultati_ricerca.get('distances')}", flush=True)

        # 3.1 Estrazione sicura del contesto
        contesto_goethe = "Il colore giallo è caldo ed eccitante. Il blu è freddo e malinconico."
        docs_ordinati: List[str] = []
        docs = risultati_ricerca.get("documents")
        
        if docs is not None and len(docs) > 0 and docs[0] is not None and len(docs[0]) > 0:
            docs_colore = [doc for doc in docs[0] if is_goethe_color_doc(doc)]
            docs_ordinati = rank_docs_by_polarity(docs_colore, risultato_nlp.polarita.value)
            docs_contesto = limit_context_docs(docs_ordinati, request.k)
            contesto_goethe = "\n\n".join(docs_contesto)
            colori_candidati = "\n".join(
                f"- {extract_goethe_heading(doc)}"
                for doc in docs_contesto
            )
            print("\n--- CONTESTO RAG TROVATO ---", flush=True)
            print(contesto_goethe[:100] + "...", flush=True) # Stampa solo i primi 100 caratteri
        else:
            colori_candidati = "- Giallo\n- Blu"
            print("\n[!] Nessun contesto trovato. Uso fallback.", flush=True)

        if not docs_ordinati:
            raise HTTPException(status_code=500, detail="Nessun documento colore trovato nel database vettoriale.")

        effective_k = min(request.k, len(docs_ordinati), MAX_GOETHE_COLORS)

        # 4. Prompting
        prompt_sintesi = f"""Analizza i documenti RAG di Goethe e genera una sonificazione.

Emozione rilevata: {risultato_nlp.emozione}
Polarita goethiana: {risultato_nlp.polarita.value}
Testo originale: "{request.text}"

Documenti RAG recuperati da Chroma:
{contesto_goethe}

Colori candidati obbligatori:
{colori_candidati}

Regole:
- Usa esclusivamente i documenti RAG qui sopra come fonte Goethe.
- Per "name" usa solo i colori candidati obbligatori, senza inventarne altri.
- Scrivi in italiano.
- Restituisci JSON valido, nessun testo fuori dal JSON.
- "colors": esattamente {effective_k} colori unici, ordinati per pertinenza.
- Ogni colore: "rank" da 1 a {effective_k}, "name", "hex" in formato #RRGGBB, "motivation" breve.
- "freq": numero riferito solo al primo colore. MINUS 100-300, PLUS 400-800, NEUTRAL 250-500.
- "wave": una tra "sine", "square", "sawtooth", "triangle".
- "quote": frase breve dai documenti RAG con " - Goethe".
- "artistic_description": massimo 35 parole.
- "melody": 4 note, ognuna con "freq", "duration", "wave".

Schema:
{{
  "colors": [
    {{"rank": 1, "name": "Blu", "hex": "#0000FF", "motivation": "Motivo breve"}}
  ],
  "freq": 220.0,
  "wave": "sine",
  "quote": "Frase Goethe. - Goethe",
  "artistic_description": "Descrizione breve.",
  "melody": [
    {{"freq": 220.0, "duration": 0.4, "wave": "sine"}}
  ]
}}
"""
        stage = "generazione LLM"
        risposta_llm = generate_sonify_llm_response(prompt_sintesi, effective_k)

        # 6. Validazione
        stage = "validazione risposta"
        colori_validati = risposta_llm.colors[:effective_k]
        if not colori_validati:
            raise ValueError("Il motore LLM non ha restituito colori.")

        for index, color in enumerate(colori_validati, start=1):
            color.rank = index
            color.hex = normalize_hex(color.hex)

        risposta_llm.freq = normalize_frequency(risposta_llm.freq, risultato_nlp.polarita.value)
        risposta_llm.wave = normalize_wave(risposta_llm.wave)
        risposta_llm.melody = normalize_melody(
            risposta_llm.melody,
            risultato_nlp.polarita.value,
            risposta_llm.wave,
        )

        mix_hex = mix_hex_colors([color.hex for color in colori_validati])
        quote = extract_goethe_quote(colori_validati[0].name, docs_ordinati) or risposta_llm.quote
        risposta_validata = SonifyResponse(
            hex=mix_hex,
            mix_hex=mix_hex,
            colors=colori_validati,
            freq=risposta_llm.freq,
            wave=risposta_llm.wave,
            quote=quote,
            artistic_description=risposta_llm.artistic_description,
            melody=risposta_llm.melody,
        )
        
        # PERCORSO 2: Tutto è andato bene (Ritorna il SonifyResponse promesso)
        return risposta_validata

    except HTTPException:
        raise
    except Exception as e:
        # PERCORSO 3: Qualcosa è crashato durante il try. 
        # Invece di far finire la funzione e restituire None, solleviamo un errore esplicito.
        print(f"\n[!] ERRORE DURANTE LA GENERAZIONE ({stage}): {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Errore interno durante {stage}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Rimosso embedding('data\goethe.txt')
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
