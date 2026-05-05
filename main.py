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
    "== GIALLO ==",
    "== ROSSO-GIALLO",
    "== ROSSO ==",
    "== GIALLO-ROSSO",
)
MINUS_SECTIONS = (
    "== BLU ==",
    "== AZZURRO ==",
    "== VIOLETTO ==",
)
NEUTRAL_SECTIONS = ("== VERDE ==",)

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
    k: int = Field(default=1, ge=1, le=8)

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

@app.post("/sonify", response_model=SonifyResponse)
async def sonify_emotion(request: SonifyRequest) -> SonifyResponse:
    
    # PERCORSO 1: Il database non è raggiungibile (solleva eccezione, quindi non serve return)
    collection = get_collection()
    if collection is None:
         raise HTTPException(status_code=500, detail="Database vettoriale non trovato o bloccato.")

    try:
        # 1. Estrazione Emozione
        risultato_nlp = parse_emotion(request.text)
        print(f"\n--- NUOVA RICHIESTA ---", flush=True)
        print(f"Emozione: {risultato_nlp.emozione} | Polarità: {risultato_nlp.polarita.value}", flush=True)

        # 2. Embedding della query
        query_embedding = ollama.embeddings(
            model="nomic-embed-text", 
            prompt=risultato_nlp.search_query
        )["embedding"]

        # 3. Interrogazione di ChromaDB
        db_size = collection.count()
        print(f"Documenti totali presenti nel DB: {db_size}", flush=True)
        if db_size <= 0:
            raise HTTPException(status_code=500, detail="Database vettoriale vuoto.")

        n_results = min(db_size, max(request.k, 8))

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
            docs_ordinati = rank_docs_by_polarity(docs[0], risultato_nlp.polarita.value)
            contesto_goethe = "\n\n".join(docs_ordinati)
            print("\n--- CONTESTO RAG TROVATO ---", flush=True)
            print(contesto_goethe[:100] + "...", flush=True) # Stampa solo i primi 100 caratteri
        else:
            print("\n[!] Nessun contesto trovato. Uso fallback.", flush=True)

        # 4. Prompting
        prompt_sintesi = f"""Sei l'intelligenza artistica di Aisthesis.
L'utente sta provando questa emozione: {risultato_nlp.emozione} (Polarità: {risultato_nlp.polarita.value}).
Testo originale: "{request.text}"

Basandoti ESCLUSIVAMENTE sui seguenti estratti della Teoria dei Colori di Goethe:
---
{contesto_goethe}
---

Genera l'output.
Regole:
- Scrivi tutti i campi testuali in italiano.
- 'colors': array di ESATTAMENTE {request.k} colori unici, ordinati dal piu attinente al meno attinente.
- Ogni elemento di 'colors' deve avere: 'rank' progressivo da 1 a {request.k}, 'name', 'hex' e 'motivation'.
- 'name' deve usare il nome italiano del colore o della sezione goethiana.
- 'motivation' deve spiegare brevemente perche quel colore e pertinente rispetto all'emozione e al testo di Goethe.
- 'freq': float riferito SOLO al colore in posizione 1. Bassi (100-300) per polarita MINUS, alti (400-800) per PLUS.
- 'wave': riferita SOLO al colore in posizione 1; scegli tra 'sine', 'square', 'sawtooth', 'triangle'.
- 'quote': copia una frase breve presente negli estratti e pertinente al colore in posizione 1, aggiungendo " - Goethe". Non citare il testo originale dell'utente.
- 'artistic_description': descrizione poetica.
- 'melody': array di 4-5 note ('freq', 'duration', 'wave'), riferita SOLO al colore in posizione 1.
"""

        # 5. Generazione LLM
        response_llm = ollama.chat(
            model="llama3",
            messages=[{'role': 'system', 'content': prompt_sintesi}],
            format=SonifyLLMResponse.model_json_schema(),
            options={'temperature': 0.3, 'num_predict': 900}
        )
        
        # 6. Validazione
        json_finale = response_llm['message']['content']
        risposta_llm = SonifyLLMResponse.model_validate_json(json_finale)
        colori_validati = risposta_llm.colors[:request.k]
        if not colori_validati:
            raise ValueError("Il motore LLM non ha restituito colori.")

        for index, color in enumerate(colori_validati, start=1):
            color.rank = index
            color.hex = normalize_hex(color.hex)

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

    except Exception as e:
        # PERCORSO 3: Qualcosa è crashato durante il try. 
        # Invece di far finire la funzione e restituire None, solleviamo un errore esplicito.
        print(f"\n[!] ERRORE DURANTE LA GENERAZIONE: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Errore interno del motore LLM: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Rimosso embedding('data\goethe.txt')
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
