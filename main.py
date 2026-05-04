import os
import chromadb
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

from goetheColor import parse_emotion

# --- 1. SETUP DEL DATABASE (Percorsi Assoluti per evitare bug di Uvicorn) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

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

class SonifyResponse(BaseModel):
    hex: str
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

        # Chiediamo a Chroma di restituirci anche le distanze (minore = più simile)
        risultati_ricerca = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "distances"] # <-- Aggiungi "distances" qui
        )
        
        # Stampiamo i punteggi per capire cosa sta succedendo
        print(f"\n--- DEBUG CHROMA DB ---", flush=True)
        print(f"Distanze calcolate: {risultati_ricerca.get('distances')}", flush=True)

        # 3.1 Estrazione sicura del contesto
        contesto_goethe = "Il colore giallo è caldo ed eccitante. Il blu è freddo e malinconico."
        docs = risultati_ricerca.get("documents")
        
        if docs is not None and len(docs) > 0 and docs[0] is not None and len(docs[0]) > 0:
            contesto_goethe = "\n\n".join(docs[0])
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
- 'hex': codice HEX.
- 'freq': float. Bassi (100-300) per polarità MINUS, alti (400-800) per PLUS.
- 'wave': scegli tra 'sine', 'square', 'sawtooth', 'triangle'.
- 'quote': estrai una citazione pertinente, aggiungendo " — Goethe".
- 'artistic_description': descrizione poetica.
- 'melody': array di 4-5 note ('freq', 'duration', 'wave').
"""

        # 5. Generazione LLM
        response_llm = ollama.chat(
            model="llama3",
            messages=[{'role': 'system', 'content': prompt_sintesi}],
            format=SonifyResponse.model_json_schema(),
            options={'temperature': 0.3, 'num_predict': 500}
        )
        
        # 6. Validazione
        json_finale = response_llm['message']['content']
        risposta_validata = SonifyResponse.model_validate_json(json_finale)
        
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