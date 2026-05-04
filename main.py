"""
main.py — Backend FastAPI per il progetto Aisthesis
=====================================================
Architettura: FastAPI + pipeline RAG (da implementare) + LLM

Fase attuale: MOCK
L'endpoint /sonify restituisce dati fittizi per permettere lo sviluppo
e il test del frontend in modo completamente indipendente.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Inizializzazione dell'app ---
app = FastAPI(
    title="Aisthesis API",
    description="Sistema di sonificazione semantica basato sulla Teoria dei Colori di Goethe",
    version="0.1.0-mock",
)

# --- Mounting del frontend statico ---
# Tutti i file in ./static/ verranno serviti direttamente da FastAPI.
# index.html sarà raggiungibile su http://127.0.0.1:8000/
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# --- Modelli Pydantic ---
class SonifyRequest(BaseModel):
    """Schema della richiesta in ingresso: una stringa di testo emotivo."""
    text: str


class SonifyResponse(BaseModel):
    """Schema della risposta: colore HEX, parametri audio e citazione di Goethe."""
    hex: str
    freq: float
    wave: str
    quote: str


# ===========================================================================
# ENDPOINT PRINCIPALE
# ===========================================================================

@app.post("/sonify", response_model=SonifyResponse)
async def sonify_emotion(request: SonifyRequest) -> SonifyResponse:
    """
    Riceve un testo emotivo e restituisce il mapping colore + suono.

    TODO (Fase 2): Sostituire il blocco "MOCK" con la pipeline reale:
        1. Carica il VectorStore (Chroma) indicizzato da data/goethe.txt
        2. Esegui una query RAG per recuperare i passaggi più rilevanti
        3. Invia il testo + contesto a un LLM (es. GPT-4o) con un prompt
           strutturato per ottenere: hex, freq, wave, quote
        4. Valida la risposta del LLM con Pydantic e restituiscila
    """

    # -----------------------------------------------------------------------
    # BLOCCO MOCK — da rimuovere nella Fase 2
    # -----------------------------------------------------------------------
    mock_response = SonifyResponse(
        hex="#FF8C00",
        freq=330.0,
        wave="square",
        quote=(
            "L'arancio dà allo spettatore un senso di calore e beatitudine, "
            "come la luce del sole al tramonto. Stimola l'attività e la vitalità, "
            "ma può diventare opprimente nella sua intensità. — Goethe, Farbenlehre"
        ),
    )
    return mock_response
    # -----------------------------------------------------------------------


# --- Entry point per lo sviluppo locale ---
# Esegui con: python main.py  oppure  uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
