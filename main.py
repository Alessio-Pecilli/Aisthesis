"""
Aisthesis — Backend FastAPI
Pipeline: classificazione emozionale (Transformers, encoder-only) → mapping cromatico
Goethe HSV O(1) → parametri sonori deterministici (psychoacoustic avanzato).

Nessuna dipendenza da LLM generativi, agenti o RAG: solo `transformers` per FEEL-IT.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast,
    pipeline,
)

# ---------------------------------------------------------------------------
# Costanti — soglia emozione e psychoacoustic
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.35
BASE_FREQ_HZ = 110.0  # Abbassato a 110Hz (A2) per dare più corpo ai bassi

# Cutoff LPF: esponenziale S ∈ [0,100] → [150 Hz, 12000 Hz]
FILTER_CUTOFF_BASE_HZ = 150.0
FILTER_CUTOFF_EXP_BASE = 80.0

# Attacco: V=100 → percussivo (5 ms), V=0 → ambient (3000 ms)
ATTACK_FAST_MS = 5.0
ATTACK_SLOW_MS = 3000.0

# Riverbero/Release: V basso → cavernoso/lungo; V alto → asciutto/corto
REVERB_WET_MAX = 0.9

EmotionName = Literal["joy", "sadness", "anger", "fear", "neutral"]
WaveformName = Literal["sine", "triangle", "sawtooth"]

# Mappatura HSV (Hue 0–360, S e V 0–100)
# S < 45: sine, S < 75: triangle, else: sawtooth
GOETHE_DATA: dict[str, dict[str, Any]] = {
    "joy": {
        "hsv": (60.0, 40.0, 100.0), # Giallo: Sine, Attacco Rapido, Brillante
        "name": "Giallo (Yellow)",
        "quote": "Il colore più vicino alla luce... possiede un carattere sereno, lieto, dolcemente eccitante.",
        "description": "L'occhio ne viene allietato, l'animo si rasserena; un calore immediato ci investe.",
    },
    "anger": {
        "hsv": (0.0, 100.0, 100.0), # Vermiglio: Sawtooth, Attacco Percussivo, Aggressivo
        "name": "Giallo-Rosso (Vermiglio)",
        "quote": "Il culmine del lato attivo... spinge all'azione; è il colore del fuoco e della passione.",
        "description": "Esercita un fascino incredibile che spinge all'azione; è l'energia vitale al suo apice.",
    },
    "sadness": {
        "hsv": (240.0, 30.0, 40.0), # Blu: Sine, Attacco Lento, Profondo/Scuro
        "name": "Blu (Blue)",
        "quote": "Il colore più vicino all'oscurità... crea una sensazione di vuoto e di distanza.",
        "description": "Mentre il giallo porta luce, il blu sembra attirare lo sguardo verso l'infinito, lasciando un'impressione di solitudine.",
    },
    "fear": {
        "hsv": (280.0, 60.0, 30.0), # Violetto: Triangle, Attacco Lento, Inquieto
        "name": "Rosso-Blu (Violetto)",
        "quote": "Rispetto al blu puro, questo colore appare più inquieto ed oppressivo.",
        "description": "Evoca una tristezza che tende all'oppressione; un colore che ha qualcosa di fastidioso.",
    },
    "neutral": {
        "hsv": (120.0, 60.0, 80.0), # Verde: Triangle, Attacco Morbido, Equilibrato
        "name": "Verde (Green)",
        "quote": "Il prodotto dell'unione tra Giallo e Blu in perfetto equilibrio... un punto di sosta perfetto.",
        "description": "L'occhio e l'animo trovano in questo colore un riposo reale. Non vuole né può andare oltre.",
    },
}

LABEL_ALIASES: dict[str, EmotionName] = {
    "joy": "joy",
    "gioia": "joy",
    "anger": "anger",
    "rabbia": "anger",
    "sadness": "sadness",
    "tristezza": "sadness",
    "fear": "fear",
    "paura": "fear",
}

logger = logging.getLogger("aisthesis")
logging.basicConfig(level=logging.INFO)

_classifier: Optional[Any] = None
MODEL_ID = "MilaNLProc/feel-it-italian-emotion"


def load_classifier() -> Any:
    """
    Carica modello e tokenizer una sola volta.
    `PreTrainedTokenizerFast` evita problemi noti del tokenizer CamemBERT lento
    con alcune combinazioni Python / sentencepiece.
    """
    device = 0 if torch.cuda.is_available() else -1
    logger.info("Inizializzazione pipeline su device: %s", device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier
    logger.info("Caricamento modello %s …", MODEL_ID)
    try:
        _classifier = load_classifier()
        logger.info("Modello pronto.")
    except Exception as e:
        logger.error("Errore critico caricamento modello: %s", e)
        raise
    yield
    _classifier = None


app = FastAPI(
    title="Aisthesis API",
    version="3.2.0",
    lifespan=lifespan,
)

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


# ---------------------------------------------------------------------------
# Schemi Pydantic
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class SemanticAnalysis(BaseModel):
    emotion: EmotionName
    confidence: float = Field(..., ge=0.0, le=1.0)


class HSVState(BaseModel):
    h: float = Field(..., ge=0.0, le=360.0)
    s: float = Field(..., ge=0.0, le=100.0)
    v: float = Field(..., ge=0.0, le=100.0)


class VisualState(BaseModel):
    color_name: str
    hsv: HSVState
    goethe_quote: str
    artistic_description: str


class AudioTargetState(BaseModel):
    """Parametri per motore di sintesi (SuperCollider, Web Audio, ecc.)."""

    waveform: WaveformName
    pitch_hz: float = Field(..., gt=0.0)
    filter_cutoff_hz: float = Field(..., gt=0.0)
    amplitude: float = Field(..., ge=0.0, le=1.0)
    attack_time_ms: float = Field(..., gt=0.0)
    reverb_mix: float = Field(..., ge=0.0, le=1.0)


class ProcessResponse(BaseModel):
    input_text: str
    semantic_analysis: SemanticAnalysis
    visual_state: VisualState
    audio_target_state: AudioTargetState


# ---------------------------------------------------------------------------
# Dominio: emozione → HSV → audio
# ---------------------------------------------------------------------------


def _normalize_label(raw: str) -> str:
    return raw.strip().lower()


def map_model_label_to_emotion(label: str) -> EmotionName:
    key = _normalize_label(label)
    if key.startswith("label_"):
        key = key.replace("label_", "", 1)
    
    # Check mapping
    mapped = LABEL_ALIASES.get(key)
    if mapped:
        return mapped
    
    # Substring match for resilience
    for alias, emotion in LABEL_ALIASES.items():
        if alias in key:
            return emotion
            
    logger.warning("Etichetta modello non mappata: %r (key=%r) → neutral", label, key)
    return "neutral"


def run_emotion_inference(text: str, clf: Any) -> tuple[EmotionName, float]:
    """Top-1 FEEL-IT; score = softmax sulla classe predetta."""
    out = clf(text, top_k=4, truncation=True, max_length=512)
    
    # Se out è una lista nested (alcune versioni di transformers)
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]
        
    logger.info("Raw Inference Output: %s", out)
    
    if not out:
        return "neutral", 0.0
        
    best = max(out, key=lambda x: float(x.get("score", 0.0)))
    raw_label = str(best.get("label", ""))
    score = float(best.get("score", 0.0))
    
    emotion = map_model_label_to_emotion(raw_label)
    
    if score < CONFIDENCE_THRESHOLD:
        logger.info("Confidence %.2f below threshold %.2f -> neutral", score, CONFIDENCE_THRESHOLD)
        emotion = "neutral"
        
    return emotion, score


def goethe_data_for_emotion(emotion: EmotionName) -> dict[str, Any]:
    return GOETHE_DATA[emotion]


def waveform_from_saturation(s: float) -> WaveformName:
    """Complessità armonica crescente con la saturazione cromatica."""
    if s < 45.0:
        return "sine"
    if s < 75.0:
        return "triangle"
    return "sawtooth"


def hsv_to_audio_target(h: float, s: float, v: float) -> dict[str, Any]:
    """
    Mapping deterministico HSV → parametri sonori (ampia differenziazione).
    """
    s_clamped = max(0.0, min(100.0, s))
    v_clamped = max(0.0, min(100.0, v))

    pitch_hz = BASE_FREQ_HZ * (2.0 ** (h / 360.0))
    waveform = waveform_from_saturation(s_clamped)

    # S=0 → 200 Hz; S=100 → 8000 Hz (crescita esponenziale)
    filter_cutoff_hz = FILTER_CUTOFF_BASE_HZ * (
        FILTER_CUTOFF_EXP_BASE ** (s_clamped / 100.0)
    )

    amplitude = v_clamped / 100.0

    # V basso → lento; V alto → rapido
    t_v = v_clamped / 100.0
    attack_time_ms = ATTACK_SLOW_MS + (ATTACK_FAST_MS - ATTACK_SLOW_MS) * t_v

    # V basso → più riverbero (suono lontano / vuoto)
    reverb_mix = REVERB_WET_MAX * (1.0 - t_v)

    return {
        "waveform": waveform,
        "pitch_hz": round(pitch_hz, 4),
        "filter_cutoff_hz": round(filter_cutoff_hz, 4),
        "amplitude": round(amplitude, 4),
        "attack_time_ms": round(attack_time_ms, 4),
        "reverb_mix": round(reverb_mix, 4),
    }


def build_process_response(text: str, emotion: EmotionName, confidence: float) -> ProcessResponse:
    data = goethe_data_for_emotion(emotion)
    h, s, v = data["hsv"]
    audio = hsv_to_audio_target(h, s, v)
    return ProcessResponse(
        input_text=text,
        semantic_analysis=SemanticAnalysis(
            emotion=emotion, confidence=round(float(confidence), 2)
        ),
        visual_state=VisualState(
            color_name=data["name"],
            hsv=HSVState(h=h, s=s, v=v),
            goethe_quote=data["quote"],
            artistic_description=data["description"],
        ),
        audio_target_state=AudioTargetState(
            waveform=audio["waveform"],
            pitch_hz=audio["pitch_hz"],
            filter_cutoff_hz=audio["filter_cutoff_hz"],
            amplitude=audio["amplitude"],
            attack_time_ms=audio["attack_time_ms"],
            reverb_mix=audio["reverb_mix"],
        ),
    )


@app.post("/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest) -> ProcessResponse:
    """
    POST `{ "text": "..." }` → emozione, stato cromatico Goethe, parametri audio avanzati.
    """
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Modello non ancora caricato.")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail='Il campo "text" non può essere vuoto.')

    try:
        emotion, confidence = run_emotion_inference(text, _classifier)
    except Exception as exc:
        logger.exception("Errore inferenza: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Errore durante la classificazione emotiva.",
        ) from exc

    payload = build_process_response(text, emotion, confidence)
    logger.info(
        "process: emotion=%s conf=%.2f waveform=%s pitch=%.1fHz reverb=%.2f len=%d",
        emotion,
        confidence,
        payload.audio_target_state.waveform,
        payload.audio_target_state.pitch_hz,
        payload.audio_target_state.reverb_mix,
        len(text),
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
