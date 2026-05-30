"""
Aisthesis — Backend FastAPI
Pipeline: classificazione emozionale (Transformers, encoder-only) → mapping cromatico
Goethe HSV O(1) → parametri sonori deterministici (psychoacoustic avanzato).

Nessuna dipendenza da LLM generativi, agenti o RAG: solo `transformers` per FEEL-IT.
"""

from __future__ import annotations

import logging
import re
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
EXPLANATION_TOP_TERMS = 5
EXPLANATION_MIN_DELTA = 0.01

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

DISPLAY_STOPWORDS = {
    "a",
    "ad",
    "al",
    "alla",
    "con",
    "da",
    "del",
    "della",
    "di",
    "e",
    "ha",
    "ho",
    "il",
    "in",
    "io",
    "la",
    "le",
    "lo",
    "ma",
    "mi",
    "ne",
    "nel",
    "non",
    "o",
    "per",
    "se",
    "si",
    "sono",
    "su",
    "tra",
    "tu",
    "un",
    "una",
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
    raw_emotion: EmotionName
    threshold_applied: bool


class EmotionScore(BaseModel):
    emotion: EmotionName
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class InfluentialTerm(BaseModel):
    term: str
    importance: float = Field(..., ge=0.0, le=1.0)


class ExplainabilityState(BaseModel):
    method: str
    decision_summary: str
    top_classes: list[EmotionScore]
    influential_terms: list[InfluentialTerm]
    rule_trace: list[str]


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
    explainability: ExplainabilityState
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


def _run_classifier(text: str, clf: Any, top_k: int = 4) -> list[dict[str, Any]]:
    out = clf(text, top_k=top_k, truncation=True, max_length=512)
    
    # Se out è una lista nested (alcune versioni di transformers)
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]

    logger.info("Raw Inference Output: %s", out)
    if not out:
        return []
    return list(out)


def _build_top_classes(out: list[dict[str, Any]]) -> list[EmotionScore]:
    classes: list[EmotionScore] = []
    for item in sorted(out, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        label = str(item.get("label", ""))
        classes.append(
            EmotionScore(
                emotion=map_model_label_to_emotion(label),
                label=label,
                score=round(float(item.get("score", 0.0)), 4),
            )
        )
    return classes


def _score_for_emotion(out: list[dict[str, Any]], emotion: EmotionName) -> float:
    for item in out:
        label = str(item.get("label", ""))
        if map_model_label_to_emotion(label) == emotion:
            return float(item.get("score", 0.0))
    return 0.0


def _model_emotion_indices(model: Any) -> dict[EmotionName, int]:
    mapping: dict[EmotionName, int] = {}
    id2label = getattr(model.config, "id2label", {})
    for idx, raw_label in id2label.items():
        emotion = map_model_label_to_emotion(str(raw_label))
        mapping[emotion] = int(idx)
    return mapping


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\b[\wÀ-ÖØ-öø-ÿ'-]+\b", text, flags=re.UNICODE):
        term = match.group(0).strip("'-")
        if len(term) < 2:
            continue
        spans.append((match.start(), match.end(), term))
    return spans


def _is_display_term(term: str) -> bool:
    normalized = term.strip("'-").lower()
    return len(normalized) >= 3 and normalized not in DISPLAY_STOPWORDS


def explain_prediction(
    text: str, clf: Any, raw_emotion: EmotionName, raw_score: float
) -> list[InfluentialTerm]:
    """
    Spiegazione locale gradient-based:
    misura la salienza dei token rispetto al logit della classe predetta
    e aggrega i contributi sui segmenti di parola.
    """
    if raw_emotion == "neutral" or raw_score <= 0.0:
        return []

    tokenizer = clf.tokenizer
    model = clf.model
    if tokenizer is None or model is None:
        return []

    emotion_indices = _model_emotion_indices(model)
    target_index = emotion_indices.get(raw_emotion)
    if target_index is None:
        logger.warning("Classe %s non trovata in id2label del modello.", raw_emotion)
        return []

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    word_scores: dict[tuple[int, int, str], float] = {
        span: 0.0 for span in _word_spans(text)
    }
    if not word_scores:
        return []

    special_tokens_mask = tokenizer.get_special_tokens_mask(
        encoded["input_ids"][0].tolist(),
        already_has_special_tokens=True,
    )

    model_device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)

    model.zero_grad(set_to_none=True)
    embeddings = model.get_input_embeddings()(input_ids).detach()
    embeddings.requires_grad_(True)
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    target_logit = outputs.logits[0, target_index]
    target_logit.backward()

    gradients = embeddings.grad
    if gradients is None:
        return []

    token_scores = (
        (embeddings[0] * gradients[0]).sum(dim=-1).abs().detach().cpu().tolist()
    )

    for token_score, (start, end), is_special in zip(token_scores, offsets, special_tokens_mask):
        if is_special or start == end or token_score <= 0.0:
            continue

        for span in word_scores:
            span_start, span_end, _ = span
            if start >= span_start and end <= span_end:
                word_scores[span] += float(token_score)
                break

    max_score = max(word_scores.values(), default=0.0)
    if max_score <= 0.0:
        return []

    influential_terms = [
        InfluentialTerm(term=term, importance=round(score / max_score, 4))
        for (_, _, term), score in word_scores.items()
        if (score / max_score) >= EXPLANATION_MIN_DELTA and _is_display_term(term)
    ]
    influential_terms.sort(key=lambda item: item.importance, reverse=True)
    return influential_terms[:EXPLANATION_TOP_TERMS]


def run_emotion_inference(text: str, clf: Any) -> dict[str, Any]:
    """Top-1 FEEL-IT con tracciato completo per explainability."""
    out = _run_classifier(text, clf)
    top_classes = _build_top_classes(out)

    if not out:
        return {
            "emotion": "neutral",
            "raw_emotion": "neutral",
            "confidence": 0.0,
            "threshold_applied": False,
            "top_classes": top_classes,
            "influential_terms": [],
        }

    best = max(out, key=lambda x: float(x.get("score", 0.0)))
    raw_label = str(best.get("label", ""))
    score = float(best.get("score", 0.0))
    raw_emotion = map_model_label_to_emotion(raw_label)
    emotion = raw_emotion
    threshold_applied = False

    if score < CONFIDENCE_THRESHOLD:
        logger.info(
            "Confidence %.2f below threshold %.2f -> neutral",
            score,
            CONFIDENCE_THRESHOLD,
        )
        emotion = "neutral"
        threshold_applied = True

    return {
        "emotion": emotion,
        "raw_emotion": raw_emotion,
        "confidence": score,
        "threshold_applied": threshold_applied,
        "top_classes": top_classes,
        "influential_terms": explain_prediction(text, clf, raw_emotion, score),
    }


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


def build_decision_summary(
    emotion: EmotionName,
    raw_emotion: EmotionName,
    confidence: float,
    threshold_applied: bool,
    influential_terms: list[InfluentialTerm],
) -> str:
    if threshold_applied:
        summary = (
            f"Il modello ha favorito '{raw_emotion}' con score {confidence:.2f}, "
            f"ma il valore è sotto la soglia {CONFIDENCE_THRESHOLD:.2f}: "
            f"il sistema restituisce quindi 'neutral'."
        )
    else:
        summary = (
            f"Il modello assegna il punteggio più alto a '{emotion}' "
            f"con score {confidence:.2f}."
        )

    if influential_terms:
        terms = ", ".join(term.term for term in influential_terms[:3])
        return f"{summary} Le parole più rilevanti per questa decisione sono: {terms}."

    return (
        f"{summary} Nessuna parola singola ha mostrato un impatto forte: "
        "la decisione sembra dipendere dal contesto complessivo."
    )


def build_rule_trace(
    emotion: EmotionName,
    color_name: str,
    h: float,
    s: float,
    v: float,
    audio: dict[str, Any],
) -> list[str]:
    return [
        f"L'emozione finale '{emotion}' viene mappata sul colore '{color_name}' con HSV ({h:.0f}, {s:.0f}, {v:.0f}).",
        f"La tonalità H={h:.0f} controlla l'altezza: pitch {audio['pitch_hz']:.1f} Hz.",
        f"La saturazione S={s:.0f} determina il timbro: forma d'onda {audio['waveform']} e cutoff {audio['filter_cutoff_hz']:.1f} Hz.",
        f"Il valore V={v:.0f} regola energia e spazio: ampiezza {audio['amplitude']:.2f}, attacco {audio['attack_time_ms']:.1f} ms, riverbero {audio['reverb_mix']:.2f}.",
    ]


def build_process_response(text: str, inference: dict[str, Any]) -> ProcessResponse:
    emotion = inference["emotion"]
    raw_emotion = inference["raw_emotion"]
    confidence = inference["confidence"]
    threshold_applied = inference["threshold_applied"]
    top_classes = inference["top_classes"]
    influential_terms = inference["influential_terms"]

    data = goethe_data_for_emotion(emotion)
    h, s, v = data["hsv"]
    audio = hsv_to_audio_target(h, s, v)
    return ProcessResponse(
        input_text=text,
        semantic_analysis=SemanticAnalysis(
            emotion=emotion,
            confidence=round(float(confidence), 2),
            raw_emotion=raw_emotion,
            threshold_applied=threshold_applied,
        ),
        explainability=ExplainabilityState(
            method="gradient-based token saliency aggregated by word spans",
            decision_summary=build_decision_summary(
                emotion=emotion,
                raw_emotion=raw_emotion,
                confidence=confidence,
                threshold_applied=threshold_applied,
                influential_terms=influential_terms,
            ),
            top_classes=top_classes,
            influential_terms=influential_terms,
            rule_trace=build_rule_trace(
                emotion=emotion,
                color_name=data["name"],
                h=h,
                s=s,
                v=v,
                audio=audio,
            ),
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
        inference = run_emotion_inference(text, _classifier)
    except Exception as exc:
        logger.exception("Errore inferenza: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Errore durante la classificazione emotiva.",
        ) from exc

    payload = build_process_response(text, inference)
    logger.info(
        "process: emotion=%s conf=%.2f waveform=%s pitch=%.1fHz reverb=%.2f len=%d",
        payload.semantic_analysis.emotion,
        payload.semantic_analysis.confidence,
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
