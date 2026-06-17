"""
Aisthesis — Backend FastAPI v4
Pipeline: FEEL-IT (top-k) → spazio affettivo (valence/arousal) → blend cromatico Goethe
→ profilo sonoro musicale → explainability (Integrated Gradients + contrastivo).
"""

from __future__ import annotations

import logging
import math
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
# Costanti
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.30
TOP_K_BLEND = 4
MIN_BLEND_WEIGHT = 0.05
MODEL_BLEND_WEIGHT = 0.50
LEXICON_BLEND_WEIGHT = 0.50
EXPLANATION_TOP_TERMS = 6
EXPLANATION_MIN_DELTA = 0.008
IG_STEPS = 20

EmotionName = Literal["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation", "love", "awe", "neutral"]
WaveformName = Literal["sine", "triangle", "sawtooth", "square"]
ScaleName = Literal["pentatonic_major", "pentatonic_minor", "dorian", "mixolydian", "phrygian", "lydian"]

# ---------------------------------------------------------------------------
# Mappature & Costanti (Aesthetics & Semantics)
# ---------------------------------------------------------------------------

SCALES: dict[ScaleName, list[int]] = {
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}

# Russell circumplex — coordinate normalizzate [-1, 1]
EMOTION_VA: dict[EmotionName, tuple[float, float]] = {
    "joy": (0.82, 0.62),
    "anger": (-0.72, 0.88),
    "sadness": (-0.78, 0.22),
    "fear": (-0.65, 0.78),
    "surprise": (0.20, 0.85),
    "disgust": (-0.60, 0.40),
    "trust": (0.60, -0.20),
    "anticipation": (0.20, 0.60),
    "love": (0.80, 0.30),
    "awe": (0.50, 0.70),
    "neutral": (0.05, 0.30),
}

SCALE_INTERVALS: dict[ScaleName, list[int]] = {
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}

GOETHE_DATA: dict[str, dict[str, Any]] = {
    "joy": {
        "hsv": (52.0, 55.0, 95.0),
        "name": "Giallo",
        "quote": "Il colore più vicino alla luce possiede un carattere sereno, lieto, dolcemente eccitante.",
        "description": "L'occhio ne viene allietato, l'animo si rasserena; un calore immediato ci investe.",
    },
    "anger": {
        "hsv": (8.0, 92.0, 92.0),
        "name": "Vermiglio",
        "quote": "Il culmine del lato attivo spinge all'azione; è il colore del fuoco e della passione.",
        "description": "Esercita un fascino incredibile che spinge all'azione; energia vitale al suo apice.",
    },
    "sadness": {
        "hsv": (225.0, 42.0, 38.0),
        "name": "Blu",
        "quote": "Il colore più vicino all'oscurità crea una sensazione di vuoto e di distanza.",
        "description": "Attira lo sguardo verso l'infinito, lasciando un'impressione di solitudine.",
    },
    "fear": {
        "hsv": (275.0, 55.0, 32.0),
        "name": "Violetto",
        "quote": "Rispetto al blu puro, questo colore appare più inquieto ed oppressivo.",
        "description": "Evoca una tristezza che tende all'oppressione; qualcosa di fastidioso nell'aria.",
    },
    "surprise": {
        "hsv": (30.0, 85.0, 95.0),
        "name": "Arancione",
        "quote": "Un colore pieno di energia che risveglia improvvisamente i sensi.",
        "description": "L'intensità dell'arancione stimola l'attenzione e cattura lo sguardo in un lampo.",
    },
    "disgust": {
        "hsv": (70.0, 60.0, 50.0),
        "name": "Verde Marcio",
        "quote": "Una mescolanza sgradevole che allontana e altera l'equilibrio.",
        "description": "Ispira un senso di repulsione e rifiuto, un tono che l'occhio cerca di evitare.",
    },
    "trust": {
        "hsv": (195.0, 45.0, 90.0),
        "name": "Azzurro",
        "quote": "Ci attira verso di sé, donando un senso di vastità e di riposo.",
        "description": "Una tonalità chiara e pacifica che infonde sicurezza e tranquillità.",
    },
    "anticipation": {
        "hsv": (345.0, 80.0, 75.0),
        "name": "Cremisi",
        "quote": "L'oscurarsi verso il rosso incute gravità e una sensazione di imminenza.",
        "description": "Densità cromatica che suggerisce qualcosa che sta per compiersi.",
    },
    "love": {
        "hsv": (330.0, 35.0, 95.0),
        "name": "Rosa",
        "quote": "Un riverbero di luce calda che evoca pura grazia e tenerezza giovanile.",
        "description": "Un'estensione addolcita del rosso che cinge in un abbraccio delicato.",
    },
    "awe": {
        "hsv": (290.0, 70.0, 60.0),
        "name": "Porpora",
        "quote": "Il colore della massima dignità, unisce in sé la più grande potenza e maestà.",
        "description": "Evoca un timore reverenziale e un profondo senso del sublime.",
    },
    "neutral": {
        "hsv": (128.0, 38.0, 72.0),
        "name": "Verde",
        "quote": "Il prodotto dell'unione tra Giallo e Blu in perfetto equilibrio.",
        "description": "L'occhio e l'animo trovano in questo colore un riposo reale.",
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
    "surprise": "surprise",
    "sorpresa": "surprise",
    "disgust": "disgust",
    "disgusto": "disgust",
    "trust": "trust",
    "fiducia": "trust",
    "anticipation": "anticipation",
    "attesa": "anticipation",
    "love": "love",
    "amore": "love",
    "awe": "awe",
    "ammirazione": "awe",
}

DISPLAY_STOPWORDS = {
    "a", "ad", "al", "alla", "con", "da", "del", "della", "di", "e", "ha", "ho",
    "il", "in", "io", "la", "le", "lo", "ma", "mi", "ne", "nel", "non", "o",
    "per", "se", "si", "sono", "su", "tra", "tu", "un", "una",
}

GENERIC_WORDS = {
    "sento", "sentire", "provo", "provare", "sono", "essere", "ho", "avere", 
    "faccio", "fare", "dico", "dire", "vedo", "vedere", "oggi", "ieri", "domani", 
    "qui", "lì", "ora", "adesso", "molto", "poco", "tanto", "troppo", "tutto", 
    "niente", "nessuno", "qualcuno", "qualcosa", "sempre", "mai", "forse", "come", 
    "perché", "quando", "dove", "chi", "che", "cui", "quale", "quali", "quanto", 
    "quanti", "sto", "stare", "stiamo", "state", "stanno", "sei", "siamo", "siete", "era", "ero", "erano"
}

EMOTION_IT: dict[EmotionName, str] = {
    "joy": "Gioia",
    "sadness": "Tristezza",
    "anger": "Rabbia",
    "fear": "Paura",
    "surprise": "Sorpresa",
    "disgust": "Disgusto",
    "trust": "Fiducia",
    "anticipation": "Attesa",
    "love": "Amore",
    "awe": "Ammirazione",
    "neutral": "Equilibrio",
}

# Lessico affettivo italiano — corregge errori del modello su testi brevi/espliciti
EMOTION_LEXICON: dict[EmotionName, tuple[str, ...]] = {
    "anger": (
        "arrabbiato", "arrabiato", "arrabbiata", "arrabiata", "rabbia", "rabbi",
        "furioso", "furiosa", "furiosi", "ira", "collera", "uffa", "incazzato",
        "incazzata", "odio", "furia", "rabbioso", "rabbiosa", "infuriato",
        "infuriata", "stizzito", "stizzita", "indignato", "indignata",
        "nervoso", "nervosa", "frustrato", "frustrata", "vendetta", "rancore", 
        "ostile", "ostilità", "aggressivo",
    ),
    "sadness": (
        "triste", "tristezza", "tristi", "piango", "piangere", "piange", "pianto",
        "lacrima", "lacrime", "malinconia", "malinconico", "dolore", "sofferenza",
        "depresso", "depressa", "vuoto", "solitudine", "solo", "sola", "abbattuto",
        "abbattuta", "disperato", "disperata", "lutto",
        "stanco", "stanca", "deluso", "delusa", "annoiato", "annoiata", "noia",
        "nostalgia", "rimpianto", "fragile", "rassegnato", "rassegnata",
    ),
    "joy": (
        "gioia", "gioioso", "gioiosa", "felice", "felicità", "allegro", "allegra",
        "lieto", "lieta", "contento", "contenta", "sereno", "serena", "entusiasta",
        "euforia", "splendido", "meraviglioso",
        "fortunato", "fortunata", "speranza", "grato", "grata",
        "fiero", "fiera", "orgoglioso", "orgogliosa", "ottimista", "soddisfatto",
        "eccitato", "eccitata", "vitalità", "energia", "entusiasmo",
    ),
    "fear": (
        "paura", "ansia", "ansioso", "ansiosa", "terrore", "spavento", "spaventato",
        "brivido", "inquieto", "inquieta", "inquietudine", "angoscia", "angosciato",
        "panico", "terrorizzato", "agitato", "agitata",
        "preoccupato", "preoccupata", "incerto", "incerta", "confuso", "confusa",
        "smarrito", "smarrita", "minaccia", "pericolo",
    ),
    "surprise": (
        "sorpresa", "sorpreso", "stupore", "meraviglia", "meravigliato",
        "inaspettato", "shock", "incredibile", "sbalordito", "sbalordita",
        "allibito", "allibita", "sconvolto", "sconvolta",
    ),
    "disgust": (
        "disgusto", "disgustato", "disgustata", "schifo", "ribrezzo", "vomito",
        "ripugnante", "orribile", "ributtante", "nausea",
    ),
    "trust": (
        "fiducia", "sicuro", "sicura", "sicurezza", "affidabile", "protetto",
        "protetta", "protezione", "pace", "pacifico", "amico", "amichevole",
        "sincero", "verità", "rilassato", "rilassata",
    ),
    "anticipation": (
        "attesa", "aspettativa", "curiosità", "curioso", "curiosa", "impazienza",
        "impaziente", "imminente", "trepidazione", "attendo", "sospeso",
    ),
    "love": (
        "amore", "amato", "amata", "amo", "amare", "cuore", "dolcezza", "tenerezza",
        "tenero", "tenera", "affetto", "abbraccio", "bacio", "passione", "innamorato",
    ),
    "awe": (
        "solenne", "sublime", "maestoso", "divino", "incanto", "sacro", "potente",
        "immensità", "infinito", "grandioso", "rispetto", "riverenza",
    ),
}

NEGATION_RE = re.compile(r"\b(non|mai|senza|niente|né|ne)\b", re.IGNORECASE | re.UNICODE)

logger = logging.getLogger("aisthesis")
logging.basicConfig(level=logging.INFO)

_classifier: Optional[Any] = None
MODEL_ID = "MilaNLProc/feel-it-italian-emotion"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def load_classifier() -> Any:
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


app = FastAPI(title="Aisthesis API", version="4.0.0", lifespan=lifespan)

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
# Schemi
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class EmotionScore(BaseModel):
    emotion: EmotionName
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class AffectiveSpace(BaseModel):
    valence: float = Field(..., ge=-1.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    description: str


class SemanticAnalysis(BaseModel):
    """`emotion_mix` guida colore e suono; `dominant_emotion` è solo la prima del mix."""
    dominant_emotion: EmotionName
    emotion: EmotionName  # alias compat = dominant_emotion
    emotion_mix: list[EmotionScore]
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_top1_emotion: EmotionName
    raw_emotion: EmotionName  # alias compat
    threshold_applied: bool
    affective_space: AffectiveSpace
    blend_weights: list[EmotionScore]


class InfluentialTerm(BaseModel):
    term: str
    importance: float = Field(..., ge=0.0, le=1.0)


class ContrastiveInsight(BaseModel):
    favored_emotion: EmotionName
    challenger_emotion: EmotionName
    margin: float
    summary: str
    terms_for_favored: list[InfluentialTerm]
    terms_for_challenger: list[InfluentialTerm]


class LexiconAlignment(BaseModel):
    """Confronto tra segnale lessicale e predizione del modello."""
    lexicon_active: bool
    lexicon_top_emotion: Optional[EmotionName] = None
    model_top_emotion: EmotionName
    contradiction: bool
    matched_terms: list[str]
    note: str


class ExplainabilityState(BaseModel):
    method: str
    decision_summary: str
    top_classes: list[EmotionScore]
    influential_terms: list[InfluentialTerm]
    lexicon_alignment: LexiconAlignment
    contrastive: Optional[ContrastiveInsight] = None
    rule_trace: list[str]


class HSVState(BaseModel):
    h: float = Field(..., ge=0.0, le=360.0)
    s: float = Field(..., ge=0.0, le=100.0)
    v: float = Field(..., ge=0.0, le=100.0)


class ColorLayer(BaseModel):
    emotion: EmotionName
    weight: float = Field(..., ge=0.0, le=1.0)
    color_name: str
    hsv: HSVState
    quote_excerpt: str


class VisualState(BaseModel):
    color_name: str
    hsv: HSVState
    goethe_quote: str
    artistic_description: str
    layers: list[ColorLayer]


class VoiceSpec(BaseModel):
    pitch_hz: float = Field(..., gt=0.0)
    waveform: WaveformName
    gain: float = Field(..., ge=0.0, le=1.0)
    detune_cents: float = 0.0
    delay_ms: float = 0.0


class SonicProfile(BaseModel):
    voices: list[VoiceSpec]
    root_pitch_hz: float = Field(..., gt=0.0)
    scale_name: ScaleName
    tempo_bpm: float = Field(..., gt=0.0)
    attack_time_ms: float = Field(..., gt=0.0)
    release_time_ms: float = Field(..., gt=0.0)
    filter_cutoff_hz: float = Field(..., gt=0.0)
    reverb_mix: float = Field(..., ge=0.0, le=1.0)
    arpeggio_pattern: list[int]
    duration_s: float = Field(..., gt=0.0)


class AudioTargetState(BaseModel):
    """Compatibilità + voce principale."""
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
    sonic_profile: SonicProfile
    audio_target_state: AudioTargetState


# ---------------------------------------------------------------------------
# Utilità emozione
# ---------------------------------------------------------------------------


def _normalize_label(raw: str) -> str:
    return raw.strip().lower()


def map_model_label_to_emotion(label: str) -> EmotionName:
    key = _normalize_label(label)
    if key.startswith("label_"):
        key = key.replace("label_", "", 1)
    mapped = LABEL_ALIASES.get(key)
    if mapped:
        return mapped
    for alias, emotion in LABEL_ALIASES.items():
        if alias in key:
            return emotion
    logger.warning("Etichetta non mappata: %r → neutral", label)
    return "neutral"


def _run_classifier(text: str, clf: Any, top_k: int = 5) -> list[dict[str, Any]]:
    out = clf(text, top_k=top_k, truncation=True, max_length=512)
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        out = out[0]
    if not out:
        return []
    return list(out)


def _build_top_classes(out: list[dict[str, Any]]) -> list[EmotionScore]:
    classes: list[EmotionScore] = []
    
    for item in sorted(out, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        label = str(item.get("label", ""))
        score = float(item.get("score", 0.0))
        if score > 0.015:
            classes.append(
                EmotionScore(
                    emotion=map_model_label_to_emotion(label),
                    label=label,
                    score=score,
                )
            )
            
    total = sum(c.score for c in classes) or 1.0
    for c in classes:
        c.score = round(c.score / total, 4)
        
    return classes


def aggregate_emotion_scores(classes: list[EmotionScore]) -> dict[EmotionName, float]:
    agg: dict[str, float] = {}
    for item in classes:
        agg[item.emotion] = agg.get(item.emotion, 0.0) + item.score
    total = sum(agg.values()) or 1.0
    return {k: v / total for k, v in agg.items()}  # type: ignore[return-value]


def _is_negated(text_lower: str, match_start: int) -> bool:
    window = text_lower[max(0, match_start - 30) : match_start]
    return bool(NEGATION_RE.search(window))


def score_lexicon(text: str) -> tuple[dict[EmotionName, float], list[str]]:
    """Punteggi da parole affettive esplicite nel testo."""
    lowered = text.lower()
    scores: dict[str, float] = {e: 0.0 for e in EMOTION_LEXICON}
    matched: list[str] = []

    for emotion, keywords in EMOTION_LEXICON.items():
        for kw in keywords:
            for match in re.finditer(re.escape(kw), lowered):
                weight = 0.25 if _is_negated(lowered, match.start()) else 1.0
                scores[emotion] += weight
                if kw not in matched:
                    matched.append(kw)

    total = sum(scores.values())
    if total <= 0:
        return {}, matched
    return {k: v / total for k, v in scores.items() if v > 0}, matched  # type: ignore[return-value]


def hybrid_aggregate(
    model_agg: dict[EmotionName, float],
    lex_agg: dict[EmotionName, float],
    text: str = "",
) -> dict[EmotionName, float]:
    if not lex_agg:
        return model_agg

    lex_weight = LEXICON_BLEND_WEIGHT
    word_count = len(_word_spans(text))
    lex_peak = max(lex_agg.values())
    if word_count <= 6 and lex_peak >= 0.4:
        lex_weight = 0.72
    model_weight = 1.0 - lex_weight

    emotions = set(model_agg) | set(lex_agg)
    hybrid: dict[str, float] = {}
    for emotion in emotions:
        hybrid[emotion] = (
            model_weight * model_agg.get(emotion, 0.0)
            + lex_weight * lex_agg.get(emotion, 0.0)
        )
    total = sum(hybrid.values()) or 1.0
    return {k: v / total for k, v in hybrid.items()}  # type: ignore[return-value]


def build_lexicon_alignment(
    model_top: EmotionName,
    lex_agg: dict[EmotionName, float],
    matched: list[str],
    hybrid_top: EmotionName,
) -> LexiconAlignment:
    if not lex_agg:
        return LexiconAlignment(
            lexicon_active=False,
            model_top_emotion=model_top,
            contradiction=False,
            matched_terms=[],
            note="Nessuna parola affettiva esplicita nel testo: il mix segue il modello FEEL-IT.",
        )

    lex_top = max(lex_agg.items(), key=lambda x: x[1])[0]
    matched_str = ", ".join(matched[:4])

    if lex_top == model_top:
        return LexiconAlignment(
            lexicon_active=True,
            contradiction=False,
            note=f"Le parole utilizzate ({matched_str}) confermano e rafforzano la sensazione di {EMOTION_IT[model_top]}.",
            lexicon_top_emotion=lex_top,
            model_top_emotion=model_top,
            matched_terms=matched[:8],
        )
        
    if lex_top == hybrid_top:
        return LexiconAlignment(
            lexicon_active=True,
            contradiction=True,
            note=f"Il modello neurale aveva percepito sfumature di {EMOTION_IT[model_top]}, ma parole esplicite come '{matched_str}' hanno orientato decisamente il risultato verso {EMOTION_IT[lex_top]}.",
            lexicon_top_emotion=lex_top,
            model_top_emotion=model_top,
            matched_terms=matched[:8],
        )
        
    return LexiconAlignment(
        lexicon_active=True,
        contradiction=True,
        note=f"Sottile contrasto: parole come '{matched_str}' richiamano {EMOTION_IT[lex_top]}, ma il tono generale del testo propende maggiormente verso {EMOTION_IT[model_top]}.",
        lexicon_top_emotion=lex_top,
        model_top_emotion=model_top,
        matched_terms=matched[:8],
    )


def get_blend_weights(agg: dict[EmotionName, float]) -> list[tuple[EmotionName, float]]:
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return [("neutral", 1.0)]
    
    top = ranked[:TOP_K_BLEND]
    
    # Se c'è una forte dominanza (es. l'utente ha scritto solo "rabbia"), evitiamo
    # lo smoothing che alzerebbe artificialmente il rumore (es. portando paura al 30%).
    # Altrimenti usiamo un power < 1.0 per bilanciare veri mix (es. "gioia e paura").
    power = 1.0 if top[0][1] > 0.75 else 0.65
    
    smoothed = [(e, math.pow(max(w, 1e-9), power)) for e, w in top]
    total = sum(w for _, w in smoothed) or 1.0
    
    # Filtriamo sotto il MIN_BLEND_WEIGHT
    filtered = [(e, w / total) for e, w in smoothed if (w / total) >= MIN_BLEND_WEIGHT]
    if not filtered:
        return [(top[0][0], 1.0)]
        
    # Rinormalizziamo in modo che la somma finale sia esattamente 1.0
    final_total = sum(w for _, w in filtered)
    return [(e, w / final_total) for e, w in filtered]


def compute_affective_space(agg: dict[EmotionName, float]) -> AffectiveSpace:
    valence = sum(agg.get(e, 0.0) * EMOTION_VA[e][0] for e in EMOTION_VA)
    arousal = sum(agg.get(e, 0.0) * EMOTION_VA[e][1] for e in EMOTION_VA)
    valence = max(-1.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal))

    if valence > 0.35 and arousal > 0.55:
        desc = "Quadrante attivo-positivo: energia luminosa, slancio vitale."
    elif valence > 0.35 and arousal <= 0.55:
        desc = "Quadrante calmo-positivo: serenità, riposo, dolcezza."
    elif valence <= -0.35 and arousal > 0.55:
        desc = "Quadrante attivo-negativo: tensione, urgenza, inquietudine."
    elif valence <= -0.35:
        desc = "Quadrante calmo-negativo: malinconia, distanza, introspezione."
    else:
        desc = "Zona centrale: equilibrio cromatico e sonoro, né estremi."

    return AffectiveSpace(
        valence=round(valence, 3),
        arousal=round(arousal, 3),
        description=desc,
    )


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
        if len(term) >= 2:
            spans.append((match.start(), match.end(), term))
    return spans


def _is_display_term(term: str) -> bool:
    normalized = term.strip("'-").lower()
    return len(normalized) >= 3 and normalized not in DISPLAY_STOPWORDS and normalized not in GENERIC_WORDS

def has_informational_content(text: str) -> bool:
    words = _word_spans(text)
    for _, _, term in words:
        normalized = term.strip("'-").lower()
        if normalized not in DISPLAY_STOPWORDS and normalized not in GENERIC_WORDS:
            return True
    return False


# ---------------------------------------------------------------------------
# Explainability — Integrated Gradients
# ---------------------------------------------------------------------------


def _token_attributions_ig(
    text: str, clf: Any, target_index: int, n_steps: int = IG_STEPS
) -> list[float]:
    tokenizer = clf.tokenizer
    model = clf.model
    if tokenizer is None or model is None:
        return []

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    encoded.pop("offset_mapping")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    emb_layer = model.get_input_embeddings()
    input_embeds = emb_layer(input_ids).detach()
    baseline = torch.zeros_like(input_embeds)

    accumulated = torch.zeros_like(input_embeds)
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interp = (baseline + alpha * (input_embeds - baseline)).clone()
        interp.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(inputs_embeds=interp, attention_mask=attention_mask).logits
        logits[0, target_index].backward()
        if interp.grad is not None:
            accumulated += interp.grad.detach()

    avg_grad = accumulated / n_steps
    attribution = (input_embeds - baseline) * avg_grad
    return attribution[0].sum(dim=-1).abs().detach().cpu().tolist()


def _aggregate_token_scores(
    text: str, token_scores: list[float], offsets: list[tuple[int, int]], special_mask: list[int]
) -> dict[tuple[int, int, str], float]:
    word_scores: dict[tuple[int, int, str], float] = {s: 0.0 for s in _word_spans(text)}
    if not word_scores:
        return {}

    for score, (start, end), is_special in zip(token_scores, offsets, special_mask):
        if is_special or start == end or score <= 0.0:
            continue
        for span in word_scores:
            ss, se, _ = span
            if start >= ss and end <= se:
                word_scores[span] += float(score)
                break
    return word_scores


def _terms_from_word_scores(word_scores: dict[tuple[int, int, str], float]) -> list[InfluentialTerm]:
    max_score = max(word_scores.values(), default=0.0)
    if max_score <= 0.0:
        return []
    terms = [
        InfluentialTerm(term=term, importance=round(score / max_score, 4))
        for (_, _, term), score in word_scores.items()
        if (score / max_score) >= EXPLANATION_MIN_DELTA and _is_display_term(term)
    ]
    terms.sort(key=lambda t: t.importance, reverse=True)
    return terms[:EXPLANATION_TOP_TERMS]


def explain_emotion_terms(text: str, clf: Any, emotion: EmotionName) -> list[InfluentialTerm]:
    if emotion == "neutral":
        return []
    model = clf.model
    if model is None:
        return []
    indices = _model_emotion_indices(model)
    target = indices.get(emotion)
    if target is None:
        return []

    tokenizer = clf.tokenizer
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    special_mask = tokenizer.get_special_tokens_mask(
        encoded["input_ids"][0].tolist(), already_has_special_tokens=True
    )
    token_scores = _token_attributions_ig(text, clf, target)
    if not token_scores:
        return []
    word_scores = _aggregate_token_scores(text, token_scores, offsets, special_mask)
    return _terms_from_word_scores(word_scores)


def build_contrastive(
    text: str,
    clf: Any,
    favored: EmotionName,
    challenger: EmotionName,
    favored_score: float,
    challenger_score: float,
) -> Optional[ContrastiveInsight]:
    if favored == challenger:
        return None
    terms_f = explain_emotion_terms(text, clf, favored)
    terms_c = explain_emotion_terms(text, clf, challenger)
    margin = favored_score - challenger_score
    
    summary = f"C'è una sottile tensione emotiva: il modello ha percepito principalmente {EMOTION_IT[favored]}, ma ha notato anche forti tracce di {EMOTION_IT[challenger]}. "
    
    if terms_f and terms_c:
        wf = ", ".join(f"'{t.term}'" for t in terms_f[:2])
        wc = ", ".join(f"'{t.term}'" for t in terms_c[:2])
        summary += f"Mentre parole come {wf} ancorano il testo a {EMOTION_IT[favored]}, espressioni come {wc} lasciano trasparire {EMOTION_IT[challenger]}."
        
    return ContrastiveInsight(
        favored_emotion=favored,
        challenger_emotion=challenger,
        margin=round(margin, 4),
        summary=summary,
        terms_for_favored=terms_f,
        terms_for_challenger=terms_c,
    )


# ---------------------------------------------------------------------------
# Blend cromatico Goethe
# ---------------------------------------------------------------------------


def blend_hsv(weights: list[tuple[EmotionName, float]]) -> tuple[float, float, float]:
    if not weights:
        h, s, v = GOETHE_DATA["neutral"]["hsv"]
        return h, s, v
    sin_sum = sum(w * math.sin(math.radians(GOETHE_DATA[e]["hsv"][0])) for e, w in weights)
    cos_sum = sum(w * math.cos(math.radians(GOETHE_DATA[e]["hsv"][0])) for e, w in weights)
    h = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
    s = sum(w * GOETHE_DATA[e]["hsv"][1] for e, w in weights)
    v = sum(w * GOETHE_DATA[e]["hsv"][2] for e, w in weights)
    return h, s, v


def build_color_layers(weights: list[tuple[EmotionName, float]]) -> list[ColorLayer]:
    layers: list[ColorLayer] = []
    for emotion, weight in weights:
        data = GOETHE_DATA[emotion]
        h, s, v = data["hsv"]
        layers.append(
            ColorLayer(
                emotion=emotion,
                weight=round(weight, 4),
                color_name=data["name"],
                hsv=HSVState(h=h, s=s, v=v),
                quote_excerpt=data["quote"][:90] + "…" if len(data["quote"]) > 90 else data["quote"],
            )
        )
    return layers


def build_goethe_explanation(dominant: EmotionName, hsv: tuple[float, float, float], affective: AffectiveSpace, blend: list[tuple[EmotionName, float]]) -> tuple[str, str, str]:
    if not blend:
        return GOETHE_DATA["neutral"]["name"], GOETHE_DATA["neutral"]["quote"], "Un equilibrio sospeso, senza spiccate polarità."

    color_name = "Mélange cromatico"
    if len(blend) == 1:
        color_name = GOETHE_DATA[dominant]["name"]

    quote = GOETHE_DATA[dominant]["quote"]

    quadrant = "neutro"
    if affective.valence > 0 and affective.arousal > 0:
        quadrant = "di slancio ed energia vitale"
    elif affective.valence > 0 and affective.arousal < 0:
        quadrant = "di profonda serenità e contemplazione"
    elif affective.valence < 0 and affective.arousal > 0:
        quadrant = "carico di tensione e urgente inquietudine"
    elif affective.valence < 0 and affective.arousal < 0:
        quadrant = "intriso di malinconia, distanza e introspezione"

    if len(blend) == 1:
        desc = f"La tinta pura del {GOETHE_DATA[dominant]['name']} domina incontrastata la tavolozza, creando un'atmosfera {quadrant}."
    else:
        dominant_color = GOETHE_DATA[dominant]["name"]
        other_colors = " e ".join(GOETHE_DATA[e]["name"] for e, w in blend if e != dominant)
        desc = f"La base dominante di {dominant_color} viene sfumata con accenti di {other_colors}. Il risultato è un paesaggio emotivo {quadrant}."

    return color_name, quote, desc


# ---------------------------------------------------------------------------
# Profilo sonoro musicale
# ---------------------------------------------------------------------------


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def choose_scale(valence: float, blend: list[tuple[EmotionName, float]]) -> ScaleName:
    emotions = {e for e, _ in blend}
    
    # Valenza positiva (Gioia/Tranquillità)
    if valence >= 0.15:
        if "anger" in emotions or "fear" in emotions:
            return "mixolydian" # Gioia mista a tensione
        return "lydian" if valence > 0.6 else "pentatonic_major"
        
    # Valenza negativa (Tristezza/Rabbia/Paura)
    elif valence < -0.15:
        if "anger" in emotions and "fear" in emotions:
            return "phrygian" # Tensione estrema
        if "anger" in emotions:
            return "phrygian" # Rabbia (molto tesa)
        if "sadness" in emotions:
            return "pentatonic_minor" # Tristezza (malinconica)
        return "dorian" # Paura o tensione moderata
        
    # Valenza neutra (Equilibrio)
    else:
        return "dorian" if "sadness" in emotions else "pentatonic_minor"


def waveform_for_timbre(saturation: float, arousal: float) -> WaveformName:
    if arousal > 0.75 and saturation > 70:
        return "triangle"
    return "sine"


def build_arpeggio_pattern(arousal: float, valence: float, n_layers: int) -> list[int]:
    if arousal > 0.7:
        base = [0, 2, 1, 3, 2, 4, 3]
    elif arousal > 0.4:
        base = [0, 2, 4, 2, 0]
    else:
        base = [0, 2, 0, 4] if valence >= 0 else [0, 3, 0, 2]
    return base[: max(4, min(8, 4 + n_layers))]


def build_sonic_profile(
    affective: AffectiveSpace,
    hsv: tuple[float, float, float],
    blend: list[tuple[EmotionName, float]],
    confidence: float,
) -> SonicProfile:
    h, s, v = hsv
    valence, arousal = affective.valence, affective.arousal
    scale_name = choose_scale(valence, blend)
    intervals = SCALE_INTERVALS[scale_name]

    root_midi = 60.0 + valence * 7.0 + arousal * 5.0
    root_midi = max(52.0, min(79.0, root_midi))
    root_hz = midi_to_hz(root_midi)

    pattern = build_arpeggio_pattern(arousal, valence, len(blend))
    wform = waveform_for_timbre(s, arousal)

    voices: list[VoiceSpec] = []
    for i, degree_idx in enumerate(pattern[:4]):
        semitone = intervals[degree_idx % len(intervals)]
        octave_bump = 12 if i >= 2 and arousal > 0.5 else 0
        midi_note = root_midi + semitone + octave_bump
        gain = (0.32 / (1 + i * 0.35)) * (0.55 + confidence * 0.45) * (v / 100.0)
        detune = (-7.0 + i * 5.0) if i > 0 else 0.0
        delay = i * (180.0 - arousal * 80.0)
        voices.append(
            VoiceSpec(
                pitch_hz=round(midi_to_hz(midi_note), 3),
                waveform=wform,
                gain=round(min(0.45, gain), 4),
                detune_cents=detune,
                delay_ms=round(delay, 1),
            )
        )

    attack = 50.0 + (1.0 - arousal) * 750.0
    release = 1200.0 + (1.0 - arousal) * 2200.0 + (1.0 - valence) * 400.0
    cutoff = 600.0 + arousal * 3200.0 + s * 25.0
    reverb = min(0.85, 0.15 + (1.0 - arousal) * 0.45 + (1.0 - v / 100.0) * 0.2)
    tempo = 68.0 + arousal * 92.0 + confidence * 20.0
    duration = 2.8 + arousal * 1.2 + (1.0 - arousal) * 1.5

    return SonicProfile(
        voices=voices,
        root_pitch_hz=round(root_hz, 3),
        scale_name=scale_name,
        tempo_bpm=round(tempo, 1),
        attack_time_ms=round(attack, 1),
        release_time_ms=round(release, 1),
        filter_cutoff_hz=round(cutoff, 1),
        reverb_mix=round(reverb, 3),
        arpeggio_pattern=pattern,
        duration_s=round(duration, 2),
    )


def sonic_to_legacy_audio(sonic: SonicProfile, hsv: tuple[float, float, float]) -> AudioTargetState:
    _, s, v = hsv
    primary = sonic.voices[0] if sonic.voices else VoiceSpec(
        pitch_hz=sonic.root_pitch_hz, waveform="sine", gain=0.3
    )
    return AudioTargetState(
        waveform=primary.waveform,
        pitch_hz=primary.pitch_hz,
        filter_cutoff_hz=sonic.filter_cutoff_hz,
        amplitude=round(min(0.9, primary.gain * 1.4), 4),
        attack_time_ms=sonic.attack_time_ms,
        reverb_mix=sonic.reverb_mix,
    )


# ---------------------------------------------------------------------------
# Inference e response
# ---------------------------------------------------------------------------


def run_emotion_inference(text: str, clf: Any) -> dict[str, Any]:
    out = _run_classifier(text, clf)
    top_classes = _build_top_classes(out)

    if not out:
        empty_agg: dict[EmotionName, float] = {"neutral": 1.0}
        return {
            "emotion": "neutral",
            "raw_emotion": "neutral",
            "raw_top1_emotion": "neutral",
            "confidence": 0.0,
            "threshold_applied": False,
            "top_classes": top_classes,
            "agg": empty_agg,
            "blend": [("neutral", 1.0)],
            "affective": compute_affective_space(empty_agg),
            "influential_terms": [],
            "contrastive": None,
            "lexicon_alignment": build_lexicon_alignment("neutral", {}, []),
        }

    best = max(out, key=lambda x: float(x.get("score", 0.0)))
    raw_label = str(best.get("label", ""))
    score = float(best.get("score", 0.0))
    raw_emotion = map_model_label_to_emotion(raw_label)
    threshold_applied = score < CONFIDENCE_THRESHOLD

    model_agg = aggregate_emotion_scores(top_classes)
    lex_agg, matched_terms = score_lexicon(text)
    agg = hybrid_aggregate(model_agg, lex_agg, text)
    blend = get_blend_weights(agg)
    emotion = blend[0][0]
    affective = compute_affective_space(agg)
    lexicon_alignment = build_lexicon_alignment(raw_emotion, lex_agg, matched_terms, emotion)

    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    favored = ranked[0][0]
    challenger = ranked[1][0] if len(ranked) > 1 else favored

    ig_target = favored
    if lexicon_alignment.contradiction and lexicon_alignment.lexicon_top_emotion:
        ig_target = lexicon_alignment.lexicon_top_emotion

    terms = explain_emotion_terms(text, clf, ig_target)
    contrastive = build_contrastive(
        text, clf, favored, raw_emotion if raw_emotion != favored else challenger,
        ranked[0][1],
        model_agg.get(raw_emotion, 0.0) if raw_emotion != favored else ranked[1][1] if len(ranked) > 1 else 0.0,
    )

    return {
        "emotion": emotion,
        "raw_emotion": raw_emotion,
        "raw_top1_emotion": raw_emotion,
        "confidence": score,
        "threshold_applied": threshold_applied,
        "top_classes": top_classes,
        "model_agg": model_agg,
        "lex_agg": lex_agg,
        "agg": agg,
        "blend": blend,
        "affective": affective,
        "influential_terms": terms,
        "ig_explain_emotion": ig_target,
        "contrastive": contrastive,
        "lexicon_alignment": lexicon_alignment,
    }


def build_decision_summary(inference: dict[str, Any]) -> str:
    blend = inference["blend"]
    affective = inference["affective"]
    terms = inference["influential_terms"]
    blend_str = " e ".join(f"{EMOTION_IT[e]} ({w:.0%})" for e, w in blend)

    head = f"Il testo rivela un mix emotivo composto da {blend_str}. "

    lex = inference.get("lexicon_alignment")
    if lex and lex.contradiction:
        head += f"Curiosamente, {lex.note.lower()} "

    summary = f"{head}Questa miscela si colloca in uno spazio affettivo caratterizzato da "
    
    if affective.valence > 0.2:
        summary += "un'energia positiva "
    elif affective.valence < -0.2:
        summary += "una forte carica negativa "
    else:
        summary += "un umore neutrale "
        
    if affective.arousal > 0.2:
        summary += "e grande intensità. "
    elif affective.arousal < -0.2:
        summary += "e una certa calma. "
    else:
        summary += "e intensità moderata. "

    if terms:
        ig_em = inference.get("ig_explain_emotion", inference["blend"][0][0])
        summary += f"Le parole che più di tutte hanno contribuito a far emergere {EMOTION_IT[ig_em]} sono state: {', '.join(t.term for t in terms[:4])}."
    return summary


def build_rule_trace(
    blend: list[tuple[EmotionName, float]],
    hsv: tuple[float, float, float],
    affective: AffectiveSpace,
    sonic: SonicProfile,
    lexicon_alignment: Optional[LexiconAlignment] = None,
) -> list[str]:
    blend_line = " e ".join(f"{GOETHE_DATA[e]['name']}" for e, w in blend)
    trace = [
        f"I colori scelti per le emozioni ({blend_line}) si sono fusi in un unico colore composito.",
        f"L'umore e l'intensità del testo hanno ispirato una melodia in scala {sonic.scale_name.replace('_', ' ')} a un ritmo di {sonic.tempo_bpm:.0f} BPM.",
        f"La frequenza di base è stata posizionata a {sonic.root_pitch_hz:.1f} Hz per risuonare con lo stato d'animo percepito.",
        f"Il suono è stato modellato con un timbro morbido ({sonic.voices[0].waveform if sonic.voices else 'sine'}), con un attacco lento che cresce per {sonic.attack_time_ms:.0f}ms e sfuma per {sonic.release_time_ms:.0f}ms.",
        f"L'effetto di riverbero aggiunge una sensazione di profondità e 'distanza' emotiva all'arpeggio.",
        "Ricorda: il legame tra colore, suono ed emozione è qui un'interpretazione estetico-poetica ispirata a Goethe, non una scienza esatta.",
    ]
    if lexicon_alignment and lexicon_alignment.lexicon_active:
        trace.insert(
            1,
            f"L'analisi lessicale: {lexicon_alignment.note.lower()}",
        )
    return trace


def build_process_response(text: str, inference: dict[str, Any]) -> ProcessResponse:
    blend = inference["blend"]
    affective = inference["affective"]
    hsv = blend_hsv(blend)
    layers = build_color_layers(blend)
    sonic = build_sonic_profile(affective, hsv, blend, inference["confidence"])
    legacy = sonic_to_legacy_audio(sonic, hsv)

    blend_scores = [
        EmotionScore(emotion=e, label=e, score=round(w, 4)) for e, w in blend
    ]

    dominant = inference["emotion"]
    color_name, quote, desc = build_goethe_explanation(dominant, hsv, affective, blend)

    return ProcessResponse(
        input_text=text,
        semantic_analysis=SemanticAnalysis(
            dominant_emotion=dominant,
            emotion=dominant,
            emotion_mix=blend_scores,
            confidence=round(float(inference["confidence"]), 2),
            raw_top1_emotion=inference["raw_top1_emotion"],
            raw_emotion=inference["raw_emotion"],
            threshold_applied=inference["threshold_applied"],
            affective_space=affective,
            blend_weights=blend_scores,
        ),
        explainability=ExplainabilityState(
            method="IG + lessico italiano ibrido (50/50) + contrastivo",
            decision_summary=build_decision_summary(inference),
            top_classes=inference["top_classes"],
            influential_terms=inference["influential_terms"],
            lexicon_alignment=inference["lexicon_alignment"],
            contrastive=inference.get("contrastive"),
            rule_trace=build_rule_trace(blend, hsv, affective, sonic, inference.get("lexicon_alignment")),
        ),
        visual_state=VisualState(
            color_name=color_name,
            hsv=HSVState(h=hsv[0], s=hsv[1], v=hsv[2]),
            goethe_quote=quote,
            artistic_description=desc,
            layers=layers,
        ),
        sonic_profile=sonic,
        audio_target_state=legacy,
    )


@app.post("/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest) -> ProcessResponse:
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Modello non ancora caricato.")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail='Il campo "text" non può essere vuoto.')
    
    if not has_informational_content(text):
        raise HTTPException(
            status_code=400, 
            detail="Testo troppo generico. Aggiungi qualche aggettivo o parola descrittiva."
        )

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
        "process v4: blend=%s valence=%.2f arousal=%.2f scale=%s voices=%d",
        [e for e, _ in inference["blend"]],
        payload.semantic_analysis.affective_space.valence,
        payload.semantic_analysis.affective_space.arousal,
        payload.sonic_profile.scale_name,
        len(payload.sonic_profile.voices),
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
