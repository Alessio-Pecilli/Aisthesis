"""
Aisthesis — Backend FastAPI v5
Pipeline Goethe-centrica: FEEL-IT → colori Goethe (7 canonici + 4 estensioni documentate)
→ blend cromatico → sonificazione da carattere cromatico → explainability (IG).
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

from goethe import (
    EMOTION_TO_GOETHE,
    GOETHE_COLOR_IT,
    GOETHE_COLORS,
    GoetheColorId,
    blend_hsv,
    build_goethe_explanation,
    build_lexicon_alignment_goethe,
    build_sonic_from_goethe_blend,
    compute_pole_balance,
    color_scores_to_emotions,
    emotions_to_goethe_agg,
    get_goethe_blend_weights,
    goethe_color_to_emotion,
    hybrid_goethe_aggregate,
    score_goethe_lexicon,
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

# Russell circumplex — coordinate normalizzate [-1, 1] (ausiliario)
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


app = FastAPI(title="Aisthesis API", version="5.0.0", lifespan=lifespan)

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


class GoetheColorScore(BaseModel):
    color_id: GoetheColorId
    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    source: str
    pole: str


class GoetheAnalysis(BaseModel):
    """Analisi primaria: mix cromatico goethiano."""
    dominant_color: GoetheColorId
    color_name: str
    color_mix: list[GoetheColorScore]
    pole_balance: float = Field(..., ge=-1.0, le=1.0)
    pole_description: str
    canonical_count: int
    extension_count: int


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


class LexiconAlignment(BaseModel):
    """Confronto tra segnale lessicale cromatico e predizione del modello."""
    lexicon_active: bool
    lexicon_top_emotion: Optional[EmotionName] = None
    lexicon_top_color: Optional[GoetheColorId] = None
    model_top_emotion: EmotionName
    model_top_color: GoetheColorId
    contradiction: bool
    matched_terms: list[str]
    note: str


class ExplainabilityState(BaseModel):
    method: str
    decision_summary: str
    top_classes: list[EmotionScore]
    influential_terms: list[InfluentialTerm]
    lexicon_alignment: LexiconAlignment
    rule_trace: list[str]


class HSVState(BaseModel):
    h: float = Field(..., ge=0.0, le=360.0)
    s: float = Field(..., ge=0.0, le=100.0)
    v: float = Field(..., ge=0.0, le=100.0)


class ColorLayer(BaseModel):
    goethe_color: GoetheColorId
    emotion: EmotionName
    weight: float = Field(..., ge=0.0, le=1.0)
    color_name: str
    hsv: HSVState
    quote_excerpt: str
    source: str
    pole: str
    attribution: Optional[str] = None


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
    goethe_analysis: GoetheAnalysis
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
    goethe_lex: dict[GoetheColorId, float],
    goethe_hybrid_top: GoetheColorId,
) -> LexiconAlignment:
    goethe_raw = build_lexicon_alignment_goethe(
        EMOTION_TO_GOETHE[model_top],
        goethe_lex,
        matched,
        goethe_hybrid_top,
    )
    lex_top_emo: Optional[EmotionName] = None
    if goethe_raw["lexicon_top_color"]:
        lex_top_emo = goethe_color_to_emotion(goethe_raw["lexicon_top_color"])

    if not lex_agg and not goethe_lex:
        return LexiconAlignment(
            lexicon_active=False,
            model_top_emotion=model_top,
            model_top_color=EMOTION_TO_GOETHE[model_top],
            contradiction=False,
            matched_terms=[],
            note=goethe_raw["note"],
        )

    if lex_agg and not goethe_lex:
        lex_top = max(lex_agg.items(), key=lambda x: x[1])[0]
        matched_str = ", ".join(matched[:4])
        if lex_top == model_top:
            note = f"Le parole utilizzate ({matched_str}) confermano {EMOTION_IT[model_top]} → {GOETHE_COLOR_IT[EMOTION_TO_GOETHE[model_top]]}."
            contradiction = False
        elif lex_top == hybrid_top:
            note = f"Il modello percepiva {EMOTION_IT[model_top]}, ma parole come '{matched_str}' orientano verso {EMOTION_IT[lex_top]} ({GOETHE_COLOR_IT[EMOTION_TO_GOETHE[lex_top]]})."
            contradiction = True
        else:
            note = f"Contrasto tra {EMOTION_IT[lex_top]} e il tono di {EMOTION_IT[model_top]}."
            contradiction = True
        return LexiconAlignment(
            lexicon_active=True,
            lexicon_top_emotion=lex_top,
            lexicon_top_color=EMOTION_TO_GOETHE.get(lex_top),
            model_top_emotion=model_top,
            model_top_color=EMOTION_TO_GOETHE[model_top],
            contradiction=contradiction,
            matched_terms=matched[:8],
            note=note,
        )

    return LexiconAlignment(
        lexicon_active=goethe_raw["lexicon_active"],
        lexicon_top_emotion=lex_top_emo,
        lexicon_top_color=goethe_raw.get("lexicon_top_color"),
        model_top_emotion=model_top,
        model_top_color=goethe_raw["model_top_color"],
        contradiction=goethe_raw["contradiction"],
        matched_terms=goethe_raw["matched_terms"],
        note=goethe_raw["note"],
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


# ---------------------------------------------------------------------------
# Layer cromatici e risposta
# ---------------------------------------------------------------------------


def build_color_layers(weights: list[tuple[GoetheColorId, float]]) -> list[ColorLayer]:
    layers: list[ColorLayer] = []
    for color_id, weight in weights:
        spec = GOETHE_COLORS[color_id]
        h, s, v = spec.hsv
        quote = spec.quote
        layers.append(
            ColorLayer(
                goethe_color=color_id,
                emotion=goethe_color_to_emotion(color_id),
                weight=round(weight, 4),
                color_name=spec.name_it,
                hsv=HSVState(h=h, s=s, v=v),
                quote_excerpt=quote[:90] + "…" if len(quote) > 90 else quote,
                source=spec.source,
                pole=spec.pole,
                attribution=spec.attribution if spec.source == "extension" else None,
            )
        )
    return layers


def _sonic_dict_to_profile(sonic: dict[str, Any]) -> SonicProfile:
    return SonicProfile(
        voices=[VoiceSpec(**v) for v in sonic["voices"]],
        root_pitch_hz=sonic["root_pitch_hz"],
        scale_name=sonic["scale_name"],
        tempo_bpm=sonic["tempo_bpm"],
        attack_time_ms=sonic["attack_time_ms"],
        release_time_ms=sonic["release_time_ms"],
        filter_cutoff_hz=sonic["filter_cutoff_hz"],
        reverb_mix=sonic["reverb_mix"],
        arpeggio_pattern=sonic["arpeggio_pattern"],
        duration_s=sonic["duration_s"],
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
        empty_goethe: dict[GoetheColorId, float] = {"gruen": 1.0}
        goethe_blend: list[tuple[GoetheColorId, float]] = [("gruen", 1.0)]
        return {
            "emotion": "neutral",
            "raw_emotion": "neutral",
            "raw_top1_emotion": "neutral",
            "confidence": 0.0,
            "threshold_applied": False,
            "top_classes": top_classes,
            "agg": {"neutral": 1.0},
            "blend": [("neutral", 1.0)],
            "goethe_agg": empty_goethe,
            "goethe_blend": goethe_blend,
            "dominant_color": "gruen",
            "affective": compute_affective_space({"neutral": 1.0}),
            "influential_terms": [],
            "lexicon_alignment": build_lexicon_alignment(
                "neutral", {}, [], "neutral", {}, "gruen"
            ),
            "pole_balance": 0.0,
            "pole_description": "Equilibrio tra i poli",
        }

    best = max(out, key=lambda x: float(x.get("score", 0.0)))
    raw_label = str(best.get("label", ""))
    score = float(best.get("score", 0.0))
    raw_emotion = map_model_label_to_emotion(raw_label)
    threshold_applied = score < CONFIDENCE_THRESHOLD

    model_agg = aggregate_emotion_scores(top_classes)
    goethe_lex, matched_terms = score_goethe_lexicon(text)
    lex_agg = color_scores_to_emotions(goethe_lex)
    agg = hybrid_aggregate(model_agg, lex_agg, text)
    blend = get_blend_weights(agg)
    emotion = blend[0][0]

    model_goethe = emotions_to_goethe_agg(model_agg)
    goethe_agg = hybrid_goethe_aggregate(model_goethe, goethe_lex, text, MODEL_BLEND_WEIGHT)
    goethe_blend = get_goethe_blend_weights(goethe_agg)
    dominant_color = goethe_blend[0][0]
    pole_balance, pole_description = compute_pole_balance(goethe_blend)

    affective = compute_affective_space(agg)
    lexicon_alignment = build_lexicon_alignment(
        raw_emotion, lex_agg, matched_terms, emotion, goethe_lex, dominant_color
    )

    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    ig_target = ranked[0][0]
    if lexicon_alignment.contradiction and lexicon_alignment.lexicon_top_emotion:
        ig_target = lexicon_alignment.lexicon_top_emotion

    terms = explain_emotion_terms(text, clf, ig_target)

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
        "goethe_agg": goethe_agg,
        "goethe_blend": goethe_blend,
        "dominant_color": dominant_color,
        "pole_balance": pole_balance,
        "pole_description": pole_description,
        "affective": affective,
        "influential_terms": terms,
        "ig_explain_emotion": ig_target,
        "lexicon_alignment": lexicon_alignment,
    }


def build_decision_summary(inference: dict[str, Any]) -> str:
    goethe_blend = inference["goethe_blend"]
    affective = inference["affective"]
    terms = inference["influential_terms"]
    blend_str = " e ".join(
        f"{GOETHE_COLOR_IT[c]} ({w:.0%})" for c, w in goethe_blend
    )

    head = f"Il testo si traduce in un mélange cromatico goethiano: {blend_str}. "
    head += f"{inference['pole_description']}. "

    lex = inference.get("lexicon_alignment")
    if lex and lex.contradiction:
        head += f"{lex.note} "

    summary = f"{head}Nello spazio affettivo (Russell): "
    if affective.valence > 0.2:
        summary += "energia positiva "
    elif affective.valence < -0.2:
        summary += "carica negativa "
    else:
        summary += "tono neutro "
    if affective.arousal > 0.2:
        summary += "ad alta intensità. "
    elif affective.arousal < -0.2:
        summary += "con calma. "
    else:
        summary += "a intensità moderata. "

    if terms:
        ig_em = inference.get("ig_explain_emotion", inference["blend"][0][0])
        dom_color = GOETHE_COLOR_IT[EMOTION_TO_GOETHE.get(ig_em, "gruen")]
        summary += (
            f"Parole salienti per {EMOTION_IT[ig_em]} → {dom_color}: "
            f"{', '.join(t.term for t in terms[:4])}."
        )
    return summary


def build_rule_trace(
    goethe_blend: list[tuple[GoetheColorId, float]],
    sonic: SonicProfile,
    sonic_rationales: list[str],
    lexicon_alignment: Optional[LexiconAlignment] = None,
) -> list[str]:
    blend_line = " e ".join(GOETHE_COLOR_IT[c] for c, _ in goethe_blend)
    dominant = goethe_blend[0][0]
    dom_spec = GOETHE_COLORS[dominant]
    trace = [
        f"Classificazione cromatica: {blend_line} si fondono in un colore composito.",
        f"Il colore base ({dom_spec.name_it}) è {'un colore CANONICO descritto originariamente da Goethe' if dom_spec.source == 'goethe' else 'un colore ESTESO derivato da studi psicologici successivi'} ({dom_spec.attribution}).",
        f"Perché questa associazione? {dom_spec.quote.strip()}",
        f"Il paesaggio sonoro è stato generato a partire da questa natura cromatica: {sonic_rationales[0] if sonic_rationales else dom_spec.sonic.sonic_rationale}",
        f"Scala {sonic.scale_name.replace('_', ' ')}, {sonic.tempo_bpm:.0f} BPM, frequenza radice {sonic.root_pitch_hz:.1f} Hz.",
        f"Sintesi: forma d'onda {sonic.voices[0].waveform if sonic.voices else 'sine'}, attacco {sonic.attack_time_ms:.0f} ms.",
    ]
    if lexicon_alignment and lexicon_alignment.lexicon_active:
        trace.insert(1, f"Lessico cromatico: {lexicon_alignment.note}")
    return trace


def build_process_response(text: str, inference: dict[str, Any]) -> ProcessResponse:
    goethe_blend = inference["goethe_blend"]
    dominant_color = inference["dominant_color"]
    affective = inference["affective"]
    hsv = blend_hsv(goethe_blend)
    layers = build_color_layers(goethe_blend)

    sonic_raw = build_sonic_from_goethe_blend(goethe_blend, inference["confidence"])
    sonic = _sonic_dict_to_profile(sonic_raw)
    legacy = sonic_to_legacy_audio(sonic, hsv)

    goethe_scores = [
        GoetheColorScore(
            color_id=c,
            name=GOETHE_COLORS[c].name_it,
            score=round(w, 4),
            source=GOETHE_COLORS[c].source,
            pole=GOETHE_COLORS[c].pole,
        )
        for c, w in goethe_blend
    ]

    emotion_blend_scores = [
        EmotionScore(
            emotion=goethe_color_to_emotion(c),
            label=c,
            score=round(w, 4),
        )
        for c, w in goethe_blend
    ]

    dominant_emotion = goethe_color_to_emotion(dominant_color)
    color_name, quote, desc = build_goethe_explanation(
        dominant_color, goethe_blend, inference["pole_description"]
    )

    canonical = sum(1 for c, _ in goethe_blend if GOETHE_COLORS[c].source == "goethe")
    extension = len(goethe_blend) - canonical

    return ProcessResponse(
        input_text=text,
        goethe_analysis=GoetheAnalysis(
            dominant_color=dominant_color,
            color_name=GOETHE_COLORS[dominant_color].name_it,
            color_mix=goethe_scores,
            pole_balance=inference["pole_balance"],
            pole_description=inference["pole_description"],
            canonical_count=canonical,
            extension_count=extension,
        ),
        semantic_analysis=SemanticAnalysis(
            dominant_emotion=dominant_emotion,
            emotion=dominant_emotion,
            emotion_mix=emotion_blend_scores,
            confidence=round(float(inference["confidence"]), 2),
            raw_top1_emotion=inference["raw_top1_emotion"],
            raw_emotion=inference["raw_emotion"],
            threshold_applied=inference["threshold_applied"],
            affective_space=affective,
            blend_weights=emotion_blend_scores,
        ),
        explainability=ExplainabilityState(
            method="Goethe + FEEL-IT→colore + lessico cromatico + IG",
            decision_summary=build_decision_summary(inference),
            top_classes=inference["top_classes"],
            influential_terms=inference["influential_terms"],
            lexicon_alignment=inference["lexicon_alignment"],
            rule_trace=build_rule_trace(
                goethe_blend,
                sonic,
                sonic_raw.get("sonic_rationales", []),
                inference.get("lexicon_alignment"),
            ),
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
        "process v5: goethe=%s pole=%.2f scale=%s voices=%d",
        [c for c, _ in inference["goethe_blend"]],
        inference["pole_balance"],
        payload.sonic_profile.scale_name,
        len(payload.sonic_profile.voices),
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
