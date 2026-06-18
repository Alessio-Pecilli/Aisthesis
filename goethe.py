"""
Ontologia cromatica Goethe per Aisthesis.

7 colori canonici da *Zur Farbenlehre* (sezione sinnlich-sittliche Wirkung).
4 estensioni documentate per qualità affettive assenti nella tassonomia goethiana:
  - rosa (amore): Palmer & Schloss 2010; Collins 2011
  - azzurro (fiducia): Marks 1974/1987 (luminosità↔pitch); Ou et al. 2018
  - verde_marcio (disgusto): Palmer et al. 2013 (disgust→verde scuro)
  - cremisi (attesa/sorpresa): Ferguson & Brewster 2017 (sharpness/arousal)

Sonificazione: carattere goethiano del colore + corrispondenze crossmodali
(luminosità→pitch/altezza, saturazione→timbro, Ferguson roughness per repulsione).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

GoetheColorId = Literal[
    "gelb",
    "gelbrot",
    "rotgelb",
    "blau",
    "blaurot",
    "rotblau",
    "gruen",
    "rosa",
    "azzurro",
    "verde_marcio",
    "cremisi",
]

EmotionName = Literal[
    "joy", "sadness", "anger", "fear", "surprise", "disgust",
    "trust", "anticipation", "love", "awe", "neutral",
]

ColorPole = Literal["active", "passive", "balanced"]
ColorSource = Literal["goethe", "extension"]
WaveformName = Literal["sine", "triangle", "sawtooth", "square"]
ScaleName = Literal[
    "pentatonic_major", "pentatonic_minor", "dorian",
    "mixolydian", "phrygian", "lydian",
]

SCALE_INTERVALS: dict[ScaleName, list[int]] = {
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}

TOP_K_BLEND = 4
MIN_BLEND_WEIGHT = 0.05

NEGATION_RE = re.compile(r"\b(non|mai|senza|niente|né|ne)\b", re.IGNORECASE | re.UNICODE)


@dataclass(frozen=True)
class SonicColorProfile:
    """Parametri sonori derivati dal carattere cromatico goethiano."""
    scale_name: ScaleName
    root_midi: float
    waveform: WaveformName
    tempo_bpm: float
    attack_time_ms: float
    release_time_ms: float
    filter_cutoff_hz: float
    reverb_mix: float
    arpeggio_pattern: tuple[int, ...]
    sonic_rationale: str


@dataclass(frozen=True)
class GoetheColorSpec:
    id: GoetheColorId
    name_it: str
    name_de: str
    hsv: tuple[float, float, float]
    quote: str
    description: str
    affect: str
    pole: ColorPole
    source: ColorSource
    attribution: str
    related_emotions: tuple[EmotionName, ...]
    sonic: SonicColorProfile


# ---------------------------------------------------------------------------
# Catalogo colori
# ---------------------------------------------------------------------------

GOETHE_COLORS: dict[GoetheColorId, GoetheColorSpec] = {
    "gelb": GoetheColorSpec(
        id="gelb",
        name_it="Giallo",
        name_de="Gelb",
        hsv=(52.0, 55.0, 95.0),
        quote="Il colore più vicino alla luce possiede un carattere sereno, lieto, dolcemente eccitante.",
        description="L'occhio ne viene allietato, l'animo si rasserena; un calore immediato ci investe.",
        affect="Gioia, serenità, nobiltà",
        pole="active",
        source="goethe",
        attribution="Goethe, Zur Farbenlehre (1810), sez. Giallo",
        related_emotions=("joy",),
        sonic=SonicColorProfile(
            scale_name="pentatonic_major",
            root_midi=67.0,
            waveform="sine",
            tempo_bpm=78.0,
            attack_time_ms=280.0,
            release_time_ms=1800.0,
            filter_cutoff_hz=2800.0,
            reverb_mix=0.22,
            arpeggio_pattern=(0, 2, 4, 2, 0),
            sonic_rationale="Sereno e dolcemente eccitante: registro medio-alto, attacco morbido (Goethe).",
        ),
    ),
    "gelbrot": GoetheColorSpec(
        id="gelbrot",
        name_it="Arancione",
        name_de="Gelbroth",
        hsv=(30.0, 85.0, 95.0),
        quote="Un giallo che cresce in densità ed energia verso il lato attivo.",
        description="Offre calore e piacere, come il sole al tramonto; energia decisa.",
        affect="Calore, beatitudine, energia",
        pole="active",
        source="goethe",
        attribution="Goethe, Zur Farbenlehre (1810), sez. Rosso-Giallo",
        related_emotions=("surprise", "joy"),
        sonic=SonicColorProfile(
            scale_name="mixolydian",
            root_midi=69.0,
            waveform="triangle",
            tempo_bpm=92.0,
            attack_time_ms=120.0,
            release_time_ms=1400.0,
            filter_cutoff_hz=3400.0,
            reverb_mix=0.18,
            arpeggio_pattern=(0, 2, 1, 3, 2, 4),
            sonic_rationale="Energia crescente verso il polo attivo: tempo vivace, timbro più brillante.",
        ),
    ),
    "rotgelb": GoetheColorSpec(
        id="rotgelb",
        name_it="Vermiglio",
        name_de="Rothgelb",
        hsv=(8.0, 92.0, 92.0),
        quote="Il culmine del lato attivo spinge all'azione; è il colore del fuoco e della passione.",
        description="Esercita un fascino che spinge all'azione; energia vitale al suo apice.",
        affect="Potenza, passione, azione",
        pole="active",
        source="goethe",
        attribution="Goethe, Zur Farbenlehre (1810), sez. Giallo-Rosso",
        related_emotions=("anger",),
        sonic=SonicColorProfile(
            scale_name="phrygian",
            root_midi=62.0,
            waveform="triangle",
            tempo_bpm=108.0,
            attack_time_ms=45.0,
            release_time_ms=900.0,
            filter_cutoff_hz=4200.0,
            reverb_mix=0.12,
            arpeggio_pattern=(0, 1, 3, 1, 0, 2),
            sonic_rationale="Fuoco e passione: attacco rapido, scala tesa, registro medio (polo attivo).",
        ),
    ),
    "blau": GoetheColorSpec(
        id="blau",
        name_it="Blu",
        name_de="Blau",
        hsv=(225.0, 42.0, 38.0),
        quote="Il colore più vicino all'oscurità crea una sensazione di vuoto e di distanza.",
        description="Attira lo sguardo verso l'infinito, lasciando un'impressione di solitudine.",
        affect="Malinconia, distanza, freddo",
        pole="passive",
        source="goethe",
        attribution="Goethe, Zur Farbenlehre (1810), sez. Blu",
        related_emotions=("sadness",),
        sonic=SonicColorProfile(
            scale_name="pentatonic_minor",
            root_midi=55.0,
            waveform="sine",
            tempo_bpm=62.0,
            attack_time_ms=520.0,
            release_time_ms=3200.0,
            filter_cutoff_hz=1200.0,
            reverb_mix=0.62,
            arpeggio_pattern=(0, 3, 0, 2, 0),
            sonic_rationale="Vuoto e distanza: registro basso (Marks: scuro→grave), lungo riverbero.",
        ),
    ),
    "blaurot": GoetheColorSpec(
        id="blaurot",
        name_it="Violetto",
        name_de="Blauroth",
        hsv=(275.0, 55.0, 32.0),
        quote="Rispetto al blu puro, questo colore appare più inquieto ed oppressivo.",
        description="Evoca una tristezza che tende all'oppressione; qualcosa di fastidioso nell'aria.",
        affect="Inquietudine, disagio, oppressione",
        pole="passive",
        source="goethe",
        attribution="Goethe, Zur Farbenlehre (1810), sez. Rosso-Blu",
        related_emotions=("fear",),
        sonic=SonicColorProfile(
            scale_name="phrygian",
            root_midi=58.0,
            waveform="sine",
            tempo_bpm=74.0,
            attack_time_ms=180.0,
            release_time_ms=2600.0,
            filter_cutoff_hz=900.0,
            reverb_mix=0.48,
            arpeggio_pattern=(0, 1, 0, 3, 0, 2),
            sonic_rationale="Inquietudine oppressiva: filtro chiuso, intervalli minori stretti.",
        ),
    ),
    "rotblau": GoetheColorSpec(
        id="rotblau",
        name_it="Porpora",
        name_de="Rothblau",
        hsv=(290.0, 70.0, 60.0),
        quote="Il colore della massima dignità, unisce in sé la più grande potenza e maestà.",
        description="Evoca un timore reverenziale e un profondo senso del sublime.",
        affect="Dignità, maestosità, equilibrio universale",
        pole="balanced",
        source="goethe",
        attribution="Goethe, Zur Farbenlehre (1810), sez. Blu-Rosso",
        related_emotions=("awe",),
        sonic=SonicColorProfile(
            scale_name="lydian",
            root_midi=64.0,
            waveform="sine",
            tempo_bpm=68.0,
            attack_time_ms=400.0,
            release_time_ms=2800.0,
            filter_cutoff_hz=2200.0,
            reverb_mix=0.55,
            arpeggio_pattern=(0, 2, 4, 6, 4, 2),
            sonic_rationale="Maestosità e gravità: scala lydian, ampio respiro, riverbero profondo.",
        ),
    ),
    "gruen": GoetheColorSpec(
        id="gruen",
        name_it="Verde",
        name_de="Grün",
        hsv=(128.0, 38.0, 72.0),
        quote="Il prodotto dell'unione tra Giallo e Blu in perfetto equilibrio.",
        description="L'occhio e l'animo trovano in questo colore un riposo reale.",
        affect="Riposo, equilibrio, soddisfazione",
        pole="balanced",
        source="goethe",
        attribution="Goethe, Zur Farbenlehre (1810), sez. Verde",
        related_emotions=("neutral", "trust"),
        sonic=SonicColorProfile(
            scale_name="dorian",
            root_midi=60.0,
            waveform="sine",
            tempo_bpm=72.0,
            attack_time_ms=350.0,
            release_time_ms=2000.0,
            filter_cutoff_hz=2000.0,
            reverb_mix=0.30,
            arpeggio_pattern=(0, 2, 3, 2, 0),
            sonic_rationale="Equilibrio e riposo: parametri moderati, scala dorica neutra.",
        ),
    ),
    # --- Estensioni documentate ---
    "rosa": GoetheColorSpec(
        id="rosa",
        name_it="Rosa",
        name_de="Rosa",
        hsv=(330.0, 35.0, 95.0),
        quote="Un riverbero di luce calda che evoca pura grazia e tenerezza.",
        description="Estensione per l'affetto e la tenerezza, assenti come categoria nella Farbenlehre.",
        affect="Amore, tenerezza, affetto",
        pole="active",
        source="extension",
        attribution="Palmer & Schloss 2010; Collins 2011 (emotion–color: love→rosa/chiaro)",
        related_emotions=("love",),
        sonic=SonicColorProfile(
            scale_name="pentatonic_major",
            root_midi=66.0,
            waveform="sine",
            tempo_bpm=76.0,
            attack_time_ms=200.0,
            release_time_ms=2200.0,
            filter_cutoff_hz=2600.0,
            reverb_mix=0.25,
            arpeggio_pattern=(0, 2, 4, 7, 4),
            sonic_rationale="Tenerezza: timbro morbido, registro medio, arpeggio dolce (mediato da emozione positiva).",
        ),
    ),
    "azzurro": GoetheColorSpec(
        id="azzurro",
        name_it="Azzurro",
        name_de="Hellblau",
        hsv=(195.0, 45.0, 90.0),
        quote="Ci attira verso di sé, donando un senso di vastità e di riposo.",
        description="Estensione per fiducia e sicurezza; il blu goethiano è malinconico, non fiduciario.",
        affect="Fiducia, sicurezza, vastità",
        pole="balanced",
        source="extension",
        attribution="Marks 1974/1987 (luminosità↔pitch); Ou et al. 2018 (calma↔azzurro chiaro)",
        related_emotions=("trust",),
        sonic=SonicColorProfile(
            scale_name="lydian",
            root_midi=68.0,
            waveform="sine",
            tempo_bpm=70.0,
            attack_time_ms=300.0,
            release_time_ms=2400.0,
            filter_cutoff_hz=3000.0,
            reverb_mix=0.28,
            arpeggio_pattern=(0, 4, 2, 4, 0),
            sonic_rationale="Luminosità e calma: pitch medio-alto (Marks), attacco lento, ampio spazio.",
        ),
    ),
    "verde_marcio": GoetheColorSpec(
        id="verde_marcio",
        name_it="Verde Marcio",
        name_de="Grünlich",
        hsv=(70.0, 60.0, 50.0),
        quote="Una mescolanza sgradevole che allontana e altera l'equilibrio.",
        description="Estensione per repulsione e disgusto, qualità non nominate da Goethe.",
        affect="Repulsione, disgusto, rifiuto",
        pole="passive",
        source="extension",
        attribution="Palmer et al. 2013 (disgust→verde-marrone scuro); Ferguson & Brewster 2017 (roughness)",
        related_emotions=("disgust",),
        sonic=SonicColorProfile(
            scale_name="phrygian",
            root_midi=53.0,
            waveform="sawtooth",
            tempo_bpm=58.0,
            attack_time_ms=80.0,
            release_time_ms=1100.0,
            filter_cutoff_hz=650.0,
            reverb_mix=0.15,
            arpeggio_pattern=(0, 1, 0, 1, 3),
            sonic_rationale="Repulsione: timbro ruvido (Ferguson roughness), registro basso, filtro opaco.",
        ),
    ),
    "cremisi": GoetheColorSpec(
        id="cremisi",
        name_it="Cremisi",
        name_de="Karmesin",
        hsv=(345.0, 80.0, 75.0),
        quote="L'oscurarsi verso il rosso incute gravità e una sensazione di imminenza.",
        description="Estensione per attesa e sorpresa attiva; tra polo attivo e gravità goethiana.",
        affect="Attesa, imminenza, tensione attiva",
        pole="active",
        source="extension",
        attribution="Ferguson & Brewster 2017 (sharpness↔urgenza); Palmer 2013 (high-arousal colors)",
        related_emotions=("anticipation", "surprise"),
        sonic=SonicColorProfile(
            scale_name="mixolydian",
            root_midi=63.0,
            waveform="triangle",
            tempo_bpm=100.0,
            attack_time_ms=60.0,
            release_time_ms=1000.0,
            filter_cutoff_hz=3800.0,
            reverb_mix=0.14,
            arpeggio_pattern=(0, 2, 3, 2, 4, 3),
            sonic_rationale="Imminenza e urgenza: tempo sostenuto, attacco netto (sharpness psicoacustico).",
        ),
    ),
}

GOETHE_COLOR_IT: dict[GoetheColorId, str] = {k: v.name_it for k, v in GOETHE_COLORS.items()}

POLE_IT: dict[ColorPole, str] = {
    "active": "Polo attivo (caldo, energico)",
    "passive": "Polo passivo (freddo, distante)",
    "balanced": "Equilibrio tra i poli",
}

# Emozione moderna → colore goethiano (canonico o estensione)
EMOTION_TO_GOETHE: dict[EmotionName, GoetheColorId] = {
    "joy": "gelb",
    "anger": "rotgelb",
    "sadness": "blau",
    "fear": "blaurot",
    "surprise": "cremisi",
    "disgust": "verde_marcio",
    "trust": "azzurro",
    "anticipation": "cremisi",
    "love": "rosa",
    "awe": "rotblau",
    "neutral": "gruen",
}

# FEEL-IT espone solo joy, anger, sadness, fear
FEELIT_EMOTIONS: frozenset[EmotionName] = frozenset({"joy", "anger", "sadness", "fear"})

# Lessico cromatico unificato (unica fonte: parola italiana → colore Goethe)
GOETHE_LEXICON: dict[GoetheColorId, tuple[str, ...]] = {
    "gelb": (
        "gioia", "gioioso", "gioiosa", "felice", "felicità", "allegro", "allegra",
        "lieto", "lieta", "sereno", "serena", "luce", "dorato", "sole",
        "nobiltà", "splendido", "contento", "contenta", "euforia", "meraviglioso",
        "fortunato", "fortunata", "speranza", "grato", "grata", "fiero", "fiera",
        "orgoglioso", "orgogliosa", "ottimista",
    ),
    "gelbrot": (
        "calore", "tramonto", "beatitudine", "vivace", "radioso", "entusiasmo",
        "eccitato", "eccitata", "vitalità", "energia",
    ),
    "rotgelb": (
        "arrabbiato", "arrabiato", "arrabbiata", "arrabiata", "rabbia", "rabbi",
        "furioso", "furiosa", "furiosi", "ira", "collera", "uffa", "passione",
        "fuoco", "fiamma", "incazzato", "incazzata", "odio", "furia", "rabbioso",
        "rabbiosa", "infuriato", "infuriata", "stizzito", "stizzita", "indignato",
        "indignata", "nervoso", "nervosa", "frustrato", "frustrata", "vendetta",
        "rancore", "aggressivo", "ostile", "ostilità",
    ),
    "blau": (
        "triste", "tristezza", "tristi", "piango", "piangere", "piange", "pianto",
        "lacrima", "lacrime", "malinconia", "malinconico", "dolore", "sofferenza",
        "depresso", "depressa", "vuoto", "solitudine", "solo", "sola", "freddo",
        "ombra", "notte", "lutto", "nostalgia", "infinito", "distanza",
        "abbattuto", "abbattuta", "disperato", "disperata", "stanco", "stanca",
        "deluso", "delusa", "annoiato", "annoiata", "noia", "rimpianto", "fragile",
        "rassegnato", "rassegnata",
    ),
    "blaurot": (
        "paura", "ansia", "ansioso", "ansiosa", "terrore", "spavento", "spaventato",
        "inquieto", "inquieta", "inquietudine", "angoscia", "angosciato", "panico",
        "terrorizzato", "agitato", "agitata", "minaccia", "pericolo", "oppressione",
        "disagio", "brivido", "preoccupato", "preoccupata", "incerto", "incerta",
        "confuso", "confusa", "smarrito", "smarrita",
    ),
    "rotblau": (
        "solenne", "sublime", "maestoso", "divino", "maestà", "dignità", "grandioso",
        "riverenza", "rispetto", "sacro", "potente", "immensità", "ammirazione", "incanto",
    ),
    "gruen": (
        "equilibrio", "riposo", "pace", "calma", "tranquillo", "tranquilla", "natura",
        "prato", "soddisfatto", "soddisfatta", "neutro", "neutra", "bilanciato", "armonia",
    ),
    "rosa": (
        "amore", "amato", "amata", "amo", "amare", "cuore", "tenerezza", "tenero",
        "tenera", "affetto", "abbraccio", "bacio", "innamorato", "innamorata", "dolce",
        "grazia", "dolcezza", "passione",
    ),
    "azzurro": (
        "fiducia", "sicuro", "sicura", "sicurezza", "affidabile", "protetto", "protetta",
        "protezione", "rilassato", "rilassata", "serenità", "vastità", "cielo", "aperto",
        "pacifico", "amico", "amichevole", "sincero", "verità",
    ),
    "verde_marcio": (
        "disgusto", "disgustato", "disgustata", "schifo", "ribrezzo", "vomito",
        "ripugnante", "orribile", "ributtante", "nausea", "marcio", "putrido",
    ),
    "cremisi": (
        "sorpresa", "sorpreso", "stupore", "meraviglia", "meravigliato", "inaspettato",
        "shock", "incredibile", "sbalordito", "sbalordita", "allibito", "allibita",
        "sconvolto", "sconvolta", "attesa", "aspettativa", "curiosità", "curioso",
        "curiosa", "impazienza", "impaziente", "imminente", "trepidazione", "attendo",
        "sospeso",
    ),
}


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _is_negated(text_lower: str, match_start: int) -> bool:
    window = text_lower[max(0, match_start - 30): match_start]
    return bool(NEGATION_RE.search(window))


def score_goethe_lexicon(text: str) -> tuple[dict[GoetheColorId, float], list[str]]:
    lowered = text.lower()
    scores: dict[str, float] = {c: 0.0 for c in GOETHE_COLORS}
    matched: list[str] = []

    for color_id, keywords in GOETHE_LEXICON.items():
        for kw in keywords:
            for match in re.finditer(re.escape(kw), lowered):
                weight = 0.25 if _is_negated(lowered, match.start()) else 1.0
                scores[color_id] += weight
                if kw not in matched:
                    matched.append(kw)

    total = sum(scores.values())
    if total <= 0:
        return {}, matched
    return {k: v / total for k, v in scores.items() if v > 0}, matched  # type: ignore[return-value]


def color_scores_to_emotions(
    color_agg: dict[GoetheColorId, float],
) -> dict[EmotionName, float]:
    """Deriva punteggi emotivi dal lessico cromatico unificato."""
    if not color_agg:
        return {}
    emo: dict[str, float] = {}
    for color_id, weight in color_agg.items():
        related = GOETHE_COLORS[color_id].related_emotions
        share = weight / len(related)
        for emotion in related:
            emo[emotion] = emo.get(emotion, 0.0) + share
    total = sum(emo.values()) or 1.0
    return {k: v / total for k, v in emo.items()}  # type: ignore[return-value]


def score_emotion_lexicon(text: str) -> tuple[dict[EmotionName, float], list[str]]:
    """Lessico unificato: stessa scansione del lessico cromatico, vista per emozione."""
    color_agg, matched = score_goethe_lexicon(text)
    return color_scores_to_emotions(color_agg), matched


def emotions_to_goethe_agg(emo_agg: dict[EmotionName, float]) -> dict[GoetheColorId, float]:
    agg: dict[str, float] = {}
    for emotion, weight in emo_agg.items():
        color_id = EMOTION_TO_GOETHE.get(emotion, "gruen")
        agg[color_id] = agg.get(color_id, 0.0) + weight
    total = sum(agg.values()) or 1.0
    return {k: v / total for k, v in agg.items()}  # type: ignore[return-value]


def hybrid_goethe_aggregate(
    model_agg: dict[GoetheColorId, float],
    lex_agg: dict[GoetheColorId, float],
    text: str = "",
    model_weight: float = 0.50,
) -> dict[GoetheColorId, float]:
    if not lex_agg:
        return model_agg

    lex_weight = 1.0 - model_weight
    word_count = len(re.findall(r"\b[\wÀ-ÖØ-öø-ÿ'-]+\b", text, flags=re.UNICODE))
    lex_peak = max(lex_agg.values()) if lex_agg else 0.0
    if word_count <= 6 and lex_peak >= 0.4:
        lex_weight = 0.72
        model_weight = 0.28

    colors = set(model_agg) | set(lex_agg)
    hybrid: dict[str, float] = {}
    for color_id in colors:
        hybrid[color_id] = (
            model_weight * model_agg.get(color_id, 0.0)
            + lex_weight * lex_agg.get(color_id, 0.0)
        )
    total = sum(hybrid.values()) or 1.0
    return {k: v / total for k, v in hybrid.items()}  # type: ignore[return-value]


def get_goethe_blend_weights(agg: dict[GoetheColorId, float]) -> list[tuple[GoetheColorId, float]]:
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return [("gruen", 1.0)]

    top = ranked[:TOP_K_BLEND]
    power = 1.0 if top[0][1] > 0.75 else 0.65
    smoothed = [(c, math.pow(max(w, 1e-9), power)) for c, w in top]
    total = sum(w for _, w in smoothed) or 1.0
    filtered = [(c, w / total) for c, w in smoothed if (w / total) >= MIN_BLEND_WEIGHT]
    if not filtered:
        return [(top[0][0], 1.0)]
    final_total = sum(w for _, w in filtered)
    return [(c, w / final_total) for c, w in filtered]


def blend_hsv(weights: list[tuple[GoetheColorId, float]]) -> tuple[float, float, float]:
    if not weights:
        h, s, v = GOETHE_COLORS["gruen"].hsv
        return h, s, v
    sin_sum = sum(w * math.sin(math.radians(GOETHE_COLORS[c].hsv[0])) for c, w in weights)
    cos_sum = sum(w * math.cos(math.radians(GOETHE_COLORS[c].hsv[0])) for c, w in weights)
    h = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
    s = sum(w * GOETHE_COLORS[c].hsv[1] for c, w in weights)
    v = sum(w * GOETHE_COLORS[c].hsv[2] for c, w in weights)
    return h, s, v


def compute_pole_balance(blend: list[tuple[GoetheColorId, float]]) -> tuple[float, str]:
    """Bilancio polo attivo/passivo goethiano in [-1, 1] (negativo = passivo)."""
    active_w = sum(w for c, w in blend if GOETHE_COLORS[c].pole == "active")
    passive_w = sum(w for c, w in blend if GOETHE_COLORS[c].pole == "passive")
    balanced_w = sum(w for c, w in blend if GOETHE_COLORS[c].pole == "balanced")
    score = active_w - passive_w
    if score > 0.25:
        desc = POLE_IT["active"]
    elif score < -0.25:
        desc = POLE_IT["passive"]
    else:
        desc = POLE_IT["balanced"] + (f" (attivo {active_w:.0%}, passivo {passive_w:.0%}, equilibrio {balanced_w:.0%})")
    return round(max(-1.0, min(1.0, score)), 3), desc


def build_goethe_explanation(
    dominant: GoetheColorId,
    blend: list[tuple[GoetheColorId, float]],
    pole_desc: str,
) -> tuple[str, str, str]:
    spec = GOETHE_COLORS[dominant]
    if len(blend) == 1:
        color_name = spec.name_it
        desc = (
            f"La tinta pura del {spec.name_it} domina la tavolozza. "
            f"{spec.affect}. {pole_desc}."
        )
    else:
        color_name = "Mélange cromatico goethiano"
        others = " e ".join(GOETHE_COLORS[c].name_it for c, w in blend if c != dominant)
        desc = (
            f"Base dominante: {spec.name_it}, sfumata con {others}. "
            f"{pole_desc}."
        )
    return color_name, spec.quote, desc


def build_sonic_from_goethe_blend(
    blend: list[tuple[GoetheColorId, float]],
    confidence: float,
) -> dict[str, Any]:
    """Profilo sonoro come media pesata dei profili cromatici goethiani."""
    if not blend:
        blend = [("gruen", 1.0)]

    dominant = blend[0][0]
    dom_sonic = GOETHE_COLORS[dominant].sonic

    def wavg(attr: str) -> float:
        return sum(w * getattr(GOETHE_COLORS[c].sonic, attr) for c, w in blend)

    root_midi = wavg("root_midi")
    conf_boost = 0.55 + confidence * 0.45
    root_midi = max(50.0, min(80.0, root_midi))

    # Scala e arpeggio dal colore dominante; parametri continui in media
    scale_name = dom_sonic.scale_name
    pattern = list(dom_sonic.arpeggio_pattern)
    intervals = SCALE_INTERVALS[scale_name]
    waveform = dom_sonic.waveform
    if any(GOETHE_COLORS[c].sonic.waveform == "sawtooth" for c, w in blend if w > 0.2):
        waveform = "sawtooth"
    elif sum(w for c, w in blend if GOETHE_COLORS[c].pole == "active") > 0.5:
        waveform = "triangle" if waveform == "sine" else waveform

    voices: list[dict[str, Any]] = []
    tempo = wavg("tempo_bpm")
    attack = wavg("attack_time_ms")
    release = wavg("release_time_ms")
    cutoff = wavg("filter_cutoff_hz")
    reverb = min(0.85, wavg("reverb_mix"))

    for i, degree_idx in enumerate(pattern[:4]):
        semitone = intervals[degree_idx % len(intervals)]
        octave_bump = 12 if i >= 2 and GOETHE_COLORS[dominant].pole == "active" else 0
        midi_note = root_midi + semitone + octave_bump
        gain = (0.32 / (1 + i * 0.35)) * conf_boost * (GOETHE_COLORS[dominant].hsv[2] / 100.0)
        voices.append({
            "pitch_hz": round(midi_to_hz(midi_note), 3),
            "waveform": waveform,
            "gain": round(min(0.45, gain), 4),
            "detune_cents": (-7.0 + i * 5.0) if i > 0 else 0.0,
            "delay_ms": round(i * (180.0 - abs(compute_pole_balance(blend)[0]) * 80.0), 1),
        })

    rationales = [GOETHE_COLORS[c].sonic.sonic_rationale for c, w in blend[:3]]
    return {
        "voices": voices,
        "root_pitch_hz": round(midi_to_hz(root_midi), 3),
        "scale_name": scale_name,
        "tempo_bpm": round(tempo, 1),
        "attack_time_ms": round(attack, 1),
        "release_time_ms": round(release, 1),
        "filter_cutoff_hz": round(cutoff, 1),
        "reverb_mix": round(reverb, 3),
        "arpeggio_pattern": pattern,
        "duration_s": round(2.5 + release / 2000.0, 2),
        "dominant_color": dominant,
        "sonic_rationales": rationales,
    }


def build_lexicon_alignment_goethe(
    model_top_color: GoetheColorId,
    lex_agg: dict[GoetheColorId, float],
    matched: list[str],
    hybrid_top: GoetheColorId,
) -> dict[str, Any]:
    if not lex_agg:
        return {
            "lexicon_active": False,
            "model_top_color": model_top_color,
            "lexicon_top_color": None,
            "contradiction": False,
            "matched_terms": [],
            "note": "Nessuna parola cromatica esplicita: il mix segue il segnale neurale tradotto in colori Goethe.",
        }

    lex_top = max(lex_agg.items(), key=lambda x: x[1])[0]
    matched_str = ", ".join(matched[:4])
    model_name = GOETHE_COLOR_IT[model_top_color]
    lex_name = GOETHE_COLOR_IT[lex_top]

    if lex_top == model_top_color:
        note = f"Le parole nel testo ({matched_str}) confermano la tonalità goethiana del {model_name}."
        contradiction = False
    elif lex_top == hybrid_top:
        note = (
            f"Il modello suggeriva sfumature di {model_name}, ma parole come '{matched_str}' "
            f"orientano il risultato verso il {lex_name}."
        )
        contradiction = True
    else:
        note = (
            f"Parole come '{matched_str}' richiamano il {lex_name}, "
            f"mentre il tono generale propende verso il {model_name}."
        )
        contradiction = True

    return {
        "lexicon_active": True,
        "model_top_color": model_top_color,
        "lexicon_top_color": lex_top,
        "contradiction": contradiction,
        "matched_terms": matched[:8],
        "note": note,
    }


def goethe_color_to_emotion(color_id: GoetheColorId) -> EmotionName:
    return GOETHE_COLORS[color_id].related_emotions[0]
