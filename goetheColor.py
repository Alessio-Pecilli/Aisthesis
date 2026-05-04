import ollama
from pydantic import BaseModel, Field
from enum import Enum

class GoethePolarity(str, Enum):
    PLUS = "plus"
    MINUS = "minus"
    NEUTRAL = "neutral"

class GoetheQueryExtractor(BaseModel):
    emozione: str = Field(..., description="L'emozione principale provata nella frase.")
    polarita: GoethePolarity = Field(..., description="Polarità goethiana: PLUS, MINUS o NEUTRAL.")
    search_query: str = Field(..., description="Query in italiano ottimizzata per cercare nel testo di Goethe con parole chiave goethiane obbligatorie.")

PLUS_QUERY = "luce e calore, sole, energia, attivita, vitalita, forza, fuoco, gioioso, eccitante"
MINUS_QUERY = "ombra e freddo, oscurita, lontananza, solitudine, malinconia, inquietante, dolorosa, lato del meno"
NEUTRAL_QUERY = "equilibrio, quiete, soddisfazione, natura, riposo, giallo e blu"

def _force_goethe_query(result: GoetheQueryExtractor) -> GoetheQueryExtractor:
    """Mantiene deterministiche le parole chiave RAG anche se l'LLM devia."""
    if result.polarita == GoethePolarity.PLUS:
        result.search_query = PLUS_QUERY
    elif result.polarita == GoethePolarity.MINUS:
        result.search_query = MINUS_QUERY
    else:
        result.search_query = NEUTRAL_QUERY
    return result

def parse_emotion(text: str) -> GoetheQueryExtractor:
    system_prompt = """Sei il modulo NLP del progetto Aisthesis.
Devi analizzare la frase utente e popolare ESATTAMENTE i tre campi richiesti.

REGOLE RIGIDE:
1. 'emozione': Inserisci SOLO il nome dell'emozione in italiano (es. Rabbia, Gioia, Tristezza, Ansia, Euforia). NON inserire mai le parole 'plus' o 'minus' in questo campo.
2. 'polarita': Classifica l'emozione secondo Goethe. Usa 'plus' per stati attivi/energici/caldi; usa 'minus' per stati passivi/malinconici/freddi.
3. 'search_query': NON descrivere l'emozione moderna. NON inserire il nome dell'emozione. NON usare parole non presenti nella teoria dei colori.
La query deve essere una stringa composta SOLO da parole chiave goethiane, separate da virgole.

VOCABOLARIO OBBLIGATORIO PER LA QUERY:
- Se la polarità è "plus", la query DEVE iniziare con: "luce e calore"
  e deve contenere almeno queste parole: "sole", "energia", "attivita", "vitalita", "forza", "fuoco".
  Non usare MAI nella query plus: "blu", "ombra", "freddo", "oscurita", "malinconia", "lontananza".
- Se la polarità è "minus", la query DEVE iniziare con: "ombra e freddo"
  e deve contenere almeno queste parole: "oscurita", "lontananza", "solitudine", "malinconia", "inquietante".
  Non usare MAI nella query minus: "luce", "calore", "sole", "energia", "fuoco".
- Se la polarità è "neutral", la query DEVE iniziare con: "equilibrio e quiete"
  e deve contenere: "soddisfazione", "natura", "riposo".

Esempi obbligatori:
Input: "Sono in piena euforia"
Output: {"emozione": "Euforia", "polarita": "plus", "search_query": "luce e calore, sole, energia, attivita, vitalita, forza, fuoco, gioioso, eccitante"}

Input: "Mi sento soffocare dal vuoto"
Output: {"emozione": "Angoscia", "polarita": "minus", "search_query": "ombra e freddo, oscurita, lontananza, solitudine, malinconia, inquietante, dolorosa, lato del meno"}
"""

    # 1. Effettuiamo la chiamata a Ollama e chiudiamo correttamente la parentesi
    response = ollama.chat(
        model="llama3",
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': text}
        ],
        format=GoetheQueryExtractor.model_json_schema(),
        options={
            'temperature': 0.1, # Bassa entropia per output deterministico
            'num_predict': 150  # Limitiamo i token massimi in output per velocizzare
        }
    ) # <--- Parentesi chiusa qui!

    # 2. Ora estraiamo la stringa e la validiamo
    json_string = response['message']['content']
    return _force_goethe_query(GoetheQueryExtractor.model_validate_json(json_string))
