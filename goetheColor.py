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
    search_query: str = Field(..., description="Query in italiano ottimizzata per cercare nel testo di Goethe.")

def parse_emotion(text: str) -> GoetheQueryExtractor:
    system_prompt = """Sei il modulo NLP del progetto Aisthesis.
Devi analizzare la frase utente e popolare ESATTAMENTE i tre campi richiesti.

REGOLE RIGIDE:
1. 'emozione': Inserisci SOLO il nome dell'emozione in italiano (es. Rabbia, Gioia, Tristezza, Ansia, Euforia). NON inserire mai le parole 'plus' o 'minus' in questo campo.
2. 'polarita': Classifica l'emozione secondo Goethe. Usa 'plus' per stati attivi/energici/caldi; usa 'minus' per stati passivi/malinconici/freddi.
3. 'search_query': DIMENTICA le parole dell'emozione moderna. Traduci l'emozione usando ESCLUSIVAMENTE il vocabolario di Goethe. 
Se la polarità è PLUS, usa parole come: "luce, caldo, fuoco, sole, forza attiva, eccitazione, azione, energia". 
Se la polarità è MINUS, usa parole come: "oscurità, freddo, ombra, passività, privazione, lontananza, irrequietezza, vuoto".

Esempio:
Input: "Mi sento soffocare dal vuoto"
Output: {"emozione": "Angoscia", "polarita": "minus", "search_query": "colori freddi legati al vuoto e all'angoscia"}
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
    return GoetheQueryExtractor.model_validate_json(json_string)
