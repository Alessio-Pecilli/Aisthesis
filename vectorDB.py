# ingest.py
import chromadb
import ollama

def embedding(file_path = 'data/goethe.txt'):
    
    
    print(f"Leggo i dati dal tuo file: {file_path}...")
    
    # 1. Apre il tuo file IN SOLA LETTURA ("r")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non è stato trovato!")
        return
    
    # Dividiamo il testo in chunk. 
    # Attualmente divide ogni volta che trova una riga vuota (doppio a capo).
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 10]
    print(f"Trovati {len(chunks)} paragrafi da vettorizzare.")
    
    # 2. Inizializza ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Resetta la collection se esiste già, così puoi lanciare questo script 
    # ogni volta che aggiorni il tuo file txt senza creare duplicati nel DB
    try:
        client.delete_collection(name="goethe_colors")
    except Exception:
        pass 
        
    collection = client.create_collection(name="goethe_colors")

    print(f"Inizio embedding... (potrebbe volerci un po' a seconda dell'hardware)")

    # 3. Vettorizza ogni chunk e salvalo in Chroma
    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        embedding = response["embedding"]
        
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
        print(f"Inserito chunk {i+1}/{len(chunks)}")

    print("Ingestion completata con successo! Il DB vettoriale è pronto.")
if __name__ == "__main__":
    # Inserisci il path corretto usando gli slash standard per evitare problemi su Windows
    embedding("data/goethe.txt")