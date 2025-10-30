import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Mantenuto dalla tua versione
from pydantic import BaseModel
from typing import List # Aggiunto per coerenza con le firme di funzione

# LangChain Core/Community Imports
from langchain_core.prompts import ChatPromptTemplate # Cambiato da PromptTemplate per LCEL
from langchain_core.output_parsers import StrOutputParser # Aggiunto per la catena LCEL
from langchain_core.documents import Document # Aggiunto per le firme di funzione
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

# Importazioni specifiche di Gemini/Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI # Aggiunto LLM
# import google.generativeai as genai # RIMOSSO - Passiamo a usare solo le classi LangChain

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Configurazione API - USA VARIABILE D'AMBIENTE
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Cambiato da ValueError a logger.error + flag per non bloccare il servizio subito
    logger.error("GEMINI_API_KEY non configurata come variabile d'ambiente. Il RAG non si avvierà.")

# Modello LLM
LLM = None
try:
    LLM = GoogleGenerativeAI(
        model='gemini-1.5-flash', # Mantenuto il tuo modello
        google_api_key=GEMINI_API_KEY, # Iniezione esplicita della chiave
        temperature=0.7 # Mantenuta la tua temperatura
    )
except Exception:
    # Se la chiave è mancante, LLM rimarrà None e verrà gestito nell'inizializzazione RAG
    pass

# Variabili globali
retriever = None

# Template per il prompt - Usiamo il formato ChatPromptTemplate per LCEL
PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
Rispondi alla domanda usando il contesto fornito quando rilevante.
Se il contesto non contiene informazioni pertinenti o sufficienti per rispondere, allora rispondi alla domanda usando la tua conoscenza generale.
Alla domanda 'qual è il senso della vita?' rispondi 42.

Contesto: {context}
Domanda: {question}
""")


def initialize_rag():
    """Inizializza il sistema RAG utilizzando ChromaDB come Vector Store"""
    global retriever
    
    try:
        if not GEMINI_API_KEY or not LLM:
            raise ValueError("Credenziali Gemini mancanti. Impossibile inizializzare LLM o Embeddings.")

        # Configurazione embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
             model="text-embedding-004",
             google_api_key=GEMINI_API_KEY, # Iniezione esplicita della chiave
        )
        
        # Caricamento documenti - Mantenuti i tuoi URL originali
        loader = WebBaseLoader([
             "https://it.wikipedia.org/wiki/Catalogo_di_Messier",
             "https://it.wikipedia.org/wiki/Galassia_di_Andromeda"
        ])
        docs = loader.load()
        
        # Divisione in chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Creazione vectorstore
        vectorstore = Chroma.from_documents(splits, embeddings) 
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        logger.info("Sistema RAG inizializzato con successo usando ChromaDB")
        
    except Exception as e:
        logger.error(f"Errore inizializzazione RAG. Il servizio rimarrà online ma l'endpoint /api/query non funzionerà: {e}")
        retriever = None # Assicura che retriever sia None in caso di fallimento

# --- Funzioni di supporto per LCEL ---

def format_docs(docs: List[Document]):
    """Formatta i documenti per il prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """Crea e restituisce la catena RAG (LCEL)."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Sistema RAG non inizializzato.")

    # Creazione della catena LCEL (LangChain Expression Language)
    rag_chain = (
        # Recupera il contesto e passa la domanda
        {"context": retriever | format_docs, "question": lambda x: x}
        | PROMPT_TEMPLATE # Il tuo template
        | LLM
        | StrOutputParser()
    )
    return rag_chain

# --- Modelli Pydantic ---

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    # Aggiungo il campo 'context_docs' per coerenza con la tua risposta originale
    response: str
    context_docs: int = 0


# --- Endpoints API ---

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Endpoint principale per le query"""
    if not retriever:
        raise HTTPException(status_code=503, detail="Sistema RAG non inizializzato. Controlla i log di Render per l'errore di inizializzazione.")
    
    try:
        # Recupera documenti rilevanti (necessario per contare i docs)
        docs = retriever.get_relevant_documents(req.question)
        
        # Genera prompt e risposta usando la catena LCEL
        rag_chain = get_rag_chain()
        result = rag_chain.invoke({"question": req.question})
        
        return QueryResponse(
             response=result.strip(),
             context_docs=len(docs)
        )
        
    except Exception as e:
        logger.error(f"Errore query: {e}")
        # Solleva la HTTPException in caso di fallimento
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Controllo stato sistema"""
    return {
        "status": "healthy",
        "rag_initialized": retriever is not None
    }

@app.get("/")
async def root():
    """Endpoint root"""
    return {
        "message": "RAG API con Google Gemini",
        "endpoints": {
            "POST /api/query": "Fai una domanda",
            "GET /api/health": "Stato sistema"
        }
    }

# Inizializzazione all'avvio
@app.on_event("startup")
async def startup_event():
    initialize_rag()

if __name__ == "__main__":
    import uvicorn
    # Importa os qui per coerenza con il blocco if name
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
