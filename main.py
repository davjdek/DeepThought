import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# LangChain Core/Community Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough 
from langchain_core.runnables.base import RunnableSequence

# ❌ RIMOZIONE: OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings

# Importazioni specifiche di Gemini/Google
# ❌ RIMOZIONE: GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI

# 🚨 NUOVO IMPORT: Hugging Face per l'embedding
from langchain_huggingface import HuggingFaceHubEmbeddings

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Configurazione API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# ❌ RIMOZIONE: OPENAI_API_KEY
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 🚨 NUOVA CHIAVE RICHIESTA 🚨
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") 

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY non configurata come variabile d'ambiente. Il RAG non si avvierà.")
if not HUGGINGFACEHUB_API_TOKEN:
    logger.error("HUGGINGFACEHUB_API_TOKEN non configurata. L'embedding non funzionerà.")

# Modello LLM
LLM = None
try:
    # 🚨 CORREZIONE NOME MODELLO: gemini-2.5-flash è raccomandato per stabilità
    LLM = GoogleGenerativeAI(
        model='gemini-2.5-flash',
        google_api_key=GEMINI_API_KEY,
        temperature=0.7
    )
except Exception as e:
    logger.error(f"Errore inizializzazione LLM: {e}")
    pass

# Variabili globali
retriever = None

# Template per il prompt
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
            raise ValueError("Credenziali Gemini mancanti. Impossibile inizializzare LLM.")
        if not HUGGINGFACEHUB_API_TOKEN:
             raise ValueError("Hugging Face API Token mancante. Impossibile inizializzare Embeddings.")
        
        # 🚨 CAMBIO DEL PROVIDER DI EMBEDDING A HUGGING FACE 🚨
        embeddings = HuggingFaceHubEmbeddings(
            # NON DEVI PASSARE api_key qui. La classe legge automaticamente la variabile
            # HUGGINGFACEHUB_API_TOKEN dall'ambiente, a patto che sia impostata.
            model='sentence-transformers/all-MiniLM-L12-v2' 
)       
        
        # Caricamento documenti
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
        
        logger.info("Sistema RAG inizializzato con successo usando Hugging Face Embeddings")
        
    except Exception as e:
        logger.error(f"Errore inizializzazione RAG: {e}")
        retriever = None 

# --- Funzioni di supporto per LCEL ---

def format_docs(docs: List[Document]):
    """Formatta i documenti per il prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain() -> RunnableSequence:
    """Crea e restituisce la catena RAG (LCEL) che restituisce risposta e documenti."""
    if not retriever:
        # Se il controllo fallisce, verrà sollevata una 503 nell'endpoint
        raise Exception("Retriever non disponibile.")

    # 1. Pipeline di recupero e formattazione del contesto
    setup_and_retrieval = RunnableParallel(
        context=(RunnablePassthrough() | retriever | format_docs),
        documents=(RunnablePassthrough() | retriever),
        question=RunnablePassthrough(),
    ).with_config(run_name="SetupAndRetrieval")
    
    # 2. Pipeline di generazione della risposta
    response_pipeline = (
        setup_and_retrieval.pick("context", "question") # Seleziona solo context e question per il prompt
        | PROMPT_TEMPLATE 
        | LLM
        | StrOutputParser()
    ).with_config(run_name="ResponseGeneration")
    
    # 3. Catena Finale: Unisce la risposta con i documenti grezzi
    full_chain = RunnableParallel(
        response=response_pipeline,
        context_docs=(lambda x: setup_and_retrieval.invoke(x)['documents'])
    ).with_config(run_name="FullRAGChain")

    return full_chain

# --- Modelli Pydantic ---

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str
    context_docs: int = 0


# --- Endpoints API ---

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Endpoint principale per le query"""
    if not retriever:
        raise HTTPException(status_code=503, detail="Sistema RAG non inizializzato. Controlla i log per l'errore di inizializzazione.")
    
    # Validazione robusta (400 Bad Request)
    if not isinstance(req.question, str) or not req.question.strip():
        logger.error(f"Errore 400: La domanda non è una stringa valida: {req.question}")
        raise HTTPException(
            status_code=400, 
            detail="La domanda non è stata fornita in formato stringa valido o è vuota."
        )
    
    try:
        rag_chain = get_rag_chain()
        
        # Eseguiamo l'intera catena LCEL
        result: Dict[str, Any] = rag_chain.invoke({"question": req.question})
        
        return QueryResponse(
            response=result['response'].strip(),
            context_docs=len(result['context_docs'])
        )
        
    except HTTPException:
        # Rilanciare le eccezioni HTTPException (come il 503/400)
        raise
    except Exception as e:
        # Cattura e logga l'errore 500
        logger.error(f"Errore query finale: {e}")
        # Solleva la HTTPException in caso di fallimento
        raise HTTPException(status_code=500, detail=f"Errore interno del server: {str(e)}")

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
    # Inizializza il RAG solo se la chiave API è disponibile
    if GEMINI_API_KEY and HUGGINGFACEHUB_API_TOKEN:
        initialize_rag()
    else:
        logger.warning("RAG non inizializzato: Chiavi API mancanti (Gemini o HuggingFace).")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Avvio del server su http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)