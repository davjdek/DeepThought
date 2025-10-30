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
# RIMOZIONE DI OPENAI E HUGGING FACE

# Importazioni specifiche di Gemini/Google
# 🚨 IMPORTAZIONE CORRETTA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Configurazione API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY non configurata come variabile d'ambiente. Il RAG non si avvierà.")

# Modello LLM
LLM = None
try:
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
        # CONTROLLO FINALE: Se LLM è null, le credenziali sono fallite
        if not GEMINI_API_KEY or not LLM:
             raise ValueError("Credenziali Gemini mancanti o LLM fallito all'avvio.")
        
        # 🚨 EMBEDDING REMOTO DI GEMINI 🚨
        embeddings = GoogleGenerativeAIEmbeddings(
             model="embedding-001", 
             google_api_key=GEMINI_API_KEY,
        )
        
        # Caricamento documenti
        logger.info("Avvio caricamento documenti...")
        loader = WebBaseLoader([
             "https://it.wikipedia.org/wiki/Catalogo_di_Messier",
             "https://it.wikipedia.org/wiki/Galassia_di_Andromeda"
        ])
        docs = loader.load()
        
        # Divisione in chunks
        logger.info("Divisione in chunks e creazione Vector Store...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Creazione vectorstore
        vectorstore = Chroma.from_documents(splits, embeddings) 
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        logger.info("Sistema RAG inizializzato con successo usando Gemini Embeddings (API remota)")
        
    except Exception as e:
        logger.error(f"Errore inizializzazione RAG: {e}")
        retriever = None 

# --- Funzioni di supporto per LCEL (RIMASTE INVARIATE) ---
# ... (il resto del codice non è cambiato)
# ...

# Inizializzazione all'avvio
@app.on_event("startup")
async def startup_event():
    # Inizializza il RAG solo se la chiave API è disponibile
    if GEMINI_API_KEY:
        initialize_rag()
    else:
        logger.warning("RAG non inizializzato: Chiave API Gemini mancante.")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Avvio del server su http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)