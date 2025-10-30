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
# Aggiunto RunnableParallel per gestire output multipli in LCEL
from langchain_core.runnables import RunnableParallel, RunnablePassthrough 

# Importazioni specifiche di Gemini/Google
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
        model='gemini-1.5-flash',
        google_api_key=GEMINI_API_KEY,
        temperature=0.7
    )
except Exception:
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
            raise ValueError("Credenziali Gemini mancanti. Impossibile inizializzare LLM o Embeddings.")

        # Configurazione embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
             model="models/text-embedding-004",
             google_api_key=GEMINI_API_KEY,
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
        
        logger.info("Sistema RAG inizializzato con successo usando ChromaDB")
        
    except Exception as e:
        logger.error(f"Errore inizializzazione RAG. Il servizio rimarrà online ma l'endpoint /api/query non funzionerà: {e}")
        retriever = None 

# --- Funzioni di supporto per LCEL ---

def format_docs(docs: List[Document]):
    """Formatta i documenti per il prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """Crea e restituisce la catena RAG (LCEL) che restituisce risposta e documenti."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Sistema RAG non inizializzato.")

    # 1. Pipeline di recupero e formattazione del contesto
    context_pipeline = RunnableParallel(
        # 'context' recupera i documenti e li formatta per il prompt
        context=(lambda x: x['question']) | retriever | format_docs,
        # 'documents' recupera i documenti e li restituisce (per il conteggio)
        documents=(lambda x: x['question']) | retriever,
        # 'question' semplicemente fa passare la domanda originale
        question=RunnablePassthrough()
    )

    # 2. Pipeline di generazione della risposta
    response_pipeline = (
        context_pipeline 
        | PROMPT_TEMPLATE 
        | LLM
        | StrOutputParser()
    )
    
    # 3. Combiniamo le pipeline per restituire tutti i risultati necessari
    # Usiamo RunnableParallel per restituire sia la risposta che i documenti
    full_chain = RunnableParallel(
        response=response_pipeline,
        # Estrarre la lista dei documenti dalla 'context_pipeline'
        context_docs=(lambda x: x['documents'])
    )

    return full_chain

# --- Modelli Pydantic ---

class QueryRequest(BaseModel):
    question: str

# Modificato per supportare la risposta del backend che ora è un dizionario più complesso
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
        
        # Eseguiamo l'intera catena LCEL con un solo invoke
        # L'input è il dizionario atteso dalla catena: {"question": "..."}
        result: Dict[str, Any] = rag_chain.invoke({"question": req.question})
        
        # L'output ora è un dizionario: 
        # {'response': 'la risposta generata', 'context_docs': [Doc1, Doc2, ...]}
        
        return QueryResponse(
            response=result['response'].strip(),
            context_docs=len(result['context_docs'])
        )
        
    except HTTPException:
        # Rilanciare le eccezioni HTTPException (come il 503)
        raise
    except Exception as e:
        # Questo catturerà eventuali errori rimanenti e l'errore 'dict' se si ripresenta
        logger.error(f"Errore query finale: {e}")
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