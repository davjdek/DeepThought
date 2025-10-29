import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Configurazione API - USA VARIABILE D'AMBIENTE
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY non configurata")

genai.configure(api_key=GEMINI_API_KEY)

# Variabili globali
retriever = None

def initialize_rag():
    """Inizializza il sistema RAG"""
    global retriever
    
    try:
        # Configurazione embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            model_kwargs={"device": "cpu"}
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
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        logger.info("Sistema RAG inizializzato con successo")
        
    except Exception as e:
        logger.error(f"Errore inizializzazione RAG: {e}")
        raise

def query_gemini(prompt: str) -> str:
    """Interroga l'API Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=200
            )
        )
        return response.text
    except Exception as e:
        logger.error(f"Errore Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Errore API: {e}")

# Template per il prompt
PROMPT_TEMPLATE = """Rispondi alla domanda usando il contesto fornito quando rilevante.
Se il contesto non contiene informazioni pertinenti o sufficienti per rispondere, allora rispondi alla domanda usando la tua conoscenza generale.
Alla domanda 'qual è il senso della vita?' rispondi 42.

Contesto: {context}
Domanda: {question}

Risposta:"""

class QueryRequest(BaseModel):
    question: str

@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    """Endpoint principale per le query"""
    if not retriever:
        raise HTTPException(status_code=503, detail="Sistema RAG non inizializzato")
    
    try:
        # Recupera documenti rilevanti
        docs = retriever.get_relevant_documents(req.question)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Genera prompt e risposta
        prompt = PROMPT_TEMPLATE.format(context=context, question=req.question)
        response = query_gemini(prompt)
        
        return {
            "response": response.strip(),
            "context_docs": len(docs)
        }
        
    except Exception as e:
        logger.error(f"Errore query: {e}")
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
    import os
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)
