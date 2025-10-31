import requests
import logging
import os
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# LangChain Core/Community Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough 
from langchain_core.runnables.base import RunnableSequence

# Importazioni dei Provider
# ➡️ NUOVA IMPORTAZIONE PER EMBEDDING ➡️
from langchain_cohere import CohereEmbeddings 
# Importazione per LLM (Gemini)
from langchain_google_genai import GoogleGenerativeAI

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# URL del PDF
pdf_url = "https://www.codas.it/images/Catalogo%20di%20Messier%20(2).pdf"

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
ALLOWED_ORIGINS = [
    "https://catalogo-messier-angular.onrender.com",  # Frontend production
    "http://localhost:4200",  # Sviluppo locale Angular
    "http://localhost:80",  # Eventuale altro sviluppo locale
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Configurazione API ---
# Chiavi di accesso lette dall'ambiente
COHERE_API_KEY = os.getenv("COHERE_API_KEY") # Per l'Embedding (API Remota)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Per l'LLM

# Controllo iniziale delle chiavi
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY non configurata. L'LLM non funzionerà.")
if not COHERE_API_KEY:
    logger.error("COHERE_API_KEY non configurata. L'Embedding non funzionerà.")


# --- Inizializzazione LLM ---
LLM: Optional[GoogleGenerativeAI] = None
try:
    if GEMINI_API_KEY:
        LLM = GoogleGenerativeAI(
            model='gemini-2.5-flash',
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
    else:
        logger.warning("LLM non inizializzato: Chiave Gemini mancante.")
except Exception as e:
    logger.error(f"Errore inizializzazione LLM: {e}")

# Variabile globale per il retriever
retriever: Optional[Any] = None # Retriever di base
history_aware_retriever: Optional[RunnableSequence] = None # Retriever con memoria


# --- Modello Pydantic per la Richiesta ---
class QuestionRequest(BaseModel):
    question: str
    # Storico come lista di dizionari (es. [{"role": "user", "content": "Ciao"}, ...])
    chat_history: List[Dict[str, str]] = []

# --- Template per il prompt ---
PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
Rispondi alla domanda usando il contesto fornito quando rilevante.
Se il contesto non contiene informazioni pertinenti o sufficienti per rispondere, allora rispondi alla domanda usando la tua conoscenza generale.
Alla domanda 'qual è il senso della vita?' rispondi 42.

Contesto: {context}
Domanda: {question}
""")
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template("""
Dato lo storico della conversazione e la successiva domanda, riformula la domanda successiva come una domanda standalone (autonoma).
Se non c'è storico, restituisci solo la domanda successiva.

Storico della Conversazione:
{chat_history}

Domanda Successiva: {question}
""")

# --- Funzioni di supporto per LCEL ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Converte lo storico da lista di dict a stringa leggibile per il prompt."""
    formatted_history = []
    for message in chat_history:
        role = message.get("role", "user").capitalize() # Utente o AI
        content = message.get("content", "")
        formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history)


# --- Funzione di Inizializzazione RAG ---
def initialize_rag():
    """Inizializza il sistema RAG utilizzando ChromaDB e Cohere Embeddings (API remota)"""
    global retriever
    global history_aware_retriever # Rendiamo accessibile la variabile globale
    # Nella funzione initialize_rag() o come variabile globale
    
    try:
        if not COHERE_API_KEY:
             raise ValueError("Cohere API Key mancante. Impossibile inizializzare Embeddings.")
        
        # 🚨 EMBEDDING REMOTO CON COHERE 🚨
        embeddings = CohereEmbeddings(
            model="embed-multilingual-v3.0",
            # Passa esplicitamente la chiave (anche se di solito viene letta dall'ambiente)
            cohere_api_key=COHERE_API_KEY 
        )
        
        # Caricamento documenti
        logger.info("Avvio caricamento documenti...")
        loader = WebBaseLoader([
             "https://it.wikipedia.org/wiki/Catalogo_di_Messier",
             "https://it.wikipedia.org/wiki/Galassia_di_Andromeda"       
        ])
        web_docs = loader.load()
        
        # 2. Gestione del PDF in modo temporaneo
        pdf_docs = []
        logger.info("Avvio scaricamento e caricamento PDF temporaneo...")

        try:
            # Scarica il contenuto del PDF
            response = requests.get(pdf_url, headers=headers)
            response.raise_for_status()

            # Crea un file temporaneo sul disco
            # Lo aprirà, lo scriverà, e lo chiuderà automaticamente al termine del blocco 'with'
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name
            
            # Usa PyPDFLoader con il percorso del file temporaneo
            pdf_loader = PyPDFLoader(temp_path)
            pdf_docs = pdf_loader.load()
            logger.info(f"PDF caricato e analizzato. Documenti trovati: {len(pdf_docs)}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Errore durante lo scaricamento del PDF: {e}")
        except Exception as e:
            logger.error(f"Errore durante l'analisi del PDF: {e}")

        finally:
            # IMPORTANTE: Pulisci il file temporaneo dopo averlo usato
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"File temporaneo eliminato: {temp_path}")

        # 3. Unione dei documenti
        docs = web_docs + pdf_docs
        logger.info(f"Caricamento completato. Numero totale di documenti: {len(docs)}")

        # Divisione in chunks
        logger.info("Divisione in chunks e creazione Vector Store...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Creazione vectorstore (USA LA COHERE API, NON LA RAM LOCALE)
        vectorstore = Chroma.from_documents(splits, embeddings) 
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        if LLM:
            history_aware_retriever = (
                # 1. Combina storico e nuova domanda
                RunnablePassthrough.assign(
                    chat_history=lambda x: format_chat_history(x["chat_history"])
                )
                # 2. Rielabora la domanda con l'LLM
                | CONDENSE_QUESTION_PROMPT
                | LLM
                | StrOutputParser()
                # 3. Usa la domanda riformulata per il recupero
                | retriever
            ).with_config(run_name="HistoryAwareRetriever")
        else:
            # Se LLM non è disponibile, usa il retriever standard come fallback
            history_aware_retriever = retriever

        
        logger.info("Sistema RAG inizializzato con successo usando Cohere Embeddings (API remota)")
        
    except Exception as e:
        logger.error(f"Errore inizializzazione RAG: {e}")
        retriever = None 
        history_aware_retriever = None



# --- Endpoint principale ---
@app.post("/ask", response_model=Dict[str, Any])
async def ask_question(request: QuestionRequest):
    if history_aware_retriever is None:
        # Se il RAG è fallito all'avvio (ad esempio, per una chiave mancante)
        logger.warning("Retriever RAG non inizializzato. Uso solo l'LLM di base.")
        
        if LLM is None:
            raise HTTPException(status_code=503, detail="LLM non inizializzato. Controlla GEMINI_API_KEY.")
            
        # Risposta base senza contesto
        response = LLM.invoke(request.question)
        return {"question": request.question, "answer": response, "source_documents": []}


    # Definizione della catena RAG (se il retriever è inizializzato)
    rag_chain = (
        RunnableParallel(
            # Il retriever prende ora question E chat_history
            context=history_aware_retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | PROMPT_TEMPLATE
        | LLM
        | StrOutputParser()
    )

    try:
        # Recupera i documenti per il campo 'source_documents' del JSON di risposta
        docs = history_aware_retriever.invoke(request.dict())
        
        # Esegue la catena RAG
        answer = rag_chain.invoke(request.dict())
        
        return {
            "question": request.question,
            "answer": answer,
            "source_documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        }
    
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione della catena RAG: {e}")
        raise HTTPException(status_code=500, detail=f"Errore interno del server durante l'esecuzione del RAG: {e}")


# --- Inizializzazione all'avvio ---
@app.on_event("startup")
async def startup_event():
    # Inizializza il RAG solo se la chiave Cohere è disponibile
    if COHERE_API_KEY:
        initialize_rag()
    else:
        logger.warning("RAG non inizializzato: Chiave API Cohere mancante.")


# --- Esecuzione Uvicorn (per testing locale) ---
if __name__ == "__main__":
    import uvicorn
    # Assicurati che PORT sia disponibile nell'ambiente per Render
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Avvio del server su http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)