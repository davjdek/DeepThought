import asyncio
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
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain.retrievers import ContextualCompressionRetriever

# Importazioni dei Provider
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_google_genai import GoogleGenerativeAI

# Importazioni necessarie per il self ping
import httpx
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Configurazione base
# ---------------------------------------------------------------------------

headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/91.0.4472.124 Safari/537.36'
    )
}

PDF_URL = "https://www.codas.it/images/Catalogo%20di%20Messier%20(2).pdf"
CHROMA_PERSIST_DIR = "./chroma_db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI()

ALLOWED_ORIGINS = [
    "https://catalogo-messier-angular.onrender.com",
    "http://localhost:4200",
    "http://localhost:80",
    "http://localhost",
    "https://2025sacquegna.iftscnosfapbologna.it",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Chiavi API
# ---------------------------------------------------------------------------

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY non configurata. L'LLM non funzionerà.")
if not COHERE_API_KEY:
    logger.error("COHERE_API_KEY non configurata. Embedding e Rerank non funzioneranno.")

# ---------------------------------------------------------------------------
# Inizializzazione LLM
# ---------------------------------------------------------------------------

LLM: Optional[GoogleGenerativeAI] = None
try:
    if GEMINI_API_KEY:
        LLM = GoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
    else:
        logger.warning("LLM non inizializzato: chiave Gemini mancante.")
except Exception as e:
    logger.error(f"Errore inizializzazione LLM: {e}")

# ---------------------------------------------------------------------------
# Variabili globali retriever
# ---------------------------------------------------------------------------

# Retriever finale (base + rerank)
retriever: Optional[Any] = None
# Retriever history-aware (condensa domanda + retriever finale)
history_aware_retriever: Optional[RunnableSequence] = None

# ---------------------------------------------------------------------------
# Modello Pydantic
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = []

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
Rispondi alla domanda usando il contesto fornito quando rilevante.
Se il contesto non contiene informazioni pertinenti o sufficienti per rispondere,
allora rispondi alla domanda usando la tua conoscenza generale.
Alla domanda 'qual è il senso della vita?' rispondi 42.

Storico della Conversazione:
{chat_history}

Contesto:
{context}

Domanda: {question}
""")

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template("""
Dato lo storico della conversazione e la successiva domanda,
riformula la domanda successiva come una domanda standalone (autonoma).
Se non c'è storico, restituisci solo la domanda successiva invariata.

Storico della Conversazione:
{chat_history}

Domanda Successiva: {question}
""")

# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Converte lo storico da lista di dict a stringa leggibile per il prompt."""
    lines = []
    for message in chat_history:
        role = message.get("role", "user").capitalize()
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Self Ping
# ---------------------------------------------------------------------------
async def self_ping():
    """
    Autopinga il servizio ogni 10 minuti per evitare lo spin-down di Render.
    Attivo solo dalle 7:00 alle 23:00 (ora italiana) per non sprecare risorse di notte.
    """
    url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000") + "/"
    rome = timezone(timedelta(hours=2))  # UTC+2 ora legale, regola a +1 in inverno

    while True:
        await asyncio.sleep(600)  # attendi 10 minuti
        ora = datetime.now(rome).hour

        if 7 <= ora < 23:
            try:
                async with httpx.AsyncClient() as client:
                    await client.get(url, timeout=10)
                logger.info(f"Self-ping eseguito ({ora}:xx).")
            except Exception as e:
                logger.warning(f"Self-ping fallito: {e}")
        else:
            logger.info(f"Self-ping saltato, fuori fascia oraria ({ora}:xx).")

# ---------------------------------------------------------------------------
# Inizializzazione RAG
# ---------------------------------------------------------------------------

def initialize_rag():
    """
    Inizializza il sistema RAG con:
    - Cohere Embeddings (remoto, multilingua)
    - ChromaDB persistente su disco
    - Retriever base con k=20
    - CohereRerank per selezionare i top_n=5 chunk più rilevanti
    - History-aware retriever (bypassa la condensazione se lo storico è vuoto)
    """
    global retriever, history_aware_retriever

    try:
        if not COHERE_API_KEY:
            raise ValueError("Cohere API Key mancante. Impossibile inizializzare Embeddings.")

        # --- Embeddings ---
        embeddings = CohereEmbeddings(
            model="embed-multilingual-v3.0",
            cohere_api_key=COHERE_API_KEY
        )

        # --- Vectorstore: carica da disco se già esistente, altrimenti crea ---
        if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            logger.info("Caricamento ChromaDB esistente da disco...")
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )
        else:
            logger.info("ChromaDB non trovato. Avvio caricamento documenti e indicizzazione...")

            # 1. Documenti web
            loader = WebBaseLoader([
                "https://it.wikipedia.org/wiki/Catalogo_di_Messier",
                "https://it.wikipedia.org/wiki/Galassia_di_Andromeda"
            ])
            web_docs = loader.load()

            # 2. PDF remoto (file temporaneo)
            pdf_docs = []
            temp_path = None
            try:
                response = requests.get(PDF_URL, headers=headers)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(response.content)
                    temp_path = tmp_file.name
                pdf_loader = PyPDFLoader(temp_path)
                pdf_docs = pdf_loader.load()
                logger.info(f"PDF caricato: {len(pdf_docs)} pagine.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Errore download PDF: {e}")
            except Exception as e:
                logger.error(f"Errore parsing PDF: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info("File temporaneo PDF eliminato.")

            # 3. Unione e split
            all_docs = web_docs + pdf_docs
            logger.info(f"Totale documenti caricati: {len(all_docs)}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_docs)
            logger.info(f"Totale chunk creati: {len(splits)}")

            # 4. Creazione vectorstore persistente
            vectorstore = Chroma.from_documents(
                splits,
                embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            logger.info("ChromaDB creato e salvato su disco.")

        # --- Retriever base: recupera i 20 candidati ---
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

        # --- Reranker Cohere: seleziona i 5 più rilevanti tra i 20 ---
        reranker = CohereRerank(
            cohere_api_key=COHERE_API_KEY,
            model="rerank-multilingual-v3.0",
            top_n=5
        )

        # --- Retriever finale: base + rerank ---
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

        # --- History-aware retriever ---
        if LLM:
            history_aware_retriever = (
                RunnablePassthrough.assign(
                    chat_history=lambda x: format_chat_history(x["chat_history"])
                )
                | CONDENSE_QUESTION_PROMPT
                | LLM
                | StrOutputParser()
                | retriever
            ).with_config(run_name="HistoryAwareRetriever")
        else:
            history_aware_retriever = retriever

        logger.info("Sistema RAG inizializzato con successo (Cohere Embeddings + Rerank).")

    except Exception as e:
        logger.error(f"Errore inizializzazione RAG: {e}")
        retriever = None
        history_aware_retriever = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "API RAG Catalogo Messier è online"}


@app.post("/ask", response_model=Dict[str, Any])
async def ask_question(request: QuestionRequest):
    # Fallback: se il RAG non è disponibile ma l'LLM sì, risponde senza contesto
    if LLM is None:
        raise HTTPException(
            status_code=503,
            detail="LLM non inizializzato. Controlla GEMINI_API_KEY."
        )

    if retriever is None or history_aware_retriever is None:
        logger.warning("RAG non disponibile, risposta senza contesto.")
        answer = LLM.invoke(request.question)
        return {"question": request.question, "answer": answer, "source_documents": []}

    try:
        # 1. RECUPERO DOCUMENTI
        # Bypass della condensazione LLM se lo storico è vuoto → risparmio di latenza
        if not request.chat_history:
            docs = retriever.invoke(request.question)
        else:
            docs = history_aware_retriever.invoke(request.dict())

        # 2. CATENA DI GENERAZIONE
        rag_chain = (
            RunnableParallel(
                context=lambda x: format_docs(x["docs"]),
                question=lambda x: x["question"],           # ✅ Fix: estrae solo la domanda
                chat_history=lambda x: format_chat_history(x["chat_history"]),
            )
            | PROMPT_TEMPLATE
            | LLM
            | StrOutputParser()
        )

        # 3. ESECUZIONE
        answer = rag_chain.invoke({
            "docs": docs,
            "question": request.question,
            "chat_history": request.chat_history,
        })

        return {
            "question": request.question,
            "answer": answer,
            "source_documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ]
        }

    except Exception as e:
        logger.error(f"Errore durante l'esecuzione della catena RAG: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Errore interno del server: {e}"
        )

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    if COHERE_API_KEY:
        initialize_rag()
    else:
        logger.warning("RAG non inizializzato: chiave API Cohere mancante.")
    asyncio.create_task(self_ping())

# ---------------------------------------------------------------------------
# Esecuzione locale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Avvio server su http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
