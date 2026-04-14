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
from langchain_groq import ChatGroq

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

KB_URL = "https://drive.google.com/uc?export=download&id=1YIDVQmldy2efy3tTwkJwP-sEtTglmnpn"
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY non configurata. L'LLM non funzionerà.")
if not COHERE_API_KEY:
    logger.error("COHERE_API_KEY non configurata. Embedding e Rerank non funzioneranno.")

# ---------------------------------------------------------------------------
# Inizializzazione LLM
# ---------------------------------------------------------------------------

LLM: Optional[ChatGroq] = None

if GROQ_API_KEY:
    try:
        LLM = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=0.7,
            model_kwargs={
                "top_p": 0.9,
                "presence_penalty": 0.8,
                "frequency_penalty": 0.8
            }
        )
        logger.info("LLM inizializzato: Groq llama-3.1-8b-instant")
    except Exception as e:
        logger.error(f"Errore inizializzazione LLM: {e}")
else:
    logger.warning("LLM non inizializzato: chiave Groq mancante.")

# ---------------------------------------------------------------------------
# Variabili globali retriever
# ---------------------------------------------------------------------------

retriever: Optional[Any] = None
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
Rispondi alla domanda usando il contesto fornito quando rilevante, ma senza menzionare esplicitamente il "contesto fornito".


Sei l'assistente di Davide, un esperto professionista di sviluppo web e SEO. Dì chi sei solo quando ti viene chiesto, non dirlo ad ogni risposta.
Usa le seguenti informazioni per rispondere alla domanda dell'utente in modo naturale, diretto e cordiale.

REGOLE RIGIDE:
1. NON menzionare mai frasi come "in base al contesto fornito", "secondo i documenti" o "le informazioni non specificano".
2. Rispondi come se questa fosse la tua conoscenza personale.
3. Se un'informazione specifica (come un prezzo esatto) non è presente, dai una risposta orientativa basata sulla tua esperienza (per esempio spiegando da quali fattori può dipendere il prezzo finale) e invita l'utente a fare un'analisi approfondita con te.
4. Usa il "io" (es. "Io mi occupo di...", "Nel mio approccio...") invece di parlare in terza persona se ti riferisci a Davide.
5. Per approfondimenti, invita l'utente a scriverti dalla pagina dei contatti. Usa il markdown [contatti](https://2025sacquegna.iftscnosfapbologna.it/portfolio/contatti)
6. BREVITÀ: Rispondi in massimo 4-5 frasi brevi. Sii sintetico e vai dritto al punto. Non ripetere più volte la stessa risposta.
7. LINK CLICCABILE: Se fornisci un link, usa il formato Markdown: [testo del link](url). Il testo del link rendilo breve (una o due parole)
8. LINGUA: Rispondi SEMPRE nella STESSA LINGUA usata dall'utente nell'ultima domanda. Se l'utente scrive in inglese, ignora l'italiano del contesto e rispondi ESCLUSIVAMENTE in inglese.
9. Quando ti viene chiesto se sei un chatbot, ammetti di essere un assistente virtuale e non una persona vera. Non dire "sono un chatbot specializzato..." ma "sono l'assistente virtuale di Davide Sacquegna, specializzato..."
10. IF THE QUESTION IN ENGLISH, REPLY IN ENGLISH
11: NON DIRE CHI SEI SE NON TI E' STATO CHIESTO

Alla domanda 'qual è il senso della vita?' rispondi 42.

Storico della Conversazione (usa questo solo per il contesto):
{chat_history}

Contesto:
{context}

DOMANDA ATTUALE (rispondi a questa):  
{question}
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
    formatted = ""
    for msg in chat_history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content")
        formatted += f"\n--- {role} ---\n{content}\n"
    return formatted

# ---------------------------------------------------------------------------
# Self Ping
# ---------------------------------------------------------------------------

async def self_ping():
    url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000") + "/"
    rome = timezone(timedelta(hours=2))

    while True:
        await asyncio.sleep(600)
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
    global retriever, history_aware_retriever

    try:
        if not COHERE_API_KEY:
            raise ValueError("Cohere API Key mancante. Impossibile inizializzare Embeddings.")

        embeddings = CohereEmbeddings(
            model="embed-multilingual-v3.0",
            cohere_api_key=COHERE_API_KEY
        )

        if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            logger.info("Caricamento ChromaDB esistente da disco...")
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )
        else:
            logger.info("ChromaDB non trovato. Avvio caricamento documenti e indicizzazione...")

            loader = WebBaseLoader([
                "https://2025Sacquegna.iftscnosfapbologna.it/portfolio/progetti",
            ])
            web_docs = loader.load()

            pdf_docs = []
            for pdf_url in [KB_URL]:
                temp_path = None
                try:
                    logger.info(f"Download PDF: {pdf_url}")
                    response = requests.get(pdf_url, headers=headers)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(response.content)
                        temp_path = tmp_file.name
                    pdf_loader = PyPDFLoader(temp_path)
                    loaded = pdf_loader.load()
                    pdf_docs += loaded
                    logger.info(f"PDF caricato: {len(loaded)} pagine da {pdf_url}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Errore download PDF {pdf_url}: {e}")
                except Exception as e:
                    logger.error(f"Errore parsing PDF {pdf_url}: {e}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                        logger.info(f"File temporaneo eliminato: {temp_path}")

            all_docs = web_docs + pdf_docs
            logger.info(f"Totale documenti caricati: {len(all_docs)}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(all_docs)
            logger.info(f"Totale chunk creati: {len(splits)}")

            vectorstore = Chroma.from_documents(
                splits,
                embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            logger.info("ChromaDB creato e salvato su disco.")

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        reranker = CohereRerank(
            cohere_api_key=COHERE_API_KEY,
            model="rerank-multilingual-v3.0",
            top_n=5
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

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
    if LLM is None:
        raise HTTPException(
            status_code=503,
            detail="LLM non inizializzato. Controlla GROQ_API_KEY."
        )

    if retriever is None or history_aware_retriever is None:
        logger.warning("RAG non disponibile, risposta senza contesto.")
        answer = LLM.invoke(request.question)
        return {"question": request.question, "answer": answer, "source_documents": []}

    try:
        if not request.chat_history:
            docs = retriever.invoke(request.question)
        else:
            # Passa solo le chiavi necessarie
            docs = history_aware_retriever.invoke({
                "question": request.question,
                "chat_history": request.chat_history
            })

        rag_chain = (
            RunnableParallel(
                context=lambda x: format_docs(x["docs"]),
                question=lambda x: x["question"],
                chat_history=lambda x: format_chat_history(x["chat_history"]),
            )
            | PROMPT_TEMPLATE
            | LLM
            | StrOutputParser()
        )

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