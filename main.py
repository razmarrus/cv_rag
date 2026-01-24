from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import logging
import time

from config.config import Config
from src.hf_client import HuggingFaceClient
from src.pgvector_client import PgVectorClient
from src.text_processor import TextProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate configuration
Config.validate()

# Initialize FastAPI
app = FastAPI(
    title="CV RAG System",
    description="Ask questions about my CV and experience",
    version="1.0.0"
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global instances (initialized on startup)
hf_client = None
db_client = None
text_processor = None


@app.on_event("startup")
async def startup_event():
    """Initialize clients on startup."""
    global hf_client, db_client, text_processor
    
    logger.info("Initializing RAG components...")
    
    try:
        hf_client = HuggingFaceClient(
            hf_token=Config.HF_TOKEN,
            embedding_model=Config.EMBEDDING_MODEL,
            llm_model=Config.LLM_MODEL
        )
        
        db_client = PgVectorClient(
            connection_string=Config.DATABASE_URL,
            embedding_dim=Config.EMBEDDING_DIM
        )
        
        text_processor = TextProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            max_context_tokens=Config.MAX_CONTEXT_TOKENS
        )
        
        logger.info("RAG components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        raise


def query_rag(question: str) -> dict:
    """
    Execute RAG query pipeline.
    
    Args:
        question: User question
        
    Returns:
        dict with 'answer', 'sources', 'num_chunks', 'execution_time'
    """
    start_time = time.time()
    
    try:
        # Step 1: Generate embedding
        logger.info(f"Step 1: Generating embedding for question: '{question}'")
        query_embedding = hf_client.get_embeddings([question])[0]
        if not isinstance(query_embedding, list):
            query_embedding = query_embedding.tolist()
        logger.info(f"Embedding generated: dimension={len(query_embedding)}")
        
        # Step 2: Search database
        logger.info(f"Step 2: Searching database (k={Config.TOP_K_CHUNKS}, threshold={Config.SIMILARITY_THRESHOLD})")
        chunks = db_client.search(
            query_embedding,
            k=Config.TOP_K_CHUNKS,
            similarity_threshold=Config.SIMILARITY_THRESHOLD
        )
        logger.info(f"Search returned {len(chunks)} chunks")
        
        if chunks:
            for i, chunk in enumerate(chunks):
                logger.info(f"Chunk {i+1}: similarity={chunk.get('similarity', 0):.3f}, source={chunk.get('source', 'unknown')}")
        
        if not chunks:
            logger.warning("No chunks found matching similarity threshold")
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "num_chunks": 0,
                "execution_time": time.time() - start_time
            }
        
        # Step 3: Assemble context
        logger.info("Step 3: Assembling context from chunks")
        context = text_processor.assemble_context(chunks, question=question)
        logger.info(f"Context assembled: {len(context)} characters")
        
        # Step 4: Generate answer
        logger.info("Step 4: Generating answer with LLM")
        answer = hf_client.generate_answer(
            question=question,
            context=context,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE
        )
        logger.info(f"Answer generated: {len(answer)} characters")
        
        execution_time = time.time() - start_time
        
        # TODO: Log query here (db_client.log_query(...))
        
        return {
            "answer": answer,
            "sources": list(set([chunk.get("source", "unknown") for chunk in chunks])),
            "num_chunks": len(chunks),
            "execution_time": execution_time
        }
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise


# Routes

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve main page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Ask Me Anything"
    })


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    """
    Handle question submission.
    
    TODO: Add rate limiting here
    """
    if not question or len(question.strip()) < 3:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Please enter a valid question (at least 3 characters).",
            "question": question
        })
    
    try:
        logger.info(f"Processing question: {question[:100]}...")
        result = query_rag(question)
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "num_chunks": result["num_chunks"],
            "execution_time": f"{result['execution_time']:.2f}"
        })
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "question": question,
            "error": "Sorry, something went wrong. Please try again."
        })


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected" if db_client and db_client.conn else "disconnected"
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down, closing connections...")
    if db_client:
        db_client.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)