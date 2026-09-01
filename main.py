from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import BackgroundTasks
import logging
import random
import time
from datetime import datetime

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
            llm_model=Config.LLM_MODEL,
            use_local_embeddings=Config.USE_LOCAL_EMBEDDINGS,
            provider=Config.HF_PROVIDER,
        )
        
        db_client = PgVectorClient(
            connection_string=Config.DATABASE_URL,
            embedding_dim=hf_client.embedding_dim
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


def _is_personal_context(chunks: list) -> bool:
    """True when retrieved chunks are from personal-life sections."""
    markers = (
        "chunk_08_personal",
        "chunk_09_more_about",
        "chunk_10_games",
        "chunk_11_music",
        "chunk_12_films",
        "chunk_13_teaching",
        "chunk_14_rick",
    )
    for chunk in chunks:
        content = chunk.get("content", "").lower()
        if any(marker in content for marker in markers):
            return True
    return False


def _pick_deflect_mode() -> str:
    """Pick prose or poetry deflect for questions not in portfolio docs."""
    return "deflect_poetry" if random.random() < 0.5 else "deflect"


def query_rag(question: str, from_pill: bool = False) -> dict:
    """Run embedding, retrieval, and answer generation."""
    start_time = time.time()
    include_contact = not from_pill

    try:
        logger.info(f"Generating embedding for: '{question}'")
        
        with hf_client.embedding_context(question) as query_embedding:
            logger.info(f"Embedding generated: dimension={len(query_embedding)}")
            
            # Step 2: Search database
            logger.info(f"Step 2: Searching database (k={Config.TOP_K_CHUNKS}, threshold={Config.SIMILARITY_THRESHOLD})")
            chunks = db_client.search(
                query_embedding,
                k=Config.TOP_K_CHUNKS,
                similarity_threshold=Config.SIMILARITY_THRESHOLD
            )
            logger.info(f"Normal search returned {len(chunks)} chunks")
            
            # Step 2b: Relaxed search if no results
            if not chunks:
                logger.info(f"No results from normal search. Trying relaxed search (threshold={Config.RELAXED_SIMILARITY_THRESHOLD})")
                chunks = db_client.search(
                    query_embedding,
                    k=Config.TOP_K_CHUNKS,
                    similarity_threshold=Config.RELAXED_SIMILARITY_THRESHOLD
                )
                if chunks:
                    logger.info(f"Relaxed search returned {len(chunks)} chunks")

        if chunks:
            for i, chunk in enumerate(chunks):
                logger.info(f"Chunk {i+1}: similarity={chunk.get('similarity', 0):.3f}, source={chunk.get('source', 'unknown')}")
        
        if not chunks:
            prompt_mode = _pick_deflect_mode()
            logger.warning(f"No chunks found - deflecting ({prompt_mode})")
            answer = hf_client.generate_answer(
                question=question,
                context="",
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.OFF_TOPIC_TEMPERATURE,
                prompt_mode=prompt_mode,
                include_contact=include_contact,
            )
            source_label = "not_in_portfolio"
            execution_time = time.time() - start_time
            logger.info(f"Fallback answer generated ({len(answer)} chars)")
            return {
                "answer": answer,
                "sources": [source_label],
                "num_chunks": 0,
                "execution_time": execution_time
            }
        
        # Step 3: Assemble context
        logger.info("Step 3: Assembling context from chunks")
        context = text_processor.assemble_context(chunks, question=question)
        logger.info(f"Context assembled: {len(context)} characters")
        
        # Step 4: Generate answer from retrieved context
        personal = _is_personal_context(chunks)
        answer = hf_client.generate_answer(
            question=question,
            context=context,
            max_new_tokens=(
                Config.MAX_PERSONAL_NEW_TOKENS if personal else Config.MAX_NEW_TOKENS
            ),
            temperature=(
                Config.PERSONAL_TEMPERATURE if personal else Config.TEMPERATURE
            ),
            prompt_mode="personal" if personal else "standard",
            include_contact=include_contact,
        )
        logger.info(f"Answer generated: {len(answer)} characters")
        
        execution_time = time.time() - start_time

        return {
            "answer": answer,
            "sources": list(set([chunk.get("source", "unknown") for chunk in chunks])),
            "num_chunks": len(chunks),
            "execution_time": execution_time
        }
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Extract IP and get current usage
    user_ip = request.client.host
    if request.headers.get("X-Forwarded-For"):
        user_ip = request.headers.get("X-Forwarded-For").split(",")[0].strip()
    
    query_count = db_client.get_daily_query_count(user_ip)
    remaining = Config.DAILY_QUERY_LIMIT - query_count
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Ask Me Anything",
        "remaining_requests": remaining,
        "daily_limit": Config.DAILY_QUERY_LIMIT
    })


@app.post("/ask")
async def ask_question(
    request: Request,
    background_tasks: BackgroundTasks,
    question: str = Form(...),
    preset: str = Form(""),
):
    user_ip = request.client.host
    if request.headers.get("X-Forwarded-For"):
        user_ip = request.headers.get("X-Forwarded-For").split(",")[0].strip()

    query_count = db_client.get_daily_query_count(user_ip)
    remaining = Config.DAILY_QUERY_LIMIT - query_count
    
    if query_count >= Config.DAILY_QUERY_LIMIT:
        return JSONResponse({
            "error": "Daily quota reached. You've used all questions for today.",
            "remaining_requests": 0,
            "daily_limit": Config.DAILY_QUERY_LIMIT
        })
    
    # Validation...
    if not question or len(question.strip()) < 3:
        return JSONResponse({
            "error": "Please enter a valid question (at least 3 characters).",
            "remaining_requests": remaining,
            "daily_limit": Config.DAILY_QUERY_LIMIT
        })
    
    try:
        result = query_rag(question, from_pill=(preset == "1"))
        
        # Calculate remaining BEFORE logging
        remaining = Config.DAILY_QUERY_LIMIT - (query_count + 1)
        
        # Schedule logging in background (doesn't block response)
        background_tasks.add_task(
            db_client.log_query,
            user_ip=user_ip,
            question=question,
            answer=result["answer"],
            execution_time=result["execution_time"],
            num_chunks=result["num_chunks"],
            sources=result["sources"],
            status="success"
        )
        
        return JSONResponse({
            "answer": result["answer"],
            "sources": result["sources"],
            "num_chunks": result["num_chunks"],
            "execution_time": f"{result['execution_time']:.2f}",
            "remaining_requests": remaining,
            "daily_limit": Config.DAILY_QUERY_LIMIT,
        })
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        
        # Log failed query in background
        background_tasks.add_task(
            db_client.log_query,
            user_ip=user_ip,
            question=question,
            answer="",
            execution_time=0,
            num_chunks=0,
            sources=[],
            status="error"
        )
        
        return JSONResponse({
            "error": "Sorry, something went wrong. Please try again.",
            "remaining_requests": remaining,
            "daily_limit": Config.DAILY_QUERY_LIMIT
        }, status_code=500)
@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy(request: Request):
    """Serve privacy policy page."""
    return templates.TemplateResponse("privacy.html", {
        "request": request,
        "title": "Privacy Policy"
    })


@app.get("/data-request", response_class=HTMLResponse)
async def data_request_form(request: Request):
    """Serve data subject rights request form."""
    return templates.TemplateResponse("data_request.html", {
        "request": request,
        "title": "Data Subject Rights Request"
    })


@app.post("/data-request", response_class=HTMLResponse)
async def submit_data_request(
    request: Request,
    request_type: str = Form(...),
    user_ip: str = Form(...),
    email: str = Form(...),
    details: str = Form(...)
):
    """Handle data subject rights requests."""
    logger.info(f"GDPR request received: {request_type} from IP {user_ip}, contact: {email}")
    
    return templates.TemplateResponse("data_request.html", {
        "request": request,
        "success": True,
        "message": "Your request has been received. We will respond within 30 days as required by GDPR."
    })


@app.get("/robots.txt")
async def robots_txt():
    """Serve robots.txt to prevent crawling."""
    from fastapi.responses import FileResponse
    return FileResponse("static/robots.txt", media_type="text/plain")


@app.get("/health")
async def health_check():
    """Health check endpoint for reverse proxy and monitoring."""
    health_status = {
        "status": "healthy",
        "service": "rag-system",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "disconnected"
    }
    
    # Verify database connection pool
    if db_client and db_client.pool:
        try:
            with db_client.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            health_status["database"] = "connected"
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            health_status["database"] = "error"
            health_status["status"] = "degraded"
    
    return health_status



@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down, closing connections...")
    if db_client:
        db_client.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)