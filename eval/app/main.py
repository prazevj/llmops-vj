from fastapi import FastAPI
from config.database import init_database
from routers import evaluations, metrics, model_config, health
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="APA Evaluation Management API", 
    version="1.0.0",
    description="API for managing evaluations and metrics"
)

# Include routers
app.include_router(evaluations.router, prefix="/evaluations", tags=["evaluations"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
app.include_router(model_config.router, prefix="/api", tags=["model-config"])
app.include_router(health.router, tags=["health"])

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()
    logger.info("Application started successfully")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "APA Evaluation Management API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
