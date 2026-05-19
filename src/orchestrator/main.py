#!/usr/bin/env python3
"""
LLM Tensor Server - Orchestrator Service
Main FastAPI application for microservices orchestration
"""

import os
import logging
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def load_environment():
    """Load environment variables from .env file"""
    try:
        from dotenv import load_dotenv
        
        # Get the project root directory (parent of parent of src/)
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"
        
        if env_file.exists():
            load_dotenv(env_file, override=True)
            print(f"✓ Loaded environment variables from {env_file}")
            
            # Log important environment variables (without exposing secrets)
            transformers_url = os.getenv("TRANSFORMERS_SERVICE_URL", "Not set")
            vllm_url = os.getenv("VLLM_SERVICE_URL", "Not set")
            tensorrt_url = os.getenv("TENSORRT_SERVICE_URL", "Not set")
            log_level = os.getenv("LOG_LEVEL", "INFO")
            
            print(f"  TRANSFORMERS_SERVICE_URL: {transformers_url}")
            print(f"  VLLM_SERVICE_URL: {vllm_url}")
            print(f"  TENSORRT_SERVICE_URL: {tensorrt_url}")
            print(f"  LOG_LEVEL: {log_level}")
        else:
            print(f"⚠ No .env file found at {env_file}")
            
    except ImportError:
        print("⚠ python-dotenv not available, using system environment variables only")

def setup_logging():
    """Configure application logging"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Set library log levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    print(f"✓ Logging configured at {log_level} level")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="LLM Tensor Server - Orchestrator",
        description="Microservices orchestration API for LLM inference",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import and include API routes
    from src.orchestrator.routes import router
    app.include_router(router)
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup event"""
        logger = logging.getLogger(__name__)
        logger.info("LLM Tensor Server Orchestrator starting up...")
        
        # Verify microservice connectivity (optional)
        transformers_url = os.getenv("TRANSFORMERS_SERVICE_URL")
        vllm_url = os.getenv("VLLM_SERVICE_URL") 
        tensorrt_url = os.getenv("TENSORRT_SERVICE_URL")
        
        logger.info("Microservice configuration:")
        logger.info(f"  Transformers: {transformers_url}")
        logger.info(f"  vLLM: {vllm_url}")
        logger.info(f"  TensorRT-LLM: {tensorrt_url}")
        
        logger.info("Orchestrator startup completed")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event"""
        logger = logging.getLogger(__name__)
        logger.info("LLM Tensor Server Orchestrator shutting down...")
        
        # Clean up orchestrator service
        try:
            from .handlers.orchestrator_service import get_orchestrator
            orchestrator = get_orchestrator()
            await orchestrator.close()
        except Exception as e:
            logger.error(f"Error during orchestrator shutdown: {e}")
        
        logger.info("Orchestrator shutdown completed")
    
    return app

def main():
    """Main application entry point"""
    print("=" * 50)
    print("LLM Tensor Server - Orchestrator Service")
    print("=" * 50)
    
    # Load environment and setup logging
    load_environment()
    setup_logging()
    
    # Create FastAPI app
    app = create_app()
    
    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8011"))
    
    print(f"🚀 Starting orchestrator service at http://{host}:{port}")
    print("📚 API documentation: http://localhost:8011/docs")
    print("🔄 Microservices orchestration mode")
    print("=" * 50)
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()