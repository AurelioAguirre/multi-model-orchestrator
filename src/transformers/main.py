#!/usr/bin/env python3
"""
Transformers Microservice
FastAPI service for HuggingFace Transformers inference
"""

import os
import sys
import logging
from pathlib import Path

# Module imports - using src.shared from container PYTHONPATH=/app

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.shared.config import load_config
from src.shared.transformers_provider import TransformersProvider
from src.transformers.routes import router, set_provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TransformersService")

# FastAPI app
app = FastAPI(
    title="Transformers Microservice",
    description="HuggingFace Transformers inference service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

# Global state
config = None
provider = None


@app.on_event("startup")
async def startup_event():
    """Initialize the service"""
    global config, provider

    try:
        # Load configuration
        config_path = Path(__file__).parent.parent.parent / "resources" / "config.yml"
        config = load_config(str(config_path))

        # Initialize provider (but don't load model yet)
        provider = TransformersProvider(config)

        # Set provider in routes module
        set_provider(provider)

        logger.info("Transformers service initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise


if __name__ == "__main__":
    # Get port from environment or default
    port = int(os.getenv("TRANSFORMERS_SERVICE_PORT", 8012))
    
    logger.info(f"Starting Transformers service on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )