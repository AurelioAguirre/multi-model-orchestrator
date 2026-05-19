#!/bin/bash

# LLM Tensor Server - Orchestrator Installation Script
# For microservices architecture - installs only orchestrator dependencies

set -e  # Exit on any error

echo "LLM Tensor Server - Orchestrator Installation"
echo "============================================="
echo ""
echo "This script installs dependencies for the orchestrator service only."
echo "Individual ML frameworks run in separate microservice containers."
echo ""
echo "For full microservices deployment, use:"
echo "  ./run_containers.sh"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    echo "Please install Python $REQUIRED_VERSION and create a virtual environment."
    exit 1
fi

echo "✅ Python version check passed: $PYTHON_VERSION"
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: You don't appear to be in a virtual environment."
    echo "It's recommended to use a virtual environment for Python packages."
    echo ""
    read -p "Continue anyway? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled. Please create and activate a virtual environment:"
        echo "  python -m venv venv"
        echo "  source venv/bin/activate  # Linux/Mac"
        echo "  venv\\Scripts\\activate     # Windows"
        exit 1
    fi
else
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
fi

echo ""
echo "📦 Installing orchestrator dependencies..."
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install base requirements (orchestrator needs minimal deps)
echo ""
echo "📥 Installing orchestrator requirements..."
pip install -r requirements/orchestrator.txt

echo ""
echo "🎉 Orchestrator installation complete!"
echo ""
echo "📋 What was installed:"
echo "  - FastAPI and Uvicorn (web server)"
echo "  - HTTP clients (requests, httpx)"  
echo "  - Configuration management (PyYAML)"
echo "  - Basic utilities"
echo ""
echo "🚀 Start the orchestrator service:"
echo "  python src/orchestrator/main.py"
echo ""
echo "🐳 For full microservices deployment:"
echo "  ./run_containers.sh"
echo ""
echo "📚 API documentation will be available at:"
echo "  http://localhost:8011/docs"