#!/bin/bash
# Banking RAG Pipeline - Project Setup Script
# Usage: bash setup_project.sh [project_dir]

PROJECT_DIR="${1:-banking-rag-app}"

echo "=== Banking RAG Pipeline Setup ==="
echo "Project directory: $PROJECT_DIR"

# Create directory structure
mkdir -p "$PROJECT_DIR"/{config,prompts,rag_engine/{innovations,extractors},governance,persistence,feedback,ui,data/{uploaded_docs,faiss_index,index}}

# Create __init__.py files for all packages
for dir in config prompts rag_engine rag_engine/innovations rag_engine/extractors governance persistence feedback ui; do
    touch "$PROJECT_DIR/$dir/__init__.py"
done

echo "✅ Directory structure created"

# Install Python dependencies
echo "Installing Python dependencies..."
sudo pip3 install streamlit openai sentence-transformers pymongo faiss-cpu \
    pdfplumber PyMuPDF pytesseract pdf2image Pillow networkx \
    numpy scikit-learn rank-bm25 2>&1 | tail -3

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y -qq tesseract-ocr poppler-utils 2>&1 | tail -3

echo "✅ Dependencies installed"

# Start MongoDB if not running
if ! pgrep -x mongod > /dev/null; then
    echo "Starting MongoDB..."
    sudo mkdir -p /data/db
    sudo mongod --dbpath /data/db --fork --logpath /tmp/mongod.log 2>&1 | tail -1
    echo "✅ MongoDB started"
else
    echo "✅ MongoDB already running"
fi

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Copy source files into $PROJECT_DIR/"
echo "  2. Set OPENAI_API_KEY environment variable"
echo "  3. Run: cd $PROJECT_DIR && streamlit run app.py --server.port 8501"
