#!/bin/bash

# Clinical Chatbot Setup Script
# This script automates the setup process for the clinical data analysis chatbot

set -e  # Exit on any error

echo "🏥 Clinical Data Analysis Chatbot Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

# Check if running on macOS, Linux, or Windows (Git Bash)
OS="unknown"
case "$(uname -s)" in
    Darwin*)    OS="macOS";;
    Linux*)     OS="Linux";;
    CYGWIN*|MINGW*|MSYS*)    OS="Windows";;
esac

print_status "Detected OS: $OS"

# Step 1: Check prerequisites
print_step "Checking prerequisites..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python version: $PYTHON_VERSION"
    
    # Check if Python version is 3.11 or higher
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
        print_status "Python version is compatible (3.11+)"
    else
        print_error "Python 3.11+ required. Please upgrade Python."
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Step 2: Install UV package manager
print_step "Installing UV package manager..."

if command -v uv &> /dev/null; then
    print_status "UV already installed: $(uv --version)"
else
    print_status "Installing UV package manager..."
    if [ "$OS" = "Windows" ]; then
        # For Windows (Git Bash/MSYS2)
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        # For macOS and Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        print_status "UV installed successfully: $(uv --version)"
    else
        print_error "UV installation failed. Please install manually from https://github.com/astral-sh/uv"
        exit 1
    fi
fi

# Step 3: Set up Python environment
print_step "Setting up Python virtual environment..."

if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists"
else
    print_status "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
if [ "$OS" = "Windows" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Step 4: Install dependencies
print_step "Installing Python dependencies..."

print_status "Installing project dependencies with UV..."
uv pip install -e .

# Install development dependencies if requested
read -p "Install development dependencies (tests, linting, etc.)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installing development dependencies..."
    uv pip install -e ".[dev]"
fi

# Step 5: Install Ollama
print_step "Checking Ollama installation..."

if command -v ollama &> /dev/null; then
    print_status "Ollama already installed: $(ollama --version)"
else
    print_status "Ollama not found. Please install Ollama manually."
    echo
    echo "Installation instructions:"
    echo "- macOS/Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "- Windows: Download from https://ollama.ai/download"
    echo
    read -p "Press Enter after installing Ollama..."
fi

# Check if Ollama is running
if ! ollama list &> /dev/null; then
    print_warning "Ollama server is not running. Starting it..."
    if [ "$OS" = "macOS" ] || [ "$OS" = "Linux" ]; then
        ollama serve &
        sleep 5
    else
        print_warning "Please start Ollama server manually on Windows"
        read -p "Press Enter after starting Ollama server..."
    fi
fi

# Step 6: Download required models
print_step "Downloading required LLM models..."

print_status "Pulling Mistral 7B model (this may take a while)..."
if ollama pull mistral:7b; then
    print_status "Mistral 7B model downloaded successfully"
else
    print_error "Failed to download Mistral model. Please check your internet connection."
fi

# Optionally download alternative models
read -p "Download additional models (llama2:8b)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Downloading Llama 2 8B model..."
    ollama pull llama2:8b
fi

# Step 7: Set up directories
print_step "Setting up project directories..."

print_status "Creating data directories..."
mkdir -p data/structured
mkdir -p data/unstructured/clinical_docs
mkdir -p logs
mkdir -p notebooks

print_status "Directory structure created"

# Step 8: Configuration setup
print_step "Setting up configuration..."

if [ -f ".env" ]; then
    print_warning ".env file already exists"
    read -p "Overwrite existing .env file? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp .env.example .env
        print_status "Copied .env.example to .env"
    fi
else
    cp .env.example .env
    print_status "Copied .env.example to .env"
fi

# Step 9: Weaviate setup guidance
print_step "Weaviate Vector Database Setup"

echo
echo "To complete the setup, you need to configure Weaviate:"
echo "1. Go to https://console.weaviate.cloud/"
echo "2. Create a free account"
echo "3. Create a new cluster"
echo "4. Copy your cluster URL and API key"
echo "5. Update the .env file with your credentials:"
echo "   - WEAVIATE_URL=https://your-cluster.weaviate.network"
echo "   - WEAVIATE_API_KEY=your-api-key"
echo

read -p "Have you set up Weaviate and updated .env? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Weaviate configuration completed"
else
    print_warning "Please complete Weaviate setup manually"
fi

# Step 10: Optional LangSmith setup
print_step "Optional: LangSmith Setup for Monitoring"

read -p "Set up LangSmith for query monitoring? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "To set up LangSmith:"
    echo "1. Go to https://smith.langchain.com/"
    echo "2. Create a free account"
    echo "3. Get your API key"
    echo "4. Update .env file:"
    echo "   - LANGSMITH_API_KEY=your-langsmith-key"
    echo
    read -p "Press Enter after updating .env with LangSmith key..."
fi

# Step 11: Test installation
print_step "Testing installation..."

print_status "Running basic tests..."

# Test Python imports
python3 -c "
try:
    import pandas as pd
    import langchain
    import weaviate
    import streamlit
    print('✅ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test Ollama connection
if ollama list > /dev/null 2>&1; then
    print_status "✅ Ollama connection successful"
else
    print_error "❌ Ollama connection failed"
fi

# Step 12: Sample data setup
print_step "Sample data setup"

echo "Place your clinical data files in:"
echo "- Structured data (CSV/Excel): data/structured/"
echo "- Unstructured data (PDFs/texts): data/unstructured/clinical_docs/"
echo

if [ -f "INSPIRE_rows_15_20.csv" ]; then
    cp INSPIRE_rows_15_20.csv data/structured/
    print_status "Sample CSV file copied to data/structured/"
fi

# Step 13: Final instructions
print_step "Setup Complete! 🎉"

echo
echo "Your clinical chatbot is ready to use!"
echo
echo "Quick Start:"
echo "1. Activate the virtual environment:"
if [ "$OS" = "Windows" ]; then
    echo "   source .venv/Scripts/activate"
else
    echo "   source .venv/bin/activate"
fi
echo
echo "2. Start the web interface:"
echo "   uv run streamlit run src/main.py"
echo
echo "3. Or use the CLI:"
echo "   uv run python src/main.py cli"
echo
echo "4. Try the Jupyter notebook:"
echo "   uv run jupyter lab notebooks/data_exploration.ipynb"
echo
echo "Configuration files:"
echo "- .env (edit this with your API keys)"
echo "- src/config/settings.py (advanced configuration)"
echo
echo "Documentation:"
echo "- README.md (comprehensive guide)"
echo "- notebooks/data_exploration.ipynb (examples)"
echo

print_status "Setup completed successfully!"
print_warning "Remember to:"
print_warning "1. Update .env with your Weaviate credentials"
print_warning "2. Add your clinical data files to data/ directories"
print_warning "3. Start Ollama server if it's not running"

echo
echo "For help and troubleshooting, see the README.md file."
echo "Happy analyzing! 🏥📊"