# Clinical Data Analysis Chatbot

A sophisticated multi-agent clinical data analysis system that can handle both structured (CSV/Excel) and unstructured (PDF/text) data queries using natural language.

## 🏗️ Architecture

The system uses a multi-agent architecture with:

- **Primary Agent**: Orchestrates the entire pipeline
- **Query Router**: Intelligently routes queries to appropriate tools
- **Structured Data Agent**: Handles SQL-like queries on tabular data
- **RAG Agent**: Processes unstructured documents for domain knowledge
- **Memory Manager**: Maintains conversation context

## 🚀 Features

- **Natural Language Queries**: Ask questions in plain English about clinical data
- **Multi-source Analysis**: Combines structured patient data with medical literature
- **Intelligent Routing**: Automatically determines which data sources to use
- **Clinical Domain Knowledge**: Understands medical terminology and synonyms
- **Conversation Memory**: Maintains context across multiple queries
- **Real-time Monitoring**: LangSmith integration for query tracking

## 📦 Installation

### Prerequisites

1. **Python 3.11+**
2. **UV Package Manager** (recommended) or pip
3. **Huggingface** for local LLM (LLAMA 3.1 8B)
4. **Weaviate Cloud** account for vector database

### Setup Steps

1. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup project**:
```bash
git clone <repository-url>
cd clinical-chatbot
```

3. **Create environment and install dependencies**:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

<!-- 4. **Install Ollama and pull Mistral model**:
```bash
# Install Ollama (see https://ollama.ai)
# Then pull the model:
ollama pull mistral:7b
``` -->
4. **Install Ollama and pull Mistral model**:
```bash
# Get your huggingface token
# Then download the model:
huggingface-cli login   # paste your token
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct
# if you want to download it in specified folder then set --local-dir /models/llama-3.1-8b otherwise it is saved in ~/.cache/huggingface/hub/
```

5. **Setup Weaviate Cloud**:
   - Create account at https://console.weaviate.cloud/
   - Create a cluster
   - Note your cluster URL and API key

6. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configurations
```

### Environment Variables

Create a `.env` file with:

```env
# LLM Configuration
LLM_MODEL=llama-3.1-8b-instruct
TEMPERATURE=0.1
MAX_TOKENS=512

# Weaviate Configuration
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-api-key

# LangSmith (Optional - for monitoring)
LANGSMITH_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=clinical-chatbot

# Data Configuration
COLLECTION_NAME=clinical_documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 📁 Data Setup

### Structured Data

1. Place CSV/Excel files in `data/structured/`:
```bash
mkdir -p data/structured
cp your_clinical_data.csv data/structured/
```

### Unstructured Data

1. Place PDFs and text files in `data/unstructured/`:
```bash
mkdir -p data/unstructured/clinical_docs
cp *.pdf data/unstructured/clinical_docs/
cp *.txt data/unstructured/clinical_docs/
```

## 🖥️ Usage

### Web Interface (Streamlit)

```bash
# uv run streamlit run src/main.py
uv run streamlit run run_app.py
```

Access at: http://localhost:8501

### Command Line Interface

```bash
# uv run python src/main.py cli
uv run python run_cli.py
```

### Example Queries

#### Basic Patient Queries
```
"How many patients died from stroke in the database?"
"When did symptoms start for patient INSP_AU010031?"
"Show me patients with high blood pressure"
```

#### Clinical Analysis
```
"What's the average NIHSS score for TACI stroke patients?"
"Find patients with hypodense stroke and their outcomes"
"Compare mortality rates between different stroke types"
```

#### Domain Knowledge
```
"What are the ways to identify hypodense stroke?"
"Explain the NIHSS scoring system"
"What does TACI stand for in stroke classification?"
```

#### Hybrid Queries
```
"How many patients died from hypodense stroke and what are the identification methods?"
"Show onset times for patients and explain what onset time means clinically"
```

## 🛠️ Development

### Project Structure

```
clinical_chatbot/
├── src/
│   ├── agents/           # Agent implementations
│   ├── tools/           # Data processing tools
│   ├── memory/          # Conversation memory
│   ├── config/          # Configuration
│   └── utils/           # Utilities
├── data/
│   ├── structured/      # CSV/Excel files
│   └── unstructured/    # PDFs/text files
├── tests/               # Test files
└── notebooks/           # Jupyter notebooks
```

### Running Tests

```bash
uv run pytest tests/
```

### Code Quality

```bash
# Format code
uv run black src/
uv run isort src/

# Lint code
uv run flake8 src/
uv run mypy src/
```

## 📊 Monitoring

### LangSmith Integration

1. Sign up at https://smith.langchain.com/
2. Get your API key
3. Add to `.env` file
4. View traces and analytics in LangSmith dashboard

### Local Monitoring

The application logs all queries and responses. Check logs for:
- Query routing decisions
- Tool execution results
- Performance metrics
- Error tracking

## 🔧 Configuration

### Clinical Domain Mappings

Edit `src/config/settings.py` to customize:

- **Clinical Synonyms**: Map terms to database columns
- **Column Mappings**: Categorize database fields
- **Query Patterns**: Define routing logic

### Model Configuration

Switch models by changing `LLM_MODEL` in `.env`:
```env
LLM_MODEL=llama2:8b        # Llama 2 8B
LLM_MODEL=codellama:7b     # Code Llama 7B
LLM_MODEL=mistral:7b       # Mistral 7B (default)
```

## 🐳 Docker Deployment (Optional)

```bash
# Build container
docker-compose build

# Run application
docker-compose up -d

# Access logs
docker-compose logs -f chatbot
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run quality checks: `uv run pytest && uv run black src/`
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Check if Ollama is running
ollama list
ollama serve
```

**2. Weaviate Connection Error**
- Verify cluster URL and API key
- Check network connectivity
- Ensure cluster is running

**3. Memory Issues**
```bash
# Reduce chunk size in settings
CHUNK_SIZE=500
MAX_TOKENS=1024
```

**4. Data Loading Errors**
- Check file permissions in `data/` directory
- Verify CSV/Excel file format
- Ensure PDF files are readable

### Performance Optimization

1. **Reduce model size**: Use `mistral:7b` instead of larger models
2. **Limit document chunks**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP`
3. **Use GPU**: Configure Ollama for GPU acceleration
4. **Cache frequently used queries**: Enable query result caching

### Getting Help

1. Check the logs for detailed error messages
2. Use CLI mode for debugging: `uv run python src/main.py cli`
3. Test individual components in Jupyter notebooks
4. Submit issues with logs and example queries

## 🎯 Roadmap

- [ ] Support for more LLM providers (OpenAI, Anthropic)
- [ ] Advanced visualization of query results
- [ ] Real-time data streaming capabilities
- [ ] Multi-language support
- [ ] Enhanced security and audit features
- [ ] Integration with clinical databases (FHIR)