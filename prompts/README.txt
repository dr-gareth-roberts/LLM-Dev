# LLM Development Environment

A comprehensive environment for testing and comparing different LLM providers and tools.

## Features

- Support for multiple LLM providers (OpenAI, Anthropic, etc.)
- Vector store integration (Chroma, Pinecone, etc.)
- GUI interface for easy testing
- Comprehensive testing suite
- Docker support
- Automated setup and configuration

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-dev-env.git
cd llm-dev-env
```

2. Install the environment:
```bash
make install
```

3. Copy `.env.template` to `.env` and add your API keys:
```bash
cp .env.template .env
# Edit .env with your API keys
```

4. Set up the environment:
```bash
make setup
```

5. Start the GUI:
```bash
make gui
```

## Available Commands

- `make install`: Install the environment
- `make setup`: Initialize the environment
- `make test`: Run tests
- `make docker`: Build and start Docker container
- `make clean`: Clean up temporary files
- `make format`: Format code
- `make lint`: Run linters
- `make jupyter`: Start Jupyter Lab
- `make gui`: Start Streamlit GUI

## Directory Structure

```
llm-dev-env/
├── config/           # Configuration files
├── data/            # Data storage
├── output/          # Output files
├── logs/            # Log files
├── prompts/         # Saved prompts
├── src/             # Source code
├── tests/           # Test files
├── .env             # Environment variables
├── Dockerfile       # Docker configuration
├── docker-compose.yml
├── Makefile
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
```

5. Finally, add a `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
.env
logs/
output/
data/
.cache/

# Jupyter
.ipynb_checkpoints
*.ipynb

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# Docker
.docker/
```

To use everything:

1. Clone/download all files to your project directory
2. Run:
```bash
make install
# Edit .env with your API keys
make setup
make gui  # For the Streamlit interface
```

The Streamlit GUI provides an easy way to:
- Test different LLM providers
- Save and load prompts
- Adjust parameters
- Compare responses
- View and export results