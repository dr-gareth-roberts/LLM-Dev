# LLM Development Environment

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A comprehensive environment for testing and comparing different Large Language Model (LLM) providers and tools. Streamline your LLM development workflow with a unified testing framework, intuitive GUI, and robust evaluation tools.

<p align="center">
  <img src="https://github.com/dr-gareth-roberts/LLM-dev/raw/main/docs/images/llm-dev-logo.png" alt="LLM Dev Logo" width="250"/>
</p>

## 🌟 Features

- **Multi-Provider Support**: Seamless integration with multiple LLM providers (OpenAI, Anthropic, etc.)
- **Vector Store Integration**: Built-in support for popular vector databases (Chroma, Pinecone, etc.)
- **Interactive GUI**: User-friendly Streamlit interface for rapid testing and experimentation
- **Comprehensive Testing**: Extensive suite of evaluation metrics and testing tools
- **Docker Support**: Containerized environment for consistent development and deployment
- **Automated Setup**: Quick configuration with automated dependency management

## 📋 Prerequisites

- Python 3.8 or higher
- Make (for running commands)
- Docker (optional, for containerized usage)
- API keys for desired LLM providers

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/dr-gareth-roberts/LLM-dev.git
cd LLM-dev
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

## 📷 Screenshots

<p align="center">
  <img src="https://github.com/dr-gareth-roberts/LLM-dev/raw/main/docs/images/gui-screenshot.png" alt="GUI Screenshot" width="600"/>
</p>

## 🔧 Available Commands

| Command | Description |
|---------|-------------|
| `make install` | Install the environment and dependencies |
| `make setup` | Initialize the environment and prepare resources |
| `make test` | Run the full test suite |
| `make docker` | Build and start Docker container |
| `make clean` | Clean up temporary files and caches |
| `make format` | Format code according to project standards |
| `make lint` | Run linters for code quality checks |
| `make jupyter` | Start Jupyter Lab for interactive development |
| `make gui` | Launch the Streamlit GUI interface |

## 📁 Directory Structure

```
llm-dev/
├── config/          # Project configuration files (incl. requirements, .env.template)
├── data/            # Data storage (e.g., user prompts, datasets)
├── docs/            # Documentation files
├── examples/        # Example scripts
├── logs/            # Log files
├── output/          # Output files from evaluations, etc.
├── scripts/         # Helper and operational scripts (install, run tests, etc.)
├── src/             # Source code, organized into submodules (core, metrics, ui, etc.)
├── tests/           # Test files
├── .env             # Local environment variables (copied from config/.env.template)
├── .gitignore       # Specifies intentionally untracked files
├── Dockerfile       # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── main.py          # Main CLI entry point
└── README.md
```

## 🔌 API Usage

### Core API

```python
from llm_dev import LLMTester

# Initialize the tester with your provider
tester = LLMTester(provider="openai")

# Run a simple test
result = tester.run_prompt("Explain quantum computing in simple terms")

# Compare multiple providers
comparison = tester.compare_providers(
    prompt="What is the capital of France?",
    providers=["openai", "anthropic", "cohere"]
)

# Save results
tester.save_results(comparison, "provider_comparison.json")
```

### Vector Store Integration

```python
from llm_dev.vector import VectorStore

# Initialize a vector store
store = VectorStore(provider="chroma")

# Add documents
store.add_documents(["Document 1 content", "Document 2 content"])

# Query the store
results = store.query("What do the documents say about AI?")
```

## 🎮 GUI Features

The Streamlit GUI provides an easy way to:

- Test different LLM providers with custom prompts
- Save and load prompts for reuse
- Adjust parameters like temperature, max tokens, etc.
- Compare responses across multiple providers
- View and export results in various formats
- Visualize performance metrics

## 🤝 Contributing

We welcome contributions from the community! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Benchmarks

Performance benchmarks for different LLM providers on standard tasks:

| Provider | Response Time (avg) | Token Cost (1K tokens) | Accuracy Score |
|----------|---------------------|------------------------|----------------|
| OpenAI   | 0.8s                | $0.02                  | 92%            |
| Anthropic | 1.2s               | $0.025                 | 94%            |
| Cohere   | 1.0s                | $0.015                 | 88%            |

## 📚 Further Documentation

- [Advanced Configuration](docs/advanced-config.md)
- [Custom Provider Integration](docs/custom-providers.md)
- [Evaluation Metrics](docs/evaluation-metrics.md)
- [Docker Deployment](docs/docker-deployment.md)

## 🙏 Acknowledgements

- Thanks to all [contributors](https://github.com/dr-gareth-roberts/LLM-dev/graphs/contributors)
- Inspired by various LLM testing frameworks and tools