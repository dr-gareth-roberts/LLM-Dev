#!/usr/bin/env python3
"""
install.py - Installation script for LLM Development Environment
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil

def check_prerequisites():
    """Check if required tools are installed."""
    requirements = ['python', 'pip', 'git', 'docker']
    
    for req in requirements:
        if shutil.which(req) is None:
            print(f"Error: {req} is not installed. Please install it first.")
            sys.exit(1)

def setup_virtual_environment():
    """Create and activate virtual environment."""
    subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
    
    # Activate virtual environment
    if sys.platform == 'win32':
        activate_script = 'venv\\Scripts\\activate'
    else:
        activate_script = 'source venv/bin/activate'
    
    print(f"To activate the virtual environment, run: {activate_script}")

def install_requirements():
    """Install required Python packages."""
    # Assuming install.py is run from the project root
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-r', 'config/requirements.txt' # Updated path
    ], check=True)

def setup_project_structure():
    """Create project directory structure."""
    directories = [
        'config',
        'data',
        'output',
        'logs',
        'prompts',
        'src'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def create_env_template():
    """Create .env template file."""
    env_content = """
# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Configuration
DEBUG=false
LOG_LEVEL=INFO

# Vector Store Settings
VECTOR_STORE_PATH=./data/vector_store
"""
    
    # Assuming install.py is run from the project root
    with open('config/.env.template', 'w') as f: # Updated path
        f.write(env_content.strip())

def main():
    """Main installation function."""
    try:
        print("Starting LLM Development Environment installation...")
        
        check_prerequisites()
        setup_project_structure()
        setup_virtual_environment()
        install_requirements()
        create_env_template()
        
        print("\nInstallation completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.template to .env and fill in your API keys")
        print("2. Activate the virtual environment")
        print("3. Run 'python main.py setup' to initialize the environment")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()