#!/usr/bin/env python3
"""
main.py - LLM Development Environment Main Script
"""

import asyncio
import click
import sys
import os
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
import yaml
from dotenv import load_dotenv

# Import our modules
from llm_environment import LLMDevEnvironment
from llm_testing import LLMTestingSuite, ToolManager

console = Console()

def create_default_configs():
    """Create default configuration files if they don't exist."""
    config_templates = {
        'openai_config.yaml': {
            'default_model': 'gpt-4-1106-preview',
            'models': {
                'gpt4_turbo': {
                    'name': 'gpt-4-1106-preview',
                    'max_tokens': 4096,
                    'temperature': 0.7
                }
            }
        },
        'anthropic_config.yaml': {
            'default_model': 'claude-2.1',
            'models': {
                'claude2': {
                    'name': 'claude-2.1',
                    'max_tokens': 100000,
                    'temperature': 0.7
                }
            }
        }
        # ... (add other config templates)
    }
    
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    for filename, content in config_templates.items():
        file_path = config_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                yaml.dump(content, f, default_flow_style=False)

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'PINECONE_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        console.print(f"[red]Missing required environment variables: {', '.join(missing_vars)}")
        console.print("Please set them in your .env file or environment.")
        sys.exit(1)

class LLMDevCLI:
    """Command Line Interface for LLM Development Environment."""
    
    def __init__(self):
        self.env = None
        self.testing_suite = None
        self.tool_manager = None
    
    async def initialize(self):
        """Initialize the LLM development environment."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Initializing LLM development environment...", total=None)
            
            try:
                self.env = LLMDevEnvironment()
                await self.env.initialize_clients()
                self.testing_suite = LLMTestingSuite(self.env)
                self.tool_manager = ToolManager(self.env)
                
                progress.update(task, completed=True)
                console.print("[green]Environment initialized successfully!")
                
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[red]Error initializing environment: {str(e)}")
                sys.exit(1)

    def display_status(self):
        """Display current status of the environment."""
        table = Table(title="LLM Development Environment Status")
        
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        for client_name in self.env.clients:
            table.add_row(client_name, "✓ Connected")
        
        console.print(table)

@click.group()
def cli():
    """LLM Development Environment CLI"""
    pass

@cli.command()
@click.option('--force-init', is_flag=True, help="Force reinitialization of configs")
async def setup(force_init):
    """Initialize the LLM development environment."""
    if force_init:
        create_default_configs()
    
    load_dotenv()
    validate_environment()
    
    cli_manager = LLMDevCLI()
    await cli_manager.initialize()
    cli_manager.display_status()

@cli.command()
@click.argument('prompt')
@click.option('--tools', '-t', multiple=True, help="Specific tools to use")
async def compare(prompt: str, tools: Optional[tuple]):
    """Compare responses from different LLM tools."""
    cli_manager = LLMDevCLI()
    await cli_manager.initialize()
    
    tools_list = list(tools) if tools else None
    results = await cli_manager.tool_manager.run_comparison(prompt, tools_list)
    
    for tool, response in results.items():
        console.print(Panel(str(response), title=f"[cyan]{tool} Response"))

@cli.command()
async def test():
    """Run all tests for the LLM development environment."""
    cli_manager = LLMDevCLI()
    await cli_manager.initialize()
    
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Running tests...", total=None)
        await cli_manager.testing_suite.run_all_tests()
        progress.update(task, completed=True)
    
    console.print("[green]Tests completed! Results saved to output directory.")

def create_docker_environment():
    """Create Docker environment files."""
    dockerfile_content = """
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p data output logs config

# Command to run the CLI
ENTRYPOINT ["python", "-m", "main"]
"""
    
    docker_compose_content = """
version: '3.8'

services:
  llm-dev:
    build: .
    volumes:
      - .:/app
      - ~/.cache:/root/.cache
    env_file:
      - .env
    ports:
      - "8888:8888"  # For Jupyter if needed
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content.strip())
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content.strip())

@cli.command()
def create_docker():
    """Create Docker environment files."""
    create_docker_environment()
    console.print("[green]Docker files created successfully!")

def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())