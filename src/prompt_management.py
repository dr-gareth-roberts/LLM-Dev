"""
Prompt Management System
"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import jinja2
from datetime import datetime

class PromptTemplate:
    def __init__(self, template: str, variables: Dict[str, str]):
        self.template = template
        self.variables = variables
        self._jinja_template = jinja2.Template(template)
    
    def render(self, values: Dict[str, str]) -> str:
        return self._jinja_template.render(**values)

class PromptManager:
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = prompts_dir
        self.templates_dir = prompts_dir / 'templates'
        self.history_dir = prompts_dir / 'history'
        self.templates_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
        
    def save_template(self, name: str, template: str, variables: Dict[str, str]):
        """Save a prompt template."""
        template_data = {
            'template': template,
            'variables': variables,
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.templates_dir / f"{name}.yaml", 'w') as f:
            yaml.dump(template_data, f)
    
    def load_template(self, name: str) -> PromptTemplate:
        """Load a prompt template."""
        with open(self.templates_dir / f"{name}.yaml", 'r') as f:
            data = yaml.safe_load(f)
        
        return PromptTemplate(data['template'], data['variables'])
    
    def save_prompt_history(self, prompt: str, response: str, metadata: Dict[str, Any]):
        """Save prompt and response history."""
        history_data = {
            'prompt': prompt,
            'response': response,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(self.history_dir / filename, 'w') as f:
            yaml.dump(history_data, f)
    
    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """Search prompt history."""
        results = []
        for file in self.history_dir.glob('*.yaml'):
            with open(file, 'r') as f:
                data = yaml.safe_load(f)
                if query.lower() in data['prompt'].lower():
                    results.append(data)
        return results