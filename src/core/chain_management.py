"""
Chain Management System for Complex LLM Workflows
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
from asyncio import sleep

from src.evaluation_framework.advanced_evaluation import LLMDevEnvironment


@dataclass
class ChainStep:
    name: str
    function: Callable
    config: Dict[str, Any]
    retry_config: Dict[str, Any] = None

class ChainManager:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.chains: Dict[str, List[ChainStep]] = {}

    def create_chain(self, name: str, steps: List[ChainStep]):
        """Create a new processing chain."""
        self.chains[name] = steps

    async def execute_chain(self, chain_name: str, input_data: Any) -> Dict[str, Any]:
        """Execute a processing chain."""
        if chain_name not in self.chains:
            raise ValueError(f"Chain {chain_name} not found")

        results = {'input': input_data}
        current_data = input_data

        for step in self.chains[chain_name]:
            try:
                current_data = await self._execute_step(step, current_data)
                results[step.name] = current_data
            except Exception as e:
                self.env.logger.error(f"Error in chain {chain_name}, step {step.name}: {str(e)}")
                results['error'] = str(e)
                break

        return results

    async def _execute_step(self, step: ChainStep, input_data: Any) -> Any:
        """Execute a single step with retry logic."""
        if step.retry_config:
            return await self._retry_execution(step, input_data)
        return await step.function(input_data, **step.config)

    async def _retry_execution(self, step: ChainStep, input_data: Any) -> Any:
        """Retry execution logic for a step."""
        retries = step.retry_config.get('retries', 3)
        delay = step.retry_config.get('delay', 1)

        for attempt in range(retries):
            try:
                return await step.function(input_data, **step.config)
            except Exception as e:
                if attempt < retries - 1:
                    self.env.logger.warning(f"Retry {attempt + 1} for step {step.name} due to error: {str(e)}")
                    await sleep(delay)
                else:
                    raise e
