"""
Task Orchestration System for Complex LLM Workflows
"""
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import networkx as nx
from pydantic import BaseModel

from .agent_system import AgentRole, AgentTeam


class TaskDefinition(BaseModel):
    name: str
    description: str
    requirements: List[str]
    agent_roles: List[str]
    expected_output: Dict[str, Any]
    timeout: int = 300

class TaskResult(BaseModel):
    task_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    result: Optional[Any]
    error: Optional[str]

class TaskOrchestrator:
    def __init__(self, env: "LLM-Dev"):
        self.env = env
        self.task_definitions: Dict[str, TaskDefinition] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.task_graph = nx.DiGraph()
        
    def register_task(self, task_def: TaskDefinition):
        """Register a new task definition."""
        self.task_definitions[task_def.name] = task_def
    
    def create_task_pipeline(self, 
                           name: str,
                           tasks: List[str],
                           dependencies: Dict[str, List[str]] = None):
        """Create a pipeline of tasks with dependencies."""
        if dependencies is None:
            dependencies = {}
        
        # Create pipeline graph
        pipeline = nx.DiGraph()
        for task in tasks:
            if task not in self.task_definitions:
                raise ValueError(f"Task {task} not registered")
            pipeline.add_node(task, definition=self.task_definitions[task])
        
        # Add dependencies
        for task, deps in dependencies.items():
            for dep in deps:
                pipeline.add_edge(dep, task)
        
        # Verify no cycles
        if not nx.is_directed_acyclic_graph(pipeline):
            raise ValueError("Pipeline contains cycles")
        
        self.task_graph = pipeline
    
    async def execute_pipeline(self, input_data: Any) -> Dict[str, Any]:
        """Execute the task pipeline."""
        results = {}
        
        # Get execution order
        execution_order = list(nx.topological_sort(self.task_graph))
        
        for task_name in execution_order:
            task_def = self.task_graph.nodes[task_name]['definition']
            
            # Get input from dependencies
            task_input = self._prepare_task_input(task_name, input_data, results)
            
            # Execute task
            try:
                result = await self._execute_task(task_def, task_input)
                results[task_name] = result
            except Exception as e:
                self.env.logger.error(f"Error executing task {task_name}: {str(e)}")
                raise
        
        return results
    
    def _prepare_task_input(self, 
                           task_name: str,
                           initial_input: Any,
                           previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for a task based on dependencies."""
        task_input = {'initial_input': initial_input}
        
        # Add results from dependencies
        predecessors = list(self.task_graph.predecessors(task_name))
        for pred in predecessors:
            if pred in previous_results:
                task_input[pred] = previous_results[pred]
        
        return task_input
    
    async def _execute_task(self, 
                           task_def: TaskDefinition,
                           task_input: Dict[str, Any]) -> Any:
        """Execute a single task."""
        task_id = f"{task_def.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create task result
        self.task_results[task_id] = TaskResult(
            task_id=task_id,
            status="running",
            start_time=datetime.now(),
            result=None
        )
        
        try:
            # Execute task with timeout
            result = await asyncio.wait_for(
                self._run_task(task_def, task_input),
                timeout=task_def.timeout
            )
            
            # Update task result
            self.task_results[task_id].status = "completed"
            self.task_results[task_id].end_time = datetime.now()
            self.task_results[task_id].result = result
            
            return result
            
        except asyncio.TimeoutError:
            self.task_results[task_id].status = "timeout"
            self.task_results[task_id].end_time = datetime.now()
            self.task_results[task_id].error = "Task execution timed out"
            raise
            
        except Exception as e:
            self.task_results[task_id].status = "error"
            self.task_results[task_id].end_time = datetime.now()
            self.task_results[task_id].error = str(e)
            raise
    
    async def _run_task(self, 
                        task_def: TaskDefinition,
                        task_input: Dict[str, Any]) -> Any:
        """Run the actual task implementation."""
        # Create agent team for task
        team = AgentTeam(self.env)
        for role in task_def.agent_roles:
            team.add_agent(AgentRole(role))
        
        # Execute task with agent team
        return await team.execute_task(task_input)
