"""
advanced_visualization.py - Advanced Visualization and Reporting System
"""
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_cytoscape as cyto
from datetime import datetime, timedelta

class AdvancedVisualizer:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.viz_dir = env.env_config.output_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        self.current_dashboard = None
    
    def create_interactive_dashboard(self, data: Dict[str, Any]) -> dash.Dash:
        """Create an interactive dashboard for real-time monitoring."""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("LLM Analysis Dashboard"),
            
            # Tabs for different views
            dcc.Tabs([
                dcc.Tab(label='Performance Metrics', children=[
                    self._create_performance_tab(data)
                ]),
                dcc.Tab(label='Error Analysis', children=[
                    self._create_error_analysis_tab(data)
                ]),
                dcc.Tab(label='Response Distribution', children=[
                    self._create_response_distribution_tab(data)
                ]),
                dcc.Tab(label='Token Usage', children=[
                    self._create_token_usage_tab(data)
                ])
            ]),
            
            # Real-time updates
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # in milliseconds
                n_intervals=0
            )
        ])
        
        self._setup_callbacks(app)
        self.current_dashboard = app
        return app
    
    def create_comparison_visualization(self,
                                     results: List[Dict[str, Any]],
                                     metrics: List[str] = None) -> go.Figure:
        """Create comparative visualization of multiple models/runs."""
        if metrics is None:
            metrics = ['accuracy', 'latency', 'cost']
        
        # Create parallel coordinates plot
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=np.random.rand(len(results)),
                         colorscale='Viridis'),
                dimensions=[{
                    'label': metric,
                    'values': [r['metrics'][metric] for r in results]
                } for metric in metrics]
            )
        )
        
        fig.update_layout(
            title="Model Comparison Across Metrics",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_error_heatmap(self, error_data: pd.DataFrame) -> go.Figure:
        """Create interactive error heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=error_data.values,
            x=error_data.columns,
            y=error_data.index,
            colorscale='RdYlBu_r',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Error Distribution Heatmap",
            xaxis_title="Error Types",
            yaxis_title="Test Cases"
        )
        
        return fig
    
    def _setup_callbacks(self, app: dash.Dash):
        """Set up interactive callbacks for the dashboard."""
        @app.callback(
            Output('live-update-graph', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_graph_live(n):
            # Update real-time data
            return self._create_live_performance_graph()
        
        @app.callback(
            Output('error-distribution', 'figure'),
            Input('error-type-dropdown', 'value')
        )
        def update_error_distribution(error_type):
            return self._create_error_distribution(error_type)

class AutomatedTesting:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.test_results: List[Dict[str, Any]] = []
        self.current_test_suite: Optional[Dict[str, Any]] = None
    
    async def run_test_suite(self,
                           test_suite: Dict[str, Any],
                           parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        if parameters is None:
            parameters = {}
        
        self.current_test_suite = test_suite
        results = {
            'timestamp': datetime.now(),
            'test_suite': test_suite['name'],
            'parameters': parameters,
            'results': []
        }
        
        try:
            for test_case in test_suite['test_cases']:
                test_result = await self._run_test_case(test_case, parameters)
                results['results'].append(test_result)
            
            # Analyze results
            results['analysis'] = self._analyze_test_results(results['results'])
            
            self.test_results.append(results)
            return results
            
        except Exception as e:
            self.env.logger.error(f"Test suite error: {str(e)}")
            raise
    
    async def _run_test_case(self,
                            test_case: Dict[str, Any],
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run individual test case with retry logic."""
        max_retries = parameters.get('max_retries', 3)
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                result = await self._execute_test(test_case)
                return result
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    return {
                        'test_case': test_case['name'],
                        'status': 'error',
                        'error': str(e),
                        'retries': retry_count
                    }
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff

class InteractiveDebugger:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.debug_history: List[Dict[str, Any]] = []
        self.breakpoints: Dict[str, Any] = {}
        self.current_session: Optional[Dict[str, Any]] = None
    
    async def start_debug_session(self, 
                                input_data: Any,
                                break_on: List[str] = None) -> Dict[str, Any]:
        """Start interactive debugging session."""
        if break_on is None:
            break_on = ['error', 'unexpected_output', 'threshold_exceeded']
        
        session = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'input_data': input_data,
            'break_on': break_on,
            'history': [],
            'state': {}
        }
        
        self.current_session = session
        return session
    
    async def step_through(self, 
                          session_id: str,
                          action: str = 'next') -> Dict[str, Any]:
        """Step through execution process."""
        if self.current_session is None or \
           self.current_session['session_id'] != session_id:
            raise ValueError("No active debug session")
        
        try:
            if action == 'next':
                result = await self._execute_next_step()
            elif action == 'continue':
                result = await self._continue_execution()
            elif action == 'step_back':
                result = self._step_back()
            else:
                raise ValueError(f"Unknown action: {action}")
            
            self.current_session['history'].append(result)
            return result
            
        except Exception as e:
            self.env.logger.error(f"Debug error: {str(e)}")
            raise
    
    def inspect_state(self, 
                     session_id: str,
                     variable: Optional[str] = None) -> Dict[str, Any]:
        """Inspect current state or specific variable."""
        if self.current_session is None or \
           self.current_session['session_id'] != session_id:
            raise ValueError("No active debug session")
        
        if variable is None:
            return self.current_session['state']
        
        if variable not in self.current_session['state']:
            raise ValueError(f"Variable {variable} not found")
        
        return {
            'variable': variable,
            'value': self.current_session['state'][variable],
            'type': type(self.current_session['state'][variable]).__name__,
            'history': self._get_variable_history(variable)
        }
    
    def add_breakpoint(self, condition: str):
        """Add conditional breakpoint."""
        try:
            # Validate condition
            ast.parse(condition)
            breakpoint_id = hashlib.md5(condition.encode()).hexdigest()
            self.breakpoints[breakpoint_id] = {
                'condition': condition,
                'enabled': True,
                'hit_count': 0
            }
            return breakpoint_id
        except SyntaxError:
            raise ValueError("Invalid breakpoint condition")
    
    def _get_variable_history(self, variable: str) -> List[Dict[str, Any]]:
        """Get history of variable changes."""
        history = []
        for step in self.current_session['history']:
            if variable in step.get('state_changes', {}):
                history.append({
                    'timestamp': step['timestamp'],
                    'value': step['state_changes'][variable],
                    'step_id': step['step_id']
                })
        return history