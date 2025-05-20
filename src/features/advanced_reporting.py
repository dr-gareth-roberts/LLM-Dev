"""
advanced_reporting.py - Advanced Report Generation and Real-time Dashboard System
"""
from typing import List, Dict, Any, Optional, Union
import asyncio
import jinja2
import markdown
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import base64
import io
from fpdf import FPDF
import plotly.express as px
from wordcloud import WordCloud
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

class ReportGenerator:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.template_dir = env.env_config.config_dir / 'report_templates'
        self.output_dir = env.env_config.output_dir / 'reports'
        self.template_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir))
        )
        
    async def generate_report(self,
                            data: Dict[str, Any],
                            template_name: str,
                            output_format: str = 'html',
                            custom_styles: Dict[str, Any] = None) -> Path:
        """Generate comprehensive report from data."""
        try:
            # Prepare report context
            context = self._prepare_report_context(data)
            
            # Generate visualizations
            visualizations = await self._generate_visualizations(data)
            context['visualizations'] = visualizations
            
            # Apply template
            template = self.jinja_env.get_template(f"{template_name}.{output_format}.j2")
            content = template.render(**context)
            
            # Generate output
            output_path = self._generate_output(
                content, 
                output_format,
                custom_styles
            )
            
            return output_path
            
        except Exception as e:
            self.env.logger.error(f"Error generating report: {str(e)}")
            raise
    
    async def _generate_visualizations(self, 
                                    data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate various visualizations for the report."""
        visualizations = {}
        
        # Performance metrics visualization
        if 'performance_metrics' in data:
            visualizations['performance'] = self._create_performance_visualization(
                data['performance_metrics']
            )
        
        # Error analysis visualization
        if 'error_analysis' in data:
            visualizations['errors'] = self._create_error_visualization(
                data['error_analysis']
            )
        
        # Response distribution visualization
        if 'responses' in data:
            visualizations['distribution'] = self._create_distribution_visualization(
                data['responses']
            )
        
        # Token usage visualization
        if 'token_usage' in data:
            visualizations['tokens'] = self._create_token_visualization(
                data['token_usage']
            )
        
        return visualizations
    
    def _create_performance_visualization(self, 
                                       metrics: Dict[str, Any]) -> go.Figure:
        """Create performance metrics visualization."""
        fig = go.Figure()
        
        # Add traces for different metrics
        for metric_name, values in metrics.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                name=metric_name,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Performance Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white"
        )
        
        return fig

class RealTimeDashboard:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.app = dash.Dash(__name__, 
                           external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data_cache = {}
        self.update_interval = 5  # seconds
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("LLM Development Dashboard"), width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id='system-metrics-graph'),
                            dcc.Interval(
                                id='system-metrics-update',
                                interval=self.update_interval * 1000
                            )
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Performance"),
                        dbc.CardBody([
                            dcc.Graph(id='model-performance-graph'),
                            dcc.Interval(
                                id='model-performance-update',
                                interval=self.update_interval * 1000
                            )
                        ])
                    ])
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Error Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id='error-analysis-graph'),
                            dcc.Interval(
                                id='error-analysis-update',
                                interval=self.update_interval * 1000
                            )
                        ])
                    ])
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Active Alerts"),
                        dbc.CardBody(id='alerts-container'),
                        dcc.Interval(
                            id='alerts-update',
                            interval=self.update_interval * 1000
                        )
                    ])
                ], width=12)
            ])
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks."""
        @self.app.callback(
            Output('system-metrics-graph', 'figure'),
            Input('system-metrics-update', 'n_intervals')
        )
        def update_system_metrics(n):
            return self._get_system_metrics_figure()
        
        @self.app.callback(
            Output('model-performance-graph', 'figure'),
            Input('model-performance-update', 'n_intervals')
        )
        def update_model_performance(n):
            return self._get_model_performance_figure()
        
        @self.app.callback(
            Output('error-analysis-graph', 'figure'),
            Input('error-analysis-update', 'n_intervals')
        )
        def update_error_analysis(n):
            return self._get_error_analysis_figure()
        
        @self.app.callback(
            Output('alerts-container', 'children'),
            Input('alerts-update', 'n_intervals')
        )
        def update_alerts(n):
            return self._get_alerts_component()
    
    def _get_system_metrics_figure(self) -> go.Figure:
        """Generate system metrics visualization."""
        metrics = self.env.monitoring_system.get_system_metrics()
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=("CPU Usage", "Memory Usage",
                                         "Disk Usage", "Network Usage"))
        
        # Add traces for each metric
        fig.add_trace(
            go.Scatter(x=metrics['timestamp'], 
                      y=metrics['cpu_percent'],
                      name="CPU"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=metrics['timestamp'], 
                      y=metrics['memory_percent'],
                      name="Memory"),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=metrics['timestamp'], 
                      y=metrics['disk_usage'],
                      name="Disk"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=metrics['timestamp'], 
                      y=metrics['network_usage'],
                      name="Network"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        return fig
    
    def _get_alerts_component(self) -> List[dbc.Alert]:
        """Generate alerts component."""
        alerts = self.env.alerting_system.get_active_alerts()
        alert_components = []
        
        for alert in alerts:
            alert_components.append(
                dbc.Alert(
                    f"{alert['rule_name']}: {alert['message']}",
                    color="danger" if alert['severity'] == 'critical' else "warning",
                    dismissable=True
                )
            )
        
        return alert_components
    
    def start_dashboard(self, port: int = 8050):
        """Start the dashboard server."""
        self.app.run_server(debug=True, port=port)

class CustomReportTemplates:
    """Custom report templates for different use cases."""
    
    @staticmethod
    def get_performance_report_template() -> str:
        return """
        # Performance Analysis Report
        
        ## Overview
        {{ overview }}
        
        ## Performance Metrics
        {% for metric, value in metrics.items() %}
        - {{ metric }}: {{ value }}
        {% endfor %}
        
        ## Visualizations
        {{ visualizations.performance }}
        
        ## Error Analysis
        {{ visualizations.errors }}
        
        ## Recommendations
        {% for recommendation in recommendations %}
        - {{ recommendation }}
        {% endfor %}
        """
    
    @staticmethod
    def get_error_analysis_template() -> str:
        return """
        # Error Analysis Report
        
        ## Summary
        {{ summary }}
        
        ## Error Distribution
        {{ visualizations.error_distribution }}
        
        ## Common Error Patterns
        {% for pattern in error_patterns %}
        ### {{ pattern.name }}
        - Frequency: {{ pattern.frequency }}
        - Impact: {{ pattern.impact }}
        - Suggested Fix: {{ pattern.suggestion }}
        {% endfor %}
        
        ## Detailed Analysis
        {{ detailed_analysis }}
        """