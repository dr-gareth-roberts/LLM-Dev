"""
advanced_monitoring.py - Advanced Monitoring and Integration System
"""
import wandb
import mlflow
from mlflow.tracking import MlflowClient
import optuna
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psutil
import gputil
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import threading
import queue
import json
from pathlib import Path
import tempfile
import boto3
from botocore.exceptions import ClientError

class MonitoringSystem:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.metrics_queue = queue.Queue()
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_monitors: Dict[str, threading.Thread] = {}
        self.prometheus_metrics: Dict[str, Any] = {}
        self._initialize_prometheus_metrics()
        
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.prometheus_metrics.update({
            'request_latency': Histogram(
                'llm_request_latency_seconds',
                'LLM request latency in seconds',
                ['model', 'endpoint']
            ),
            'token_usage': Counter(
                'llm_token_usage_total',
                'Total token usage',
                ['model', 'type']
            ),
            'error_rate': Counter(
                'llm_errors_total',
                'Total number of errors',
                'model', 'error_type']
            ),
            'memory_usage': Gauge(
                'llm_memory_usage_bytes',
                'Memory usage in bytes'
            )
        })
        
        # Start Prometheus HTTP server
        start_http_server(8000)

    async def start_monitoring(self, config: Dict[str, Any]):
        """Start monitoring with specified configuration."""
        try:
            # Initialize monitoring threads
            if config.get('system_metrics', True):
                self._start_system_monitoring()
            
            if config.get('model_metrics', True):
                self._start_model_monitoring()
            
            if config.get('cost_tracking', True):
                self._start_cost_monitoring()
            
            # Initialize external integrations
            if config.get('wandb_enabled', False):
                self._initialize_wandb(config['wandb_config'])
            
            if config.get('mlflow_enabled', False):
                self._initialize_mlflow(config['mlflow_config'])
            
            self.env.logger.info("Monitoring system started successfully")
            
        except Exception as e:
            self.env.logger.error(f"Error starting monitoring: {str(e)}")
            raise

    def _start_system_monitoring(self):
        """Start system resource monitoring."""
        def monitor_system():
            while True:
                metrics = {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                }
                
                if gputil.getGPUs():
                    metrics['gpu_utilization'] = gputil.getGPUs()[0].load * 100
                    metrics['gpu_memory'] = gputil.getGPUs()[0].memoryUtil * 100
                
                self.metrics_queue.put(('system', metrics))
                self._check_alerts('system', metrics)
                
                # Update Prometheus metrics
                self.prometheus_metrics['memory_usage'].set(
                    psutil.virtual_memory().used
                )
                
                time.sleep(5)
        
        thread = threading.Thread(target=monitor_system, daemon=True)
        thread.start()
        self.active_monitors['system'] = thread

class ExternalIntegrations:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.wandb_run = None
        self.mlflow_client = None
        self.optuna_study = None
        
    def initialize_wandb(self, config: Dict[str, Any]):
        """Initialize Weights & Biases integration."""
        wandb.init(
            project=config['project_name'],
            entity=config['entity'],
            config=config['parameters']
        )
        self.wandb_run = wandb.run
    
    def initialize_mlflow(self, config: Dict[str, Any]):
        """Initialize MLflow integration."""
        mlflow.set_tracking_uri(config['tracking_uri'])
        mlflow.set_experiment(config['experiment_name'])
        self.mlflow_client = MlflowClient()
    
    def initialize_optuna(self, config: Dict[str, Any]):
        """Initialize Optuna for hyperparameter optimization."""
        self.optuna_study = optuna.create_study(
            study_name=config['study_name'],
            direction=config['direction']
        )
    
    async def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to all active tracking systems."""
        if self.wandb_run:
            wandb.log(metrics, step=step)
        
        if self.mlflow_client:
            with mlflow.start_run(nested=True):
                mlflow.log_metrics(metrics, step=step)

class PerformanceProfiler:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.profiles: Dict[str, Any] = {}
        self.current_profile = None
    
    async def start_profiling(self, profile_name: str):
        """Start performance profiling session."""
        self.current_profile = {
            'name': profile_name,
            'start_time': datetime.now(),
            'events': [],
            'metrics': {},
            'resource_usage': []
        }
        
        # Start resource monitoring
        self._start_resource_monitoring()
    
    async def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log profiling event."""
        if self.current_profile is None:
            raise ValueError("No active profiling session")
        
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'data': data
        }
        
        self.current_profile['events'].append(event)
    
    async def end_profiling(self) -> Dict[str, Any]:
        """End profiling session and return results."""
        if self.current_profile is None:
            raise ValueError("No active profiling session")
        
        self.current_profile['end_time'] = datetime.now()
        self.current_profile['duration'] = (
            self.current_profile['end_time'] - 
            self.current_profile['start_time']
        ).total_seconds()
        
        # Calculate metrics
        self.current_profile['metrics'] = self._calculate_metrics()
        
        # Save profile
        self.profiles[self.current_profile['name']] = self.current_profile
        
        # Generate report
        report = self._generate_profile_report()
        
        return report
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from profile data."""
        events = pd.DataFrame(self.current_profile['events'])
        metrics = {
            'total_requests': len(events),
            'average_latency': events['data'].apply(
                lambda x: x.get('latency', 0)
            ).mean(),
            'error_rate': len(events[events['type'] == 'error']) / len(events),
            'throughput': len(events) / self.current_profile['duration']
        }
        
        return metrics

class AlertingSystem:
    def __init__(self, env: 'LLMDevEnvironment'):
        self.env = env
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.notification_channels: Dict[str, Any] = {}
    
    def add_alert_rule(self, 
                      name: str,
                      condition: str,
                      threshold: float,
                      severity: str = 'warning',
                      channels: List[str] = None):
        """Add new alert rule."""
        self.alert_rules[name] = {
            'condition': condition,
            'threshold': threshold,
            'severity': severity,
            'channels': channels or ['default'],
            'created_at': datetime.now()
        }
    
    async def check_alerts(self, metrics: Dict[str, float]):
        """Check metrics against alert rules."""
        for rule_name, rule in self.alert_rules.items():
            try:
                # Evaluate condition
                if eval(rule['condition'], {'metrics': metrics}):
                    await self._trigger_alert(rule_name, metrics)
            except Exception as e:
                self.env.logger.error(f"Error evaluating alert rule {rule_name}: {str(e)}")
    
    async def _trigger_alert(self, rule_name: str, metrics: Dict[str, float]):
        """Trigger alert and send notifications."""
        alert = {
            'rule_name': rule_name,
            'timestamp': datetime.now(),
            'metrics': metrics,
            'severity': self.alert_rules[rule_name]['severity']
        }
        
        self.alert_history.append(alert)
        
        # Send notifications
        for channel in self.alert_rules[rule_name]['channels']:
            await self._send_notification(channel, alert)
    
    async def _send_notification(self, channel: str, alert: Dict[str, Any]):
        """Send alert notification through specified channel."""
        if channel not in self.notification_channels:
            raise ValueError(f"Unknown notification channel: {channel}")
        
        try:
            await self.notification_channels[channel].send_alert(alert)
        except Exception as e:
            self.env.logger.error(f"Error sending alert through {channel}: {str(e)}")