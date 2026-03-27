"""Workflow pattern implementations based on Anthropic's agent patterns."""

from eyetor.workflows.chain import PromptChain
from eyetor.workflows.router import Router
from eyetor.workflows.parallel import Parallel
from eyetor.workflows.orchestrator import OrchestratorWorkflow
from eyetor.workflows.evaluator import EvaluatorOptimizer

__all__ = [
    "PromptChain",
    "Router",
    "Parallel",
    "OrchestratorWorkflow",
    "EvaluatorOptimizer",
]
