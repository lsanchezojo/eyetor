"""Dreams module — nocturnal self-reflection and improvement proposal system."""

from eyetor.config import DreamConfig, DreamsThresholds
from eyetor.dreams.store import DreamProposal, DreamsStore
from eyetor.dreams.analyzer import DreamsAnalyzer
from eyetor.dreams.proposer import ProposalGenerator

__all__ = [
    "DreamConfig",
    "DreamsThresholds",
    "DreamProposal",
    "DreamsStore",
    "DreamsAnalyzer",
    "ProposalGenerator",
]