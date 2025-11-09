"""
Argentum Cost Optimization: Comprehensive toolkit for reducing AI agent costs.

This package provides utilities for token management, caching, model selection,
and other cost optimization strategies for multi-agent systems.

Quick Start:
    >>> from argentum.cost_optimization import CostOptimizationOrchestrator
    >>> 
    >>> orchestrator = CostOptimizationOrchestrator(
    ...     total_budget_tokens=1000000,
    ...     enable_caching=True,
    ...     enable_model_selection=True
    ... )
    >>> 
    >>> # Use orchestrator to optimize all operations
    >>> result = orchestrator.optimize_request(
    ...     prompt="What is AI?",
    ...     context=agent_context,
    ...     agent_id="researcher"
    ... )

Advanced Usage:
    >>> from argentum.cost_optimization import (
    ...     TokenBudgetManager,
    ...     CacheLayer,
    ...     ContextOptimizer,
    ...     CostTracker
    ... )
    >>> 
    >>> budget = TokenBudgetManager(budget_tokens=50000)
    >>> cache = CacheLayer(ttl=3600)
    >>> optimizer = ContextOptimizer(max_tokens=4000)
    >>> tracker = CostTracker()
"""

# Advanced optimizations
from .batch_optimizer import BatchOptimizer, BatchRequest
from .budget_allocator import AllocationStrategy, BudgetAllocator

# Optimization strategies
from .cache import CacheConfig, CacheHit, CacheLayer, CacheMiss
from .context_optimizer import ContextOptimizer, OptimizationStrategy
from .context_pruner import ContextPruner, PruningStrategy
from .cost_tracker import CostBreakdown, CostReport, CostTracker
from .deduplicator import DuplicateDetectionResult, RequestDeduplicator
from .model_selector import ModelConfig, ModelRecommendation, ModelSelector

# Orchestration
from .orchestrator import CostOptimizationOrchestrator, OptimizationConfig
from .prompt_optimizer import PromptOptimizationResult, PromptOptimizer

# Core cost management
from .token_budget import BudgetExceededError, BudgetStatus, TokenBudgetManager
from .token_counter import TokenCounter, TokenizerType, TokenUsage

__all__ = [
    # Core cost management
    "TokenBudgetManager",
    "BudgetExceededError",
    "BudgetStatus",
    "CostTracker",
    "CostReport",
    "CostBreakdown",
    "TokenCounter",
    "TokenUsage",
    "TokenizerType",
    # Optimization strategies
    "CacheLayer",
    "CacheConfig",
    "CacheHit",
    "CacheMiss",
    "ContextOptimizer",
    "OptimizationStrategy",
    "ModelSelector",
    "ModelRecommendation",
    "ModelConfig",
    "PromptOptimizer",
    "PromptOptimizationResult",
    # Advanced optimizations
    "BatchOptimizer",
    "BatchRequest",
    "RequestDeduplicator",
    "DuplicateDetectionResult",
    "ContextPruner",
    "PruningStrategy",
    "BudgetAllocator",
    "AllocationStrategy",
    # Orchestration
    "CostOptimizationOrchestrator",
    "OptimizationConfig",
]

__version__ = "0.1.0"
