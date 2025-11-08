# Argentum

**Cost intelligence and optimization for AI agent systems**

[![PyPI version](https://img.shields.io/pypi/v/argentum-agent)](https://pypi.org/project/argentum-agent/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/MarsZDF/argentum/workflows/CI/badge.svg)](https://github.com/MarsZDF/argentum/actions)

Argentum provides intelligent cost optimization and monitoring for AI agent systems, helping organizations reduce AI operational costs by 30-80% while maintaining performance. Our platform combines real-time cost tracking, predictive optimization, and comprehensive visibility into agent behavior and spending patterns.

## 🚀 Quick Start

```bash
pip install argentum-agent
```

```python
from argentum import CostOptimizationOrchestrator, CostTracker

# Set up cost optimization with budget management
orchestrator = CostOptimizationOrchestrator(
    total_budget_tokens=1000000,  # 1M token budget
    enable_caching=True,          # Automatic response caching
    enable_model_selection=True,  # Smart model selection
    prefer_cheap_models=True      # Cost optimization priority
)

# Track and optimize all AI operations
result = orchestrator.optimize_request(
    prompt="Analyze this customer feedback",
    context=customer_data,
    agent_id="customer_analyzer"
)

print(f"Cost saved: ${result.cost_saved:.2f}")
print(f"Tokens saved: {result.tokens_saved}")
print(f"Optimizations applied: {result.optimizations_applied}")

# Real-time cost monitoring
tracker = CostTracker()
report = tracker.get_cost_report()
print(f"Total spend: ${report.total_cost:.2f}")
print(f"Cost by agent: {report.breakdown.by_agent}")
```

## 💰 Cost Optimization Engine

### CostOptimizationOrchestrator - Comprehensive Cost Management
Automatically reduce AI costs through intelligent optimization strategies including caching, model selection, context optimization, and batch processing.

```python
from argentum import CostOptimizationOrchestrator

# Configure cost optimization for your environment
orchestrator = CostOptimizationOrchestrator(
    total_budget_tokens=5000000,       # 5M token monthly budget
    per_agent_budget=100000,           # 100K tokens per agent
    enable_caching=True,               # 40-60% cost reduction
    enable_model_selection=True,       # 20-40% cost reduction
    enable_context_optimization=True,  # 10-30% cost reduction
    enable_prompt_optimization=True    # 5-15% cost reduction
)

# All requests automatically optimized
result = orchestrator.optimize_request(
    prompt="Generate marketing copy for new product",
    context=product_specifications,
    agent_id="marketing_agent"
)

# Typical results: 50-70% cost reduction
print(f"Original estimated cost: ${result.original_cost:.2f}")
print(f"Optimized cost: ${result.final_cost:.2f}")
print(f"Total savings: ${result.cost_saved:.2f}")
```

### CostTracker - Real-Time Cost Intelligence
Monitor and analyze AI spending across all agents, operations, and time periods with detailed cost attribution and budget alerts.

```python
from argentum import CostTracker

tracker = CostTracker()

# Real-time cost monitoring
report = tracker.get_cost_report(timeframe="last_24h")

# Detailed cost breakdown
print(f"Total cost: ${report.total_cost:.2f}")
print(f"Cost by agent: {report.breakdown.by_agent}")
print(f"Most expensive operations: {report.top_cost_operations}")
print(f"Budget utilization: {report.budget_utilization:.1%}")

# Set up cost alerts
tracker.set_budget_alert(
    threshold=0.8,  # Alert at 80% budget usage
    notification_method="email"
)
```

### TokenBudgetManager - Budget Control & Governance
Enforce spending limits and prevent budget overruns with sophisticated budget management and allocation strategies.

```python
from argentum import TokenBudgetManager, BudgetAllocator

# Set up budget management
budget = TokenBudgetManager(budget_tokens=1000000)  # 1M token budget
allocator = BudgetAllocator(strategy="priority_based")

# Allocate budget by agent priority
allocations = allocator.allocate_budget({
    "critical_agents": 500000,    # 50% for critical operations
    "standard_agents": 300000,    # 30% for standard operations  
    "experimental_agents": 200000 # 20% for experiments
})

# Budget enforcement
if budget.can_afford(estimated_tokens):
    result = process_request()
    budget.consume(result.tokens_used)
else:
    print("Request would exceed budget - suggesting optimization")
```

## 🔍 Advanced Monitoring & Debugging

### StateDiff - Cost-Aware State Evolution Tracking
Track and analyze how agent state changes over time with integrated cost attribution. Understand both behavior evolution and the cost implications of each decision.

```python
from argentum import StateDiff

# Cost-aware state tracking
diff = StateDiff(track_costs=True)
diff.snapshot("initialization", initial_state, cost_context={"operation": "init"})
diff.snapshot("after_reasoning", updated_state, cost_context={"operation": "reasoning", "tokens_used": 1500})

# See exactly what changed and what it cost
changes = diff.get_changes("initialization", "after_reasoning")
# {
#   'goals': {'removed': ['understand_task']}, 
#   'confidence': {'from': 0.3, 'to': 0.8},
#   'cost_impact': {'tokens_used': 1500, 'estimated_cost': 0.003}
# }
```

### Handoff - Cost-Efficient Multi-Agent Coordination
Standardized protocol for agent-to-agent context transfer with cost attribution and efficiency optimization.

```python
from argentum import HandoffProtocol

protocol = HandoffProtocol(track_costs=True)
handoff = protocol.create_handoff(
    from_agent="data_collector",
    to_agent="data_analyzer", 
    context_summary="Collected 1000 records from API",
    artifacts=["raw_data.json"],
    confidence=0.95,
    cost_context={"tokens_used": 2500, "processing_cost": 0.005}
)

# Analyze handoff efficiency and costs
efficiency_report = protocol.analyze_handoff_efficiency(handoff)
print(f"Context transfer cost: ${efficiency_report.transfer_cost:.3f}")
print(f"Efficiency score: {efficiency_report.efficiency_score:.2f}")

# Serialize for network transfer with cost metadata
json_data = protocol.to_json(handoff, include_cost_data=True)
```

### ContextDecay - Cost-Optimized Memory Management
Manage agent memory with natural forgetting and cost-based importance scoring. Automatically reduce memory costs while preserving critical information.

```python
from argentum import ContextDecay

# Cost-aware context management
decay = ContextDecay(
    half_life_steps=20,
    cost_optimization=True,
    max_context_cost=0.10  # Maximum $0.10 for context storage
)

# Add context with cost-based importance scoring
decay.add("user_name", "Alice", importance=0.9, storage_cost=0.001)
decay.add("session_data", temp_data, importance=0.3, storage_cost=0.05)
decay.add("expensive_analysis", large_analysis, importance=0.7, storage_cost=0.08)

# Time passes... automatic cost-based pruning occurs
for _ in range(10):
    decay.step()

# Get cost-optimized active context
active = decay.get_active(threshold=0.5)
cost_report = decay.get_cost_report()

print(f"Context storage cost: ${cost_report.total_cost:.3f}")
print(f"Items pruned for cost: {cost_report.items_pruned}")
print(f"Cost savings: ${cost_report.cost_saved:.3f}")
```

### PlanLinter - Cost-Impact Analysis & Validation
Static analysis for agent execution plans with cost estimation and optimization recommendations. Prevent expensive mistakes before execution.

```python
from argentum import PlanLinter

linter = PlanLinter(enable_cost_analysis=True)
result = linter.lint(
    plan=agent_generated_plan,
    tool_specs=available_tools,
    secrets=["api_key", "sk-", "password"],
    auto_fix=True,
    cost_budget=0.50  # $0.50 maximum cost for this plan
)

# Cost analysis and optimization
print(f"Estimated plan cost: ${result.cost_analysis.estimated_cost:.3f}")
print(f"Budget compliance: {'✓' if result.cost_analysis.within_budget else '✗'}")

if result.has_cost_optimizations():
    print("Cost optimization opportunities:")
    for opt in result.cost_optimizations:
        print(f"  {opt.description} - Save ${opt.estimated_savings:.3f}")
    
    # Apply cost optimizations
    optimized_plan = result.apply_cost_optimizations(plan)
    print(f"Optimized cost: ${result.optimized_cost:.3f}")

if result.has_errors():
    print("Issues found:")
    for issue in result.issues:
        print(f"  {issue.code}: {issue.message}")
```

## 🛡️ Security Features

Argentum includes comprehensive security controls for production deployments:

- **Input Validation**: Automatic sanitization and size limits
- **Secrets Detection**: Scan for exposed API keys and credentials  
- **Resource Limits**: Prevent memory exhaustion and DoS attacks
- **Safe Serialization**: JSON-only, no unsafe pickle/eval

```python
from argentum import configure_security

# Production security settings
configure_security(
    max_state_size_mb=5,
    max_context_items=5000,
    enable_all_protections=True
)
```

## 📦 Installation Options

```bash
# Core functionality
pip install argentum-agent

# With plan linting features
pip install argentum-agent[lint]

# Development dependencies
pip install argentum-agent[dev]

# Everything included
pip install argentum-agent[all]
```

## 🎯 Use Cases

### Enterprise Cost Management
- **Budget Control**: Set and enforce spending limits across all AI operations
- **Cost Attribution**: Track spending by department, project, and agent
- **ROI Analysis**: Measure cost per outcome and optimize for business value
- **Compliance Reporting**: Generate detailed cost reports for financial auditing

### Production Cost Optimization
- **Automatic Savings**: 30-80% cost reduction through intelligent optimizations
- **Real-time Monitoring**: Instant alerts when costs exceed thresholds
- **Predictive Budgeting**: Forecast future costs based on usage patterns
- **Resource Optimization**: Right-size models and context for each workload

### Multi-Agent Cost Intelligence
- **Agent Coordination Costs**: Track and optimize handoff efficiency
- **Load Balancing**: Distribute work based on cost-effectiveness
- **Performance vs Cost**: Find the optimal balance for each use case
- **Scaling Economics**: Understand cost implications of scaling agent systems

### Development & Testing
- **Cost-Aware Development**: Understand cost implications during development
- **A/B Testing**: Compare not just performance but cost-effectiveness
- **Staging Cost Control**: Prevent expensive mistakes in non-production environments
- **Plan Validation**: Estimate costs before executing expensive agent workflows

## 📚 Examples

Check out the [examples/](examples/) directory for complete working examples:

- **[state_diff_example.py](examples/state_diff_example.py)** - Query processing with state tracking
- **[handoff_example.py](examples/handoff_example.py)** - Content creation pipeline
- **[context_decay_example.py](examples/context_decay_example.py)** - Conversational AI memory
- **[plan_lint_example.py](examples/plan_lint_example.py)** - ML pipeline validation

## 🏗️ Framework Integration

Argentum works with any agent framework:

```python
# LangChain
from langchain.agents import Agent
agent = Agent(...)
session = create_agent_session("langchain_agent")
session['state_diff'].snapshot("start", agent.memory)

# Custom frameworks
class MyAgent:
    def process(self, input_data):
        session['state_diff'].snapshot("before", self.state)
        result = self._internal_process(input_data)
        session['state_diff'].snapshot("after", self.state)
        return result
```

## 🔧 Configuration

### Basic Setup
```python
from argentum import create_agent_session

# Quick setup with security enabled
session = create_agent_session(
    agent_id="my_agent",
    half_life_steps=20,
    secure=True  # Applies security defaults
)
```

### Advanced Configuration
```python
from argentum.security import SecurityConfig, set_security_config

# Custom security policy
config = SecurityConfig(
    max_state_size=5 * 1024 * 1024,  # 5MB limit
    max_context_items=5000,
    enable_injection_protection=True,
    sanitize_log_data=True
)
set_security_config(config)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=argentum

# Run performance benchmarks
pytest tests/test_performance.py --benchmark-only
```

## 📖 Documentation

- **[API Documentation](https://argentum-agent.readthedocs.io)** - Complete API reference
- **[Security Guide](SECURITY.md)** - Security features and best practices
- **[Changelog](CHANGELOG.md)** - Version history and updates

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/MarsZDF/argentum.git
cd argentum
pip install -e .[dev]
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Claude Code](https://claude.ai/code)
- Inspired by the need for better agent debugging tools
- Thanks to the AI agent developer community for feedback and ideas

---

**Argentum** - *Making AI agents observable, debuggable, and reliable.*