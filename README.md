# Argentum

**Agent state tracking, debugging, and coordination utilities for AI systems**

[![PyPI version](https://img.shields.io/pypi/v/argentum-agent)](https://pypi.org/project/argentum-agent/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/MarsZDF/argentum/workflows/CI/badge.svg)](https://github.com/MarsZDF/argentum/actions)

Argentum provides essential tools for debugging, monitoring, and coordinating AI agents in production. When your multi-agent system isn't behaving as expected, Argentum helps you understand what's happening and why.

## 🚀 Quick Start

```bash
pip install argentum-agent
```

```python
import argentum

# Create an agent session with all debugging tools
session = argentum.create_agent_session("my_agent")

# Track state changes over time
diff = session["state_diff"]
diff.snapshot("start", {"memory": [], "goals": ["research topic"]})
# ... agent processes ...
diff.snapshot("progress", {"memory": ["fact1", "fact2"], "goals": ["write summary"]})

# See exactly what changed
changes = diff.get_changes("start", "progress") 
# {'memory': {'added': ['fact1', 'fact2']}, 'goals': {'from': ['research topic'], 'to': ['write summary']}}

# Hand off context between agents
protocol = argentum.HandoffProtocol()
handoff = protocol.create_handoff(
    from_agent="researcher",
    to_agent="writer",
    context_summary="Found 3 key sources on quantum computing",
    artifacts=["research_notes.json"],
    confidence=0.85
)

# Manage agent memory with natural decay
context = session["context_decay"]
context.add("user_name", "Alice", importance=0.9)
context.add("temp_data", "processing...", importance=0.3)
context.step()  # time passes
active = context.get_active()  # important stuff survives, temp data fades
```

## 🔍 Core Features

> **🎯 Functionality Guide:**
> - 🟢 **Works Out of the Box** - No external dependencies or API keys needed
> - 🟡 **Needs Configuration** - Requires setup (webhooks, file paths) but no LLM APIs  
> - 🔴 **Needs LLM API** - Would require OpenAI/Anthropic/etc API access (future feature)

### StateDiff - Debug Agent State Evolution 🟢

Track exactly how your agent's state changes over time. Essential for debugging unexpected behavior and understanding agent reasoning.

```python
from argentum import StateDiff

# Track an agent processing a user query
diff = StateDiff()

# Initial state
diff.snapshot("start", {
    "user_query": "What is machine learning?",
    "knowledge": [],
    "confidence": 0.0,
    "processing_steps": 0
})

# After web search
diff.snapshot("searched", {
    "user_query": "What is machine learning?", 
    "knowledge": ["ML is a subset of AI", "Uses algorithms to learn patterns"],
    "confidence": 0.7,
    "processing_steps": 1
})

# See what changed
changes = diff.get_changes("start", "searched")
print(changes["knowledge"])  # {'added': ['ML is a subset of AI', 'Uses algorithms to learn patterns']}
print(changes["confidence"]) # {'from': 0.0, 'to': 0.7}

# Analyze the full evolution
sequence = diff.get_sequence_changes()
for step in sequence:
    print(f"{step['from']} → {step['to']}: {len(step['changes'])} changes")
```

**Use cases:**
- Debug why an agent made unexpected decisions
- Track how confidence scores evolve during reasoning
- Monitor memory growth and information retention
- Understand multi-step reasoning processes

### HandoffProtocol - Agent-to-Agent Coordination 🟢

Standardized context transfer between specialized agents. Prevents information loss and enables sophisticated multi-agent workflows.

```python
from argentum import HandoffProtocol

protocol = HandoffProtocol()

# Research agent hands off to writing agent
handoff = protocol.create_handoff(
    from_agent="research_agent",
    to_agent="writing_agent",
    context_summary="Analyzed 50 customer reviews, identified 3 main themes",
    artifacts=["analysis_results.json", "theme_breakdown.csv"],
    confidence=0.92,
    suggested_next_action="generate_executive_summary",
    metadata={
        "analysis_method": "sentiment_analysis", 
        "sample_size": 50,
        "processing_time": "2.3s"
    }
)

# Serialize for network transfer
json_data = protocol.to_json(handoff)

# Receiving agent recreates the handoff
received = protocol.from_json(json_data)
print(f"Received handoff: {received.context_summary}")
print(f"Next action: {received.suggested_next_action}")

# Generate receipt for confirmation
receipt = protocol.generate_receipt(received, "accepted")
```

**Use cases:**
- Research → Writing agent pipelines
- Data collection → Analysis → Reporting chains  
- Task planning → Execution → Validation workflows
- Microservice-style agent architectures

### ContextDecay - Smart Memory Management 🟢

Natural forgetting for long-running agents. Important information persists while temporary context naturally fades away.

```python
from argentum import ContextDecay

# Create decay manager (information half-life: 20 interactions)
context = ContextDecay(half_life_steps=20)

# Add information with different importance levels
context.add("user_name", "Alice", importance=0.9)        # Very important
context.add("current_task", "write_email", importance=1.0) # Critical
context.add("weather_comment", "sunny today", importance=0.3) # Temporary

# Time passes (simulate 10 interactions)
for _ in range(10):
    context.step()

# Get information that's still relevant
active = context.get_active(threshold=0.5)
for key, value, current_weight in active:
    print(f"{key}: {value} (weight: {current_weight:.2f})")

# Clean up completely faded memories
removed = context.clear_expired(threshold=0.1)
print(f"Cleaned up {removed} expired memories")

# Get memory statistics
stats = context.get_stats()
print(f"Total items: {stats['total_items']}")
print(f"Active items: {stats['active_items']}")
print(f"Average decay: {stats['avg_decay']:.2f}")
```

**Use cases:**
- Conversational AI with natural forgetting
- Long-running task agents that accumulate context
- Personal assistant agents that remember preferences
- Gaming AI that adapts to player behavior over time

### PlanLinter - Validate Agent Plans 🟢

Static analysis for agent execution plans. Catch errors before expensive execution and ensure plans are safe and valid.

```python
from argentum import PlanLinter

linter = PlanLinter()

# Agent-generated execution plan
plan = {
    "steps": [
        {
            "id": "fetch_data",
            "tool": "web_search", 
            "parameters": {"query": "machine learning trends 2024"},
            "outputs": ["search_results"]
        },
        {
            "id": "analyze",
            "tool": "data_analyzer",
            "parameters": {"data": "{{search_results}}", "method": "sentiment"},
            "outputs": ["analysis"]
        }
    ]
}

# Tool specifications
tools = {
    "web_search": {
        "parameters": {
            "query": {"type": "string", "required": True},
            "limit": {"type": "integer", "required": False, "default": 10}
        }
    },
    "data_analyzer": {
        "parameters": {
            "data": {"type": "string", "required": True},
            "method": {"type": "string", "required": False, "default": "basic"}
        }
    }
}

# Validate the plan
result = linter.lint(plan, tools, secrets=["api_key", "password"])

if result.has_errors():
    print("❌ Plan has errors:")
    for issue in result.errors:
        print(f"  {issue.code}: {issue.message}")
        
if result.has_warnings():
    print("⚠️ Plan warnings:")
    for warning in result.warnings:
        print(f"  {warning.code}: {warning.message}")

if result.is_valid():
    print("✅ Plan is valid and ready for execution")
```

**Use cases:**
- Validate LLM-generated execution plans
- Prevent expensive API calls with malformed parameters
- Security scanning for credential exposure
- Catch circular dependencies in complex plans

## 💰 Cost Intelligence Features

### CostTracker - Monitor and Estimate Costs 🟢

Track token usage and estimate costs across agents and operations. Works with estimates when no real API calls are made.

```python
# Available via session (recommended)
session = argentum.create_agent_session("my_agent")
tracker = session.get("cost_tracker")

if tracker:
    # Record token usage (works with estimates)
    tracker.record_usage(
        operation="completion",
        tokens_used=1500,
        agent_id="research_agent",
        model="gpt-4"
    )
    
    # Get cost reports
    report = tracker.get_cost_report()
    print(f"Total estimated cost: ${report.total_cost:.4f}")
    print(f"Cost by agent: {report.breakdown.by_agent}")
```

### CostAlerts - Budget Monitoring 🟡

Set up cost thresholds and receive notifications. Alert configuration works out of the box, but actually sending alerts requires webhook endpoints.

```python
session = argentum.create_agent_session("production_agent")
alerts = session.get("alerts")

if alerts:
    # Configure alerts (works immediately)
    alerts.add_webhook(
        url="https://hooks.slack.com/your-webhook",  # Needs real endpoint
        threshold=0.8,  # 80% of budget
        message="🚨 AI costs approaching limit!"
    )
    
    # Check thresholds (works with estimates)
    triggered = alerts.check_thresholds(
        current_cost=450,
        budget=500,
        agent_id="production_system"
    )
```

## 🛡️ Built-in Security 🟢

Argentum includes production-ready security features:

```python
import argentum

# Configure security for production
argentum.configure_security(
    max_state_size_mb=10,      # Prevent memory exhaustion
    max_context_items=5000,    # Limit context growth
    enable_all_protections=True # Input validation, XSS prevention, etc.
)

# Create secure session
session = argentum.create_agent_session("production_agent", secure=True)
```

**Security features:**
- Input validation and sanitization
- Resource limits and DoS protection  
- Secrets detection in plans and context
- Path traversal prevention
- Safe JSON-only serialization

## 🚧 Future: Advanced Cost Optimization 🔴

> These features are partially implemented but would need LLM API integration to be fully functional:

- **Smart Model Selection** - Automatically choose cheaper models based on task complexity
- **Prompt Optimization** - Reduce token usage while maintaining quality  
- **Context Compression** - Intelligently summarize long contexts
- **Batch Processing** - Group multiple requests for efficiency
- **Semantic Caching** - Cache similar requests with embedding similarity

*Currently available as framework components, but require LLM APIs for real optimization.*

## 📦 Installation Options

```bash
# Core functionality
pip install argentum-agent

# With plan linting (requires jsonschema)
pip install argentum-agent[lint]

# Development dependencies
pip install argentum-agent[dev]
```

## 🎯 When to Use Argentum

**🟢 Ready to use immediately:**
- Multi-agent systems that need coordination
- Production agents that require debugging visibility  
- Long-running agents with memory management needs
- Teams that need to understand agent behavior
- Cost monitoring with token estimates

**🟡 Ready with minimal setup:**
- Production cost alerts (need webhook URLs)
- Custom export formats (need file paths)
- Team notifications (need Slack/Discord webhooks)

**🔴 Future capabilities (would need LLM APIs):**
- Automatic cost optimization during inference
- Smart model selection based on task analysis
- Semantic caching with embedding similarity

**Not intended for:**
- Simple single-shot LLM calls  
- Systems where you don't need debugging or coordination
- Lightweight applications where minimal dependencies are critical

## 🚀 Framework Integration

Argentum works with any agent framework:

```python
# LangChain integration
from langchain.agents import Agent
agent = Agent(...)

session = argentum.create_agent_session("langchain_agent")
session['state_diff'].snapshot("before", agent.memory)
result = agent.run("What is the weather?")
session['state_diff'].snapshot("after", agent.memory)

# Custom framework integration
class MyAgent:
    def __init__(self):
        self.session = argentum.create_agent_session("my_agent")
        self.context = self.session['context_decay']
        self.state_diff = self.session['state_diff']
    
    def process(self, input_data):
        self.state_diff.snapshot("start", self.state)
        self.context.add("last_input", input_data, importance=0.8)
        
        # ... your agent logic ...
        
        self.state_diff.snapshot("end", self.state) 
        return result
```

## 🔧 Real-World Examples

Complete working examples in the [examples/](examples/) directory:

- **[state_diff_example.py](examples/state_diff_example.py)** - Reasoning agent with state tracking
- **[handoff_example.py](examples/handoff_example.py)** - Multi-agent content pipeline  
- **[context_decay_example.py](examples/context_decay_example.py)** - Conversational AI memory
- **[plan_lint_example.py](examples/plan_lint_example.py)** - Plan validation for ML pipelines

## 📊 Performance

Argentum is designed for production use:

- **Low overhead**: Minimal impact on agent performance
- **Memory efficient**: Smart cleanup and configurable limits
- **Thread safe**: Safe for concurrent multi-agent systems
- **Tested**: Comprehensive test suite with performance benchmarks

Run benchmarks: `pytest tests/test_performance.py --benchmark-only`

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage  
pytest --cov=argentum

# Performance benchmarks
pytest tests/test_performance.py --benchmark-only
```

## 📖 Documentation

- **[Security Guide](SECURITY.md)** - Security features and best practices
- **[Agent Guide](AGENT_GUIDE.md)** - LLM integration patterns and validation
- **[Changelog](CHANGELOG.md)** - Version history and updates

## 🤝 Contributing

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md).

```bash
git clone https://github.com/MarsZDF/argentum.git
cd argentum
pip install -e .[dev]
pre-commit install
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Argentum** - *Making AI agents observable, debuggable, and reliable.*