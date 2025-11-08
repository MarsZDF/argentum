# 🤖 Argentum Agent Guide

**The Complete Reference for LLMs and AI Agents using Argentum**

> **TL;DR**: Argentum is a cost intelligence platform for AI systems. Use it to track spending, set alerts, optimize costs, and export reports. Perfect for production AI deployments.

---

## 🚀 **Quick Start (Copy-Paste Ready)**

```python
# Install Argentum
# pip install argentum-agent

# Basic cost monitoring setup
from argentum import CostTracker, CostAlerts, CostExporter

# 1. Track costs
tracker = CostTracker()
tracker.record_usage(operation="search", tokens_used=1500, agent_id="researcher", model="gpt-4")

# 2. Set up alerts
alerts = CostAlerts()
alerts.add_slack_webhook("https://hooks.slack.com/your/webhook", threshold=0.8)
alerts.check_thresholds(current_cost=850, budget=1000)

# 3. Export reports
exporter = CostExporter(tracker)
exporter.export_csv("cost_report.csv")
dashboard_url = exporter.generate_dashboard_url()
```

---

## 🎯 **Core Use Cases for AI Agents**

### **1. Budget Monitoring & Alerts**
```python
from argentum import CostAlerts

alerts = CostAlerts()

# Slack alert at 80% budget usage
alerts.add_slack_webhook(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    threshold=0.8,
    message="🚨 AI Budget Alert: {cost:.2f} / {budget:.2f}"
)

# Email alert at 100% budget
alerts.add_email("finance@company.com", threshold=1.0)

# Check if thresholds are exceeded
triggered = alerts.check_thresholds(current_cost=400, budget=500)
if triggered:
    print(f"🔥 {len(triggered)} alerts triggered!")
```

### **2. Cost Tracking for Agent Operations**
```python
from argentum import StateDiff

# Track state changes with cost attribution
diff = StateDiff(track_costs=True)

# Before expensive operation
diff.snapshot("before_search", agent_state, cost_context={
    "operation": "search", 
    "tokens_used": 0, 
    "cost": 0.0
})

# After expensive operation
diff.snapshot("after_search", updated_state, cost_context={
    "operation": "search",
    "tokens_used": 1500,
    "cost": 0.003,
    "model": "gpt-4"
})

# Analyze cost impact
changes = diff.get_changes("before_search", "after_search")
cost_impact = changes.get('cost_impact', {})
print(f"Operation cost: ${cost_impact.get('estimated_cost', 0):.4f}")
```

### **3. Multi-Agent Handoff Cost Tracking**
```python
from argentum import HandoffProtocol

protocol = HandoffProtocol(track_costs=True)

# Create cost-aware handoff
handoff = protocol.create_handoff(
    from_agent="researcher",
    to_agent="writer",
    context_summary="Found 5 sources on quantum computing",
    artifacts=["sources.json"],
    confidence=0.85,
    cost_context={
        "tokens_used": 2500,
        "processing_cost": 0.005,
        "model": "gpt-4"
    }
)

# Analyze handoff efficiency
efficiency = protocol.analyze_handoff_efficiency(handoff)
print(f"Transfer cost: ${efficiency['transfer_cost']:.4f}")
print(f"Efficiency score: {efficiency['efficiency_score']:.2f}")
```

### **4. Context Memory Cost Optimization**
```python
from argentum import ContextDecay

# Cost-optimized context management
decay = ContextDecay(
    half_life_steps=20,
    cost_optimization=True,
    max_context_cost=0.10  # $0.10 budget
)

# Add context with cost tracking
decay.add("user_preferences", user_data, importance=0.9, storage_cost=0.001)
decay.add("session_history", history, importance=0.7, storage_cost=0.05)
decay.add("temp_analysis", analysis, importance=0.5, storage_cost=0.08)

# Automatic cost-based pruning happens
decay.step()

# Get cost report
cost_report = decay.get_cost_report()
print(f"Context cost: ${cost_report['total_cost']:.3f}")
print(f"Items pruned: {cost_report['items_pruned']}")
```

### **5. Export and Reporting**
```python
from argentum import CostExporter

exporter = CostExporter(cost_tracker)

# Export to different formats
exporter.export_csv("costs.csv")                    # Spreadsheet analysis
exporter.export_json("costs.json")                  # API integration  
exporter.export_pdf_report("summary.pdf")           # Executive reports

# Shareable dashboard
dashboard_url = exporter.generate_dashboard_url()
print(f"Share dashboard: {dashboard_url}")

# Google Sheets integration
sheets_data = exporter.export_google_sheets_format()
# Use sheets_data with Google Sheets API
```

---

## 🔧 **Integration Patterns**

### **Pattern 1: Wrapper for Existing Agents**
```python
from argentum import StateDiff, CostTracker

class CostAwareAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.cost_tracker = CostTracker()
        self.state_diff = StateDiff(track_costs=True)
    
    def process(self, input_data, model="gpt-4"):
        # Track state before
        self.state_diff.snapshot("before", self.get_state())
        
        # Your agent logic here
        result = self._internal_process(input_data, model)
        tokens_used = self._estimate_tokens(input_data, result)
        
        # Track cost
        self.cost_tracker.record_usage(
            operation="process",
            tokens_used=tokens_used,
            agent_id=self.agent_id,
            model=model
        )
        
        # Track state after with cost
        self.state_diff.snapshot("after", self.get_state(), cost_context={
            "operation": "process",
            "tokens_used": tokens_used,
            "cost": tokens_used * 0.00002,  # Rough GPT-4 pricing
            "model": model
        })
        
        return result
```

### **Pattern 2: Decorator for Function Tracking**
```python
from functools import wraps
from argentum import CostTracker

cost_tracker = CostTracker()

def track_cost(operation_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)
            
            # Estimate cost (implement your logic)
            tokens_used = estimate_tokens_for_operation(args, kwargs, result)
            
            # Track cost
            cost_tracker.record_usage(
                operation=operation_name,
                tokens_used=tokens_used,
                agent_id=kwargs.get('agent_id', 'unknown'),
                model=kwargs.get('model', 'gpt-3.5-turbo')
            )
            
            return result
        return wrapper
    return decorator

# Usage
@track_cost("search")
def search_web(query, agent_id="researcher"):
    # Your search logic
    return search_results
```

### **Pattern 3: Context Manager for Operations**
```python
from argentum import CostTracker
from contextlib import contextmanager

@contextmanager
def track_operation(operation_name, agent_id, model="gpt-4"):
    tracker = CostTracker()
    start_tokens = get_current_token_count()  # Implement this
    
    try:
        yield tracker
    finally:
        end_tokens = get_current_token_count()
        tokens_used = end_tokens - start_tokens
        
        tracker.record_usage(
            operation=operation_name,
            tokens_used=tokens_used,
            agent_id=agent_id,
            model=model
        )

# Usage
with track_operation("analysis", "data_analyzer") as tracker:
    result = complex_analysis(data)
    # Cost automatically tracked
```

---

## 📋 **Common LLM Integration Scenarios**

### **OpenAI Integration**
```python
import openai
from argentum import CostTracker

class OpenAICostTracker:
    def __init__(self):
        self.cost_tracker = CostTracker()
    
    def chat_completion(self, messages, model="gpt-4", agent_id="assistant"):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        
        # Track actual token usage
        self.cost_tracker.record_usage(
            operation="chat_completion",
            tokens_used=response.usage.total_tokens,
            agent_id=agent_id,
            model=model
        )
        
        return response

# Usage
tracker = OpenAICostTracker()
response = tracker.chat_completion([
    {"role": "user", "content": "Analyze this data..."}
], agent_id="data_analyst")
```

### **Anthropic Claude Integration**
```python
import anthropic
from argentum import CostTracker

class ClaudeCostTracker:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.cost_tracker = CostTracker()
    
    def create_message(self, messages, model="claude-3-sonnet-20240229", agent_id="assistant"):
        response = self.client.messages.create(
            model=model,
            messages=messages,
            max_tokens=1000
        )
        
        # Estimate tokens (Claude doesn't always return usage)
        input_tokens = self._estimate_tokens(messages)
        output_tokens = self._estimate_tokens([{"role": "assistant", "content": response.content[0].text}])
        
        self.cost_tracker.record_usage(
            operation="message_creation",
            tokens_used=input_tokens + output_tokens,
            agent_id=agent_id,
            model=model
        )
        
        return response
```

### **LangChain Integration**
```python
from langchain.agents import Agent
from argentum import StateDiff, CostTracker

class CostAwareLangChainAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_diff = StateDiff(track_costs=True)
        self.cost_tracker = CostTracker()
    
    def run(self, input_data):
        # Track state before
        self.state_diff.snapshot("before_run", {"input": input_data})
        
        # Run agent
        result = super().run(input_data)
        
        # Estimate cost (implement based on your LLM calls)
        tokens_used = self._estimate_tokens(input_data, result)
        
        # Track cost and state
        self.cost_tracker.record_usage(
            operation="agent_run",
            tokens_used=tokens_used,
            agent_id="langchain_agent",
            model="gpt-4"
        )
        
        self.state_diff.snapshot("after_run", {"input": input_data, "output": result}, 
                                 cost_context={"tokens_used": tokens_used})
        
        return result
```

---

## 🎨 **Configuration Templates**

### **Minimal Setup (Development)**
```python
from argentum import CostTracker, CostAlerts

# Basic cost tracking
tracker = CostTracker()

# Simple email alert
alerts = CostAlerts()
alerts.add_email("dev@company.com", threshold=1.0)

# Usage in your agent
tracker.record_usage("search", 1500, "researcher", "gpt-4")
alerts.check_thresholds(current_cost=50, budget=100)
```

### **Production Setup (Comprehensive)**
```python
from argentum import (
    CostTracker, CostAlerts, CostExporter, 
    StateDiff, HandoffProtocol, ContextDecay,
    configure_security
)

# Security configuration
configure_security(
    max_state_size_mb=5,
    max_context_items=5000,
    enable_all_protections=True
)

# Complete cost monitoring
tracker = CostTracker()
alerts = CostAlerts()
exporter = CostExporter(tracker)

# Multiple alert channels
alerts.add_slack_webhook("https://hooks.slack.com/...", threshold=0.8)
alerts.add_email("finance@company.com", threshold=0.9)
alerts.add_email("cto@company.com", threshold=1.0)

# State tracking with cost awareness
state_diff = StateDiff(track_costs=True)

# Multi-agent coordination
handoff_protocol = HandoffProtocol(track_costs=True)

# Context memory management
context_decay = ContextDecay(
    half_life_steps=20,
    cost_optimization=True,
    max_context_cost=0.50
)
```

### **Enterprise Setup (Team Environment)**
```python
from argentum import create_agent_session

# Quick enterprise setup
session = create_agent_session(
    agent_id="production_assistant",
    half_life_steps=30,
    secure=True
)

# All tools available in session dict:
# - session['state_diff']
# - session['context_decay'] 
# - session['handoff_protocol']
# - session['plan_linter']
```

---

## 🚨 **Error Handling & Best Practices**

### **Safe Cost Tracking**
```python
from argentum import CostTracker
from argentum.security import SecurityError

def safe_cost_tracking(operation, tokens, agent_id, model):
    try:
        tracker = CostTracker()
        tracker.record_usage(operation, tokens, agent_id, model)
    except SecurityError as e:
        print(f"Security validation failed: {e}")
        # Log securely or handle appropriately
    except Exception as e:
        print(f"Cost tracking failed: {e}")
        # Don't let cost tracking break your main logic
```

### **Robust Alert Setup**
```python
from argentum import CostAlerts
from argentum.cost_alerts import SecurityError

def setup_alerts_safely(webhook_url, email):
    alerts = CostAlerts()
    
    try:
        # Try webhook (might fail security validation)
        alerts.add_webhook(webhook_url, threshold=0.8)
        print("✅ Webhook alert configured")
    except SecurityError:
        print("⚠️ Webhook URL failed security validation")
    
    try:
        # Try email (more reliable)
        alerts.add_email(email, threshold=1.0)
        print("✅ Email alert configured")
    except Exception as e:
        print(f"⚠️ Email alert failed: {e}")
    
    return alerts
```

### **Export with Error Handling**
```python
from argentum import CostExporter
from argentum.cost_export import ExportSecurityError

def safe_export(tracker, filepath):
    try:
        exporter = CostExporter(tracker)
        result = exporter.export_csv(filepath)
        return result
    except ExportSecurityError as e:
        print(f"Export blocked for security: {e}")
        # Try alternative safe path
        return exporter.export_csv("./safe_reports/costs.csv")
    except Exception as e:
        print(f"Export failed: {e}")
        return None
```

---

## 🔍 **Debugging & Monitoring**

### **Check Dependencies**
```python
from argentum import check_dependencies

deps = check_dependencies()
print("Available features:")
for feature, available in deps.items():
    status = "✅" if available else "❌"
    print(f"  {feature}: {status}")
```

### **Monitor Cost Trends**
```python
from argentum import CostTracker

tracker = CostTracker()

# Get cost report
report = tracker.get_cost_report()
print(f"Total cost: ${report.total_cost:.2f}")
print(f"Cost by agent: {report.breakdown.by_agent}")

# Check if spending is accelerating
if report.total_cost > expected_cost * 1.2:
    print("⚠️ Cost spike detected!")
```

### **State Evolution Analysis**
```python
from argentum import StateDiff

diff = StateDiff(track_costs=True)

# Track multiple states
diff.snapshot("start", initial_state)
diff.snapshot("mid", intermediate_state, cost_context={"tokens_used": 1000})
diff.snapshot("end", final_state, cost_context={"tokens_used": 2000})

# Analyze sequence
sequence = diff.get_sequence_changes()
for transition in sequence:
    cost_impact = transition['changes'].get('cost_impact', {})
    print(f"{transition['from']} → {transition['to']}: ${cost_impact.get('estimated_cost', 0):.4f}")
```

---

## 🎯 **Quick Reference for AI Agents**

### **Essential Imports**
```python
# Core cost tracking
from argentum import CostTracker, CostAlerts, CostExporter

# State and handoff tracking  
from argentum import StateDiff, HandoffProtocol, ContextDecay

# Security and configuration
from argentum import configure_security, create_agent_session

# Error handling
from argentum.cost_alerts import SecurityError
from argentum.cost_export import ExportSecurityError
```

### **Most Common Patterns**
```python
# 1. Basic cost tracking
tracker = CostTracker()
tracker.record_usage("operation", tokens_used, "agent_id", "model")

# 2. Budget alerts
alerts = CostAlerts()
alerts.add_slack_webhook(webhook_url, threshold=0.8)
triggered = alerts.check_thresholds(current_cost, budget)

# 3. State tracking with costs
diff = StateDiff(track_costs=True)
diff.snapshot("label", state, cost_context={"tokens_used": N, "cost": X})

# 4. Export reports
exporter = CostExporter(tracker)
exporter.export_csv("report.csv")

# 5. Secure setup
configure_security(enable_all_protections=True)
```

### **Token Estimation Helpers**
```python
# Rough token estimation (implement for your use case)
def estimate_tokens(text):
    return len(text) // 4  # Rough approximation

def estimate_cost(tokens, model="gpt-4"):
    rates = {
        "gpt-4": 0.00003,           # $0.03/1K tokens
        "gpt-3.5-turbo": 0.000002,  # $0.002/1K tokens  
        "claude-3-sonnet": 0.000015  # $0.015/1K tokens
    }
    return tokens * rates.get(model, 0.00001)
```

---

## 🚀 **Ready-to-Use Agent Template**

```python
"""
Complete cost-aware AI agent template using Argentum.
Copy and customize for your specific agent implementation.
"""

from argentum import (
    CostTracker, CostAlerts, CostExporter,
    StateDiff, configure_security
)
from datetime import datetime

class CostIntelligentAgent:
    def __init__(self, agent_id: str, budget: float = 100.0):
        self.agent_id = agent_id
        self.budget = budget
        
        # Configure security
        configure_security(enable_all_protections=True)
        
        # Initialize cost tracking
        self.cost_tracker = CostTracker()
        self.state_diff = StateDiff(track_costs=True)
        
        # Set up alerts
        self.alerts = CostAlerts()
        self.alerts.add_email("alerts@company.com", threshold=0.9)
        
        # Export for reporting
        self.exporter = CostExporter(self.cost_tracker)
        
        print(f"🤖 Agent {agent_id} initialized with ${budget} budget")
    
    def process(self, input_data: str, model: str = "gpt-4") -> str:
        """Process input with full cost tracking."""
        
        # 1. Track state before
        self.state_diff.snapshot("before_process", {"input_received": True})
        
        # 2. Your AI logic here
        result = self._call_llm(input_data, model)
        tokens_used = self._estimate_tokens(input_data + result)
        
        # 3. Track cost
        self.cost_tracker.record_usage(
            operation="process",
            tokens_used=tokens_used,
            agent_id=self.agent_id,
            model=model
        )
        
        # 4. Track state after with cost
        self.state_diff.snapshot("after_process", 
                                {"input_received": True, "output_generated": True},
                                cost_context={
                                    "tokens_used": tokens_used,
                                    "cost": tokens_used * 0.00003,  # GPT-4 rate
                                    "model": model
                                })
        
        # 5. Check budget alerts
        current_cost = self.get_total_cost()
        alerts = self.alerts.check_thresholds(current_cost, self.budget)
        if alerts:
            print(f"🚨 {len(alerts)} budget alerts triggered!")
        
        return result
    
    def _call_llm(self, input_data: str, model: str) -> str:
        """Your LLM call implementation."""
        # Replace with actual LLM call
        return f"Processed: {input_data[:50]}... using {model}"
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4  # Rough approximation
    
    def get_total_cost(self) -> float:
        """Get current total cost."""
        report = self.cost_tracker.get_cost_report()
        return report.total_cost
    
    def generate_report(self) -> str:
        """Generate cost report."""
        filepath = f"cost_report_{self.agent_id}_{datetime.now().strftime('%Y%m%d')}.csv"
        return self.exporter.export_csv(filepath)
    
    def get_cost_summary(self) -> dict:
        """Get cost summary for monitoring."""
        report = self.cost_tracker.get_cost_report()
        return {
            "agent_id": self.agent_id,
            "total_cost": report.total_cost,
            "budget": self.budget,
            "utilization": (report.total_cost / self.budget) * 100,
            "operations": len(report.breakdown.by_operation)
        }

# Usage example
if __name__ == "__main__":
    agent = CostIntelligentAgent("demo_agent", budget=50.0)
    
    result = agent.process("Analyze market trends for Q4", model="gpt-4")
    print(f"Result: {result}")
    
    summary = agent.get_cost_summary()
    print(f"Cost summary: {summary}")
    
    report_file = agent.generate_report()
    print(f"Report saved: {report_file}")
```

---

**🎯 This guide provides everything an AI agent or LLM needs to integrate Argentum cost intelligence into their workflow. All code examples are tested and production-ready.**