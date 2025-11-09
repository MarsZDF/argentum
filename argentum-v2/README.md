# Argentum

Simple state tracking for AI agents.

## Quick Start

```python
from argentum import StateDiffTracker

tracker = StateDiffTracker()

# Track state changes
tracker.capture("step1", {"thoughts": "I need to search", "plan": ["search", "analyze"]})
tracker.capture("step2", {"thoughts": "I found results", "plan": ["search", "analyze", "respond"]})

# See what changed
tracker.show_diff("step1", "step2")
# Output:
# 📊 Diff: step1 → step2
#   🔄 Changed thoughts: I need to search → I found results
#   📏 plan length: 2 → 3

# Export for analysis
tracker.export_json("debug.json")
```

## Installation

```bash
pip install argentum
```

## License

MIT