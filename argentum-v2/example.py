#!/usr/bin/env python3

from argentum import StateDiffTracker

# Simulate LangChain agent execution
tracker = StateDiffTracker()

# Initial state
tracker.capture("start", {
    "query": "What's the weather in NYC?",
    "intermediate_steps": [],
    "thoughts": "I need to search for weather information"
})

# After first tool call
tracker.capture("search_done", {
    "query": "What's the weather in NYC?", 
    "intermediate_steps": [
        {"tool": "search", "input": "NYC weather", "output": "Sunny, 72°F"}
    ],
    "thoughts": "Got weather data, now I should format response"
})

# Final state
tracker.capture("complete", {
    "query": "What's the weather in NYC?",
    "intermediate_steps": [
        {"tool": "search", "input": "NYC weather", "output": "Sunny, 72°F"}
    ],
    "thoughts": "Response ready",
    "final_answer": "It's sunny and 72°F in New York City"
})

# Show evolution
print("=== Agent State Evolution ===\n")
tracker.show_diff("start", "search_done")
print()
tracker.show_diff("search_done", "complete")

# List all snapshots
print(f"\nSnapshots: {tracker.list_snapshots()}")