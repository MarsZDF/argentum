import json
from typing import Any, Dict, List, Optional
from copy import deepcopy


class StateDiffTracker:
    """Simple state tracking for AI agents."""
    
    def __init__(self):
        self._snapshots: Dict[str, Any] = {}
    
    def capture(self, name: str, state: Any) -> None:
        """Capture a state snapshot."""
        self._snapshots[name] = deepcopy(state)
    
    def show_diff(self, from_name: str, to_name: str) -> None:
        """Print diff between two snapshots."""
        if from_name not in self._snapshots:
            print(f"❌ Snapshot '{from_name}' not found")
            return
        if to_name not in self._snapshots:
            print(f"❌ Snapshot '{to_name}' not found")
            return
            
        from_state = self._snapshots[from_name]
        to_state = self._snapshots[to_name]
        
        print(f"📊 Diff: {from_name} → {to_name}")
        self._print_diff(from_state, to_state)
    
    def export_json(self, filepath: str) -> None:
        """Export all snapshots to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self._snapshots, f, indent=2, default=str)
    
    def list_snapshots(self) -> List[str]:
        """List all snapshot names."""
        return list(self._snapshots.keys())
    
    def _print_diff(self, old: Any, new: Any, path: str = "") -> None:
        """Recursively print differences."""
        if old == new:
            return
            
        if isinstance(old, dict) and isinstance(new, dict):
            all_keys = set(old.keys()) | set(new.keys())
            for key in sorted(all_keys):
                key_path = f"{path}.{key}" if path else key
                if key not in old:
                    print(f"  ✅ Added {key_path}: {new[key]}")
                elif key not in new:
                    print(f"  ❌ Removed {key_path}: {old[key]}")
                else:
                    self._print_diff(old[key], new[key], key_path)
        elif isinstance(old, list) and isinstance(new, list):
            if len(old) != len(new):
                print(f"  📏 {path} length: {len(old)} → {len(new)}")
            for i, (o, n) in enumerate(zip(old, new)):
                self._print_diff(o, n, f"{path}[{i}]")
        else:
            print(f"  🔄 Changed {path}: {old} → {new}")