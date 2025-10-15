import json
import re
from typing import Any, Dict, Optional, Tuple, List


class PlanStore:
    """In-memory store for a dialogue plan.

    This store expects a dict structure following the spec:
      - main_N nodes with `text`, optional `confirmations`, optional `subplans`
      - explain_* or other nodes with `text` and `next`
      - an optional `end` node
    """

    def __init__(self):
        self._plan: Dict[str, Any] = {}
        self.plan_id: str = "default"

    # ------------ load/save ------------
    def load_from_file(self, path: str, plan_id: str = "default") -> None:
        with open(path, "r", encoding="utf-8") as f:
            self._plan = json.load(f)
        self.plan_id = plan_id

    def set_plan(self, plan: Dict[str, Any], plan_id: str = "default") -> None:
        self._plan = plan
        self.plan_id = plan_id

    # ------------ accessors ------------
    def has_node(self, node_id: str) -> bool:
        return node_id in self._plan

    def get(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._plan.get(node_id)

    def get_text(self, node_id: str) -> Optional[str]:
        node = self.get(node_id)
        if not node:
            return None
        return node.get("text")

    def get_confirmations(self, node_id: str) -> List[Dict[str, Any]]:
        node = self.get(node_id) or {}
        return node.get("confirmations", []) or []

    def get_subplans(self, node_id: str) -> List[Dict[str, Any]]:
        node = self.get(node_id) or {}
        return node.get("subplans", []) or []

    def get_next_from_explain(self, node_id: str) -> Optional[str]:
        node = self.get(node_id) or {}
        return node.get("next")

    def get_start_main_id(self) -> Optional[str]:
        # Prefer main_1 if exists, else the smallest main_<n>
        if "main_1" in self._plan:
            return "main_1"
        mains = [k for k in self._plan.keys() if k.startswith("main_")]
        def main_num(k: str) -> int:
            try:
                return int(k.split("_")[1])
            except Exception:
                return 10**9
        mains.sort(key=main_num)
        return mains[0] if mains else None

    def next_main_id(self, current_main_id: str) -> Optional[str]:
        # If `next` explicitly exists, follow it.
        node = self.get(current_main_id) or {}
        if "next" in node:
            nxt = node.get("next")
            if nxt is None:
                return None
            return nxt
        # Otherwise infer by main_N -> main_(N+1) if present; else check `end` fallback
        m = re.match(r"main_(\d+)", current_main_id)
        if m:
            nxt = f"main_{int(m.group(1)) + 1}"
            if nxt in self._plan:
                return nxt
        # if end exists, return 'end' so caller can terminate.
        return "end" if "end" in self._plan else None

    def is_end(self, node_id: str) -> bool:
        return node_id == "end"

    def validate(self) -> Tuple[bool, List[str]]:
        """Light validation to ensure required fields exist according to the spec."""
        errors: List[str] = []
        # All subplans must have return_to
        for k, node in self._plan.items():
            subplans = node.get("subplans", []) or []
            for i, sp in enumerate(subplans):
                if "return_to" not in sp:
                    errors.append(f"{k}.subplans[{i}] missing return_to")
        # Confirmations must have user_responses mapping
        for k, node in self._plan.items():
            confs = node.get("confirmations", []) or []
            for i, cf in enumerate(confs):
                ur = cf.get("user_responses")
                if not isinstance(ur, dict) or not ur:
                    errors.append(f"{k}.confirmations[{i}] user_responses missing")
        return (len(errors) == 0, errors)
