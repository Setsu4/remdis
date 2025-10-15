from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfirmationPolicy:
    min_gap_sentences: int = 2
    max_per_session: int = 6


class ConfirmationController:
    def __init__(self, policy: ConfirmationPolicy):
        self.policy = policy
        self._since_last_confirm: int = policy.min_gap_sentences  # allow at start
        self._used: int = 0

    def tick_sentence(self):
        self._since_last_confirm += 1

    def reset_gap(self):
        self._since_last_confirm = 0

    def allow_insert(self) -> bool:
        if self._used >= self.policy.max_per_session:
            return False
        return self._since_last_confirm >= self.policy.min_gap_sentences

    def mark_used(self):
        self._used += 1
        self.reset_gap()

    def remaining(self) -> int:
        return max(0, self.policy.max_per_session - self._used)
