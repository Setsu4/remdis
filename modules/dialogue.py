import sys
import threading
import queue
import time
import re

from base import RemdisModule
from news_orchestrator import Orchestrator


class Dialogue(RemdisModule):
    """Thin wrapper to keep existing entry point.
    It simply starts the News Orchestrator which manages the whole dialogue.
    """

    def __init__(self):
        super().__init__()
        self._orch = Orchestrator()

    def run(self):
        # Delegate to orchestrator
        self._orch.run()


def main():
    Dialogue().run()


if __name__ == '__main__':
    main()
