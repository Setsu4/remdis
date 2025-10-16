import sys
import threading
import queue
import time
import re

from base import RemdisModule, RemdisUpdateType
from news_orchestrator import Orchestrator


class Dialogue(RemdisModule):
    """Thin wrapper to keep existing entry point.
    It simply starts the News Orchestrator which manages the whole dialogue.
    """

    def __init__(self):
        super().__init__()
        self._orch = Orchestrator()
        # If running in dev_mode, subscribe to tts to log COMMITs for debugging
        try:
            dev_mode = bool(self.config.get('DIALOGUE', {}).get('dev_mode', False))
        except Exception:
            dev_mode = False
        if dev_mode:
            # subscribe in a background thread to avoid blocking init
            # Use the orchestrator's subscription since this Module may not
            # have a 'tts' entry in its own sub_connections. Wrap in try/except
            # to avoid crashing the thread if the exchange isn't available.
            def _start_dev_tts_sub():
                try:
                    self._orch.subscribe('tts', self._callback_tts)
                except Exception as e:
                    print(f"[dialogue] dev-mode tts subscribe failed: {e}", file=sys.stderr)

            t = threading.Thread(target=_start_dev_tts_sub, daemon=True)
            t.start()

    def _callback_tts(self, ch, method, properties, in_msg):
        try:
            iu = self.parse_msg(in_msg)
            if iu.get('update_type') == RemdisUpdateType.COMMIT:
                # Print a concise debug line for COMMIT
                print(f"[dev-mode] TTS COMMIT received: id={iu.get('id')} channel={iu.get('channel')}")
        except Exception:
            pass

    def run(self):
        # Delegate to orchestrator
        self._orch.run()


def main():
    Dialogue().run()


if __name__ == '__main__':
    main()
