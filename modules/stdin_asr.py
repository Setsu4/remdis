import sys
import threading
import time

from base import RemdisModule, RemdisUpdateType

class StdinASR(RemdisModule):
    """Simple ASR replacement that reads user input lines from stdin.

    Behavior:
    - For each non-empty line entered, publish an ADD IU for each token (or the whole line) and then a COMMIT IU.
    - Special commands:
      - "exit" or Ctrl-D: stop the module.
      - empty line: ignored.

    This is intended for development / testing where microphone ASR isn't needed.
    """

    def __init__(self, pub_exchanges=['asr']):
        super().__init__(pub_exchanges=pub_exchanges)
        self._is_running = True

    def run(self):
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()
        t.join()

    def _read_loop(self):
        try:
            print("[stdin_asr] Type a line and press Enter to send. Type 'exit' to quit.")
            for line in sys.stdin:
                line = line.rstrip('\n')
                if line is None:
                    break
                text = line.strip()
                if text == 'exit':
                    break
                if text == '':
                    continue

                # Publish ADD IU (we send the whole line as one token)
                iu = self.createIU(text, 'asr', RemdisUpdateType.ADD)
                self.printIU(iu)
                self.publish(iu, 'asr')

                # Small pause to mimic incremental tokens
                time.sleep(0.01)

                # Publish COMMIT IU to mark end of utterance
                commit = self.createIU('', 'asr', RemdisUpdateType.COMMIT)
                self.printIU(commit)
                self.publish(commit, 'asr')

        except Exception as e:
            print(f"[stdin_asr] Error: {e}")

if __name__ == '__main__':
    m = StdinASR()
    m.run()
