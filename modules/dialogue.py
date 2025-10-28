import sys
import traceback
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
        # create our own subscription connection for 'tts' to avoid
        # sharing the Orchestrator's channel (which can cause pika errors
        # when two different consumers try to use the same channel)
        super().__init__(sub_exchanges=['tts'])
        self._orch = Orchestrator()
        # If running in dev_mode, subscribe to tts to log COMMITs for debugging
        try:
            dev_mode = bool(self.config.get('DIALOGUE', {}).get('dev_mode', False))
        except Exception:
            dev_mode = False
        if dev_mode:
            # Ensure we have subscription connections for dev-mode exchanges
            for extra_ex in ('asr', 'dialogue'):
                if extra_ex not in self.sub_connections:
                    try:
                        self.sub_connections[extra_ex] = self.mk_sub_connection(extra_ex)
                        print(f"[dialogue] dev-mode created sub_connection for '{extra_ex}'", file=sys.stderr)
                    except Exception as e:
                        print(f"[dialogue] failed to create sub_connection for '{extra_ex}': {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
            # subscribe in a background thread to avoid blocking init
            # Use this module's own subscription (self.subscribe) so it
            # uses a separate connection/channel from the Orchestrator.
            def _start_dev_subs():
                # Start each subscribe in its own thread because subscribe() calls
                # start_consuming() and blocks. Running them in separate threads
                # lets all dev subscriptions be active concurrently.
                try:
                    t_tts = threading.Thread(target=self.subscribe, args=('tts', self._callback_tts), daemon=True)
                    t_tts.start()
                    print(f"[dialogue] dev-mode started subscribe thread for 'tts'", file=sys.stderr)
                except Exception as e:
                    print(f"[dialogue] dev-mode tts subscribe failed to start thread: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                try:
                    t_asr = threading.Thread(target=self.subscribe, args=('asr', self._callback_asr), daemon=True)
                    t_asr.start()
                    print(f"[dialogue] dev-mode started subscribe thread for 'asr'", file=sys.stderr)
                except Exception as e:
                    print(f"[dialogue] dev-mode asr subscribe failed to start thread: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                try:
                    t_diag = threading.Thread(target=self.subscribe, args=('dialogue', self._callback_dialogue), daemon=True)
                    t_diag.start()
                    print(f"[dialogue] dev-mode started subscribe thread for 'dialogue'", file=sys.stderr)
                except Exception as e:
                    print(f"[dialogue] dev-mode dialogue subscribe failed to start thread: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

            t = threading.Thread(target=_start_dev_subs, daemon=True)
            t.start()

    def _format_iu_body(self, iu: dict, max_len: int = 120) -> str:
        """
        Return a short, human-readable representation of an IU body to avoid
        dumping very long strings (e.g. base64-encoded audio) into logs.

        Format: if body is large string, show prefix + '...' and append
        metadata such as length and data_type when available.
        """
        try:
            body = iu.get('body', '')
            if not isinstance(body, str):
                # Non-string bodies (numbers, dicts) - show repr
                return repr(body)

            # If body looks like base64 audio (long and contains many +/=/ characters),
            # replace with a brief placeholder including length.
            body_len = len(body)
            data_type = iu.get('data_type') or iu.get('dtype') or ''
            if body_len > max_len:
                prefix = body[:max_len].replace('\n', '')
                return f"{prefix}... (len={body_len}, type={data_type or 'str'})"

            # Short strings: show as-is
            return body
        except Exception:
            return '<unprintable body>'

    def _callback_tts(self, ch, method, properties, in_msg):
        try:
            iu = self.parse_msg(in_msg)
            # Suppress noisy audio-chunk ADD logs: print only REVOKE/COMMIT or non-audio bodies
            ut = iu.get('update_type')
            data_type = iu.get('data_type') or iu.get('dtype') or ''
            # If it's an audio chunk ADD, skip; otherwise log concise info
            if ut not in (RemdisUpdateType.COMMIT, RemdisUpdateType.REVOKE) and data_type == 'audio':
                return
            body_str = self._format_iu_body(iu)
            try:
                print(f"[{time.time():.6f}] [dev-mode] TTS IU: id={iu.get('id')} type={ut} channel={iu.get('channel')} body={body_str}")
            except Exception:
                print(f"[dev-mode] TTS IU: id={iu.get('id')} type={ut} channel={iu.get('channel')} body={body_str}")
        except Exception:
            print('[dialogue] dev-mode _callback_tts exception', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def _callback_asr(self, ch, method, properties, in_msg):
        try:
            iu = self.parse_msg(in_msg)
            # Log ASR IUs so we can see when user utterances arrive
            body_str = self._format_iu_body(iu)
            try:
                print(f"[{time.time():.6f}] [dev-mode] ASR IU: id={iu.get('id')} type={iu.get('update_type')} body={body_str}")
            except Exception:
                print(f"[dev-mode] ASR IU: id={iu.get('id')} type={iu.get('update_type')} body={body_str}")
        except Exception:
            print('[dialogue] dev-mode _callback_asr exception', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def _callback_dialogue(self, ch, method, properties, in_msg):
        try:
            iu = self.parse_msg(in_msg)
            # Log dialogue IUs to observe REVOKE sent by Orchestrator
            body_str = self._format_iu_body(iu)
            try:
                print(f"[{time.time():.6f}] [dev-mode] dialogue IU: id={iu.get('id')} type={iu.get('update_type')} body={body_str}")
            except Exception:
                print(f"[dev-mode] dialogue IU: id={iu.get('id')} type={iu.get('update_type')} body={body_str}")
        except Exception:
            print('[dialogue] dev-mode _callback_dialogue exception', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def run(self):
        # Delegate to orchestrator
        self._orch.run()


def main():
    Dialogue().run()


if __name__ == '__main__':
    main()
