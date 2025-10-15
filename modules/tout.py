import time
import threading

from base import RemdisModule, RemdisUpdateType

class TOUT(RemdisModule):
    def __init__(self,
                 sub_exchanges=['asr', 'dialogue', 'dialogue2'],
                 pub_exchanges=['tts']):
        super().__init__(pub_exchanges=pub_exchanges,
                         sub_exchanges=sub_exchanges)
        self._is_running = True
    
    import threading

    from base import RemdisModule


    class TOUT(RemdisModule):
        """Simple logger for ASR/dialogue/dialogue2 exchanges."""

        def __init__(self, sub_exchanges=['asr', 'dialogue', 'dialogue2']):
            super().__init__(sub_exchanges=sub_exchanges)

        def run(self):
            threading.Thread(target=self.listen_asr_loop, daemon=True).start()
            threading.Thread(target=self.listen_dialogue_loop, daemon=True).start()
            threading.Thread(target=self.listen_dialogue2_loop, daemon=True).start()
            # keep main thread alive
            threading.Event().wait()

        def listen_asr_loop(self):
            self.subscribe('asr', self.callback_asr)

        def listen_dialogue_loop(self):
            self.subscribe('dialogue', self.callback_dialogue)

        def listen_dialogue2_loop(self):
            self.subscribe('dialogue2', self.callback_dialogue2)

        def callback_asr(self, ch, method, properties, in_msg):
            msg = self.parse_msg(in_msg)
            print(f"[TOUT][ASR] {msg.get('update_type')} body={msg.get('body')}")

        def callback_dialogue(self, ch, method, properties, in_msg):
            msg = self.parse_msg(in_msg)
            print(f"[TOUT][DIALOGUE] {msg.get('update_type')} body={msg.get('body')}")

        def callback_dialogue2(self, ch, method, properties, in_msg):
            msg = self.parse_msg(in_msg)
            print(f"[TOUT][DIALOGUE2] {msg.get('update_type')} body={msg.get('body')}")


    if __name__ == '__main__':
        TOUT().run()