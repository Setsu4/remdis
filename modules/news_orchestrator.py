import os
import queue
import threading
import time
from typing import Optional, Dict, Any

from base import RemdisModule, RemdisUpdateType
from plan_store import PlanStore
from plan_generator import PlanGenerator
from similarity import best_subplan_match
from confirmation import ConfirmationController, ConfirmationPolicy


class Orchestrator(RemdisModule):
    """
    Dialogue Orchestrator implementing the spec using a pre-made example plan.

    - Publishes system utterances to 'dialogue' (ADD/COMMIT IUs)
    - Listens to 'asr' for user utterances (ADD/COMMIT)
    - Listens to 'tts' COMMIT to apply inter-utterance wait globally
    - States: INIT, PREPARE, READY, AWAIT_START_CUE, MAIN, CONFIRM, EXPLAIN, SUBPLAN, END
    - Reset phrase: "システムリセット"
    - Start cue: contains "今日のニュースは"
    """

    def __init__(self,
                 pub_exchanges=['dialogue'],
                 sub_exchanges=['asr', 'tts']):
        super().__init__(pub_exchanges=pub_exchanges,
                         sub_exchanges=sub_exchanges)

        # Config
        self.prepare_notice_interval_sec = self.config.get('prepare_notice_interval_sec', 5)
        self.confirm_min_gap_sentences = self.config.get('confirm_min_gap_sentences', 2)
        self.confirm_max_per_session = self.config.get('confirm_max_per_session', 6)
        self.subplan_threshold = self.config.get('subplan_threshold', 0.75)
        self.user_response_timeout_sec = self.config.get('user_response_timeout_sec', 8)
        self.sentence_interval_sec = self.config.get('sentence_interval_sec', 2)
        # 新設定: post_tts_wait_sec（なければ sentence_interval_sec を流用）
        self.post_tts_wait_sec = self.config.get('post_tts_wait_sec', self.sentence_interval_sec)
        # 動作確認モード: 開発時に開始キーワード待ちをスキップする
        self.dev_mode = bool(self.config.get('DIALOGUE', {}).get('dev_mode', False))

        # Plan and controllers
        self.plan_store = PlanStore()
        self.plan_generator = PlanGenerator()
        self.confirm_ctrl = ConfirmationController(ConfirmationPolicy(
            min_gap_sentences=self.confirm_min_gap_sentences,
            max_per_session=self.confirm_max_per_session,
        ))

        # Runtime
        self.state = 'INIT'
        self.current_node_id = None
        self.session_id = f"sess-{int(time.time())}"

        # Buffers and flags
        self._iu_buffer = []
        self._asr_queue = queue.Queue()
        self._running = True
        self._tts_commit_event = threading.Event()
        # Whether we've already announced the prepare-notice in AWAIT_START_CUE
        self._prepare_notice_sent = False

    # --------------- main loop ---------------
    def run(self):
        threading.Thread(target=self.listen_asr_loop, daemon=True).start()
        threading.Thread(target=self.listen_tts_loop, daemon=True).start()

        # Prepare
        self._transition('PREPARE')
        self._prepare()
        self._transition('READY')
        # If dev_mode is enabled, skip AWAIT_START_CUE and start MAIN immediately
        if self.dev_mode:
            self._transition('MAIN')
            self.current_node_id = self.plan_store.get_start_main_id()
            if not self.current_node_id:
                self._error_and_end("発話計画の開始位置が見つかりませんでした")
        else:
            self._transition('AWAIT_START_CUE')

        while self._running:
            if self._check_reset():
                self._transition('PREPARE')
                self._prepare()
                self._transition('READY')
                self._transition('AWAIT_START_CUE')
                continue

            if self.state == 'AWAIT_START_CUE':
                # Announce the prepare-notice only once per AWAIT_START_CUE entry
                if not getattr(self, '_prepare_notice_sent', False):
                    self._say("準備ができました。『今日のニュースは』と話しかけてください。")
                    self._commit()
                    self._wait_tts_gap()
                    self._prepare_notice_sent = True

                # Always wait for user ASR input (but don't re-announce)
                text = self._wait_asr(timeout=self.user_response_timeout_sec)
                if text and '今日のニュース' in text:
                    self._transition('MAIN')
                    self.current_node_id = self.plan_store.get_start_main_id()
                    if not self.current_node_id:
                        self._error_and_end("発話計画の開始位置が見つかりませんでした")
                        break
                    continue
                continue

            if self.state == 'MAIN':
                if self.current_node_id is None:
                    if self.plan_store.has_node('end'):
                        self._transition('END')
                        continue
                    self._error_and_end("発話計画の実行位置が不明です")
                    break

                if self.plan_store.is_end(self.current_node_id):
                    self._transition('END')
                    continue

                node = self.plan_store.get(self.current_node_id)
                if not node:
                    self._error_and_end("データ参照に不整合がありました")
                    break

                # Speak main sentence
                text = node.get('text', '')
                if text:
                    self._say(text)
                self._commit()
                self._wait_tts_gap()
                self.confirm_ctrl.tick_sentence()

                # Maybe insert confirmation
                confs = self.plan_store.get_confirmations(self.current_node_id)
                do_confirm = bool(confs) and self.confirm_ctrl.allow_insert()
                if do_confirm:
                    self._transition('CONFIRM')
                    conf = confs[0]
                    next_id = self._do_confirmation(conf)
                    self.confirm_ctrl.mark_used()
                    if next_id:
                        if next_id.startswith('explain_'):
                            self._transition('EXPLAIN')
                            self._do_explain(next_id)
                            self._transition('MAIN')
                            self.current_node_id = self.plan_store.get_next_from_explain(next_id)
                            continue
                        else:
                            self._transition('MAIN')
                            self.current_node_id = next_id
                            continue

                # Check subplan
                subplans = self.plan_store.get_subplans(self.current_node_id)
                if subplans:
                    user_text = self._wait_asr(timeout=self.user_response_timeout_sec)
                    if user_text:
                        if 'システムリセット' in user_text:
                            continue
                        match = best_subplan_match(user_text, subplans, self.subplan_threshold)
                        if match:
                            sp, _ = match
                            self._transition('SUBPLAN')
                            self._say(sp.get('answer', ''))
                            self._commit()
                            self._wait_tts_gap()
                            self._transition('MAIN')
                            self.current_node_id = sp.get('return_to')
                            continue
                        else:
                            self._say("その質問には簡単にはお答えできません。説明に戻ります。")
                            self._commit()
                            self._wait_tts_gap()

                nxt = self.plan_store.next_main_id(self.current_node_id)
                if nxt is None:
                    self._transition('END')
                else:
                    self.current_node_id = nxt
                continue

            if self.state == 'END':
                if self.plan_store.has_node('end'):
                    end_text = self.plan_store.get_text('end')
                    if end_text:
                        self._say(end_text)
                self._commit()
                self._wait_tts_gap()
                break

        self._running = False

    # --------------- listen / ASR handling ---------------
    def listen_asr_loop(self):
        self.subscribe('asr', self.callback_asr)

    def listen_tts_loop(self):
        self.subscribe('tts', self.callback_tts)

    def callback_asr(self, ch, method, properties, in_msg):
        iu = self.parse_msg(in_msg)
        ut = iu.get('update_type')
        body = iu.get('body', '')
        if ut == RemdisUpdateType.ADD and isinstance(body, str):
            self._iu_buffer.append(body)
        elif ut == RemdisUpdateType.REVOKE:
            self._iu_buffer = []
        elif ut == RemdisUpdateType.COMMIT:
            text = ''.join(self._iu_buffer).strip()
            self._iu_buffer = []
            if text:
                self._asr_queue.put(text)

    def callback_tts(self, ch, method, properties, in_msg):
        iu = self.parse_msg(in_msg)
        if iu.get('update_type') == RemdisUpdateType.COMMIT:
            self._tts_commit_event.set()

    def _wait_asr(self, timeout: float) -> Optional[str]:
        try:
            if timeout is None:
                return self._asr_queue.get()
            return self._asr_queue.get(timeout=timeout)
        except Exception:
            return None

    def _check_reset(self) -> bool:
        try:
            txt = self._asr_queue.get_nowait()
            if 'システムリセット' in txt:
                return True
            self._asr_queue.put(txt)
        except Exception:
            pass
        return False

    def _drain_asr_queue(self):
        try:
            while True:
                self._asr_queue.get_nowait()
        except Exception:
            pass

    # --------------- confirmation / explain ---------------
    def _do_confirmation(self, conf: Dict[str, Any]) -> Optional[str]:
        self._drain_asr_queue()
        self._say(conf.get('text', ''))
        self._commit()
        self._wait_tts_gap()
        while True:
            ans = self._wait_asr(timeout=None)
            if not ans:
                continue
            if 'システムリセット' in ans:
                try:
                    self._asr_queue.put(ans)
                except Exception:
                    pass
                return None
            break

        cat = self._classify_confirmation_answer(ans)
        ur = conf.get('user_responses', {}) or {}
        return ur.get(cat) or ur.get('曖昧') or ur.get('知らない') or ur.get('ない') or ur.get('知っている')

    def _classify_confirmation_answer(self, text: str) -> str:
        t = text.strip()
        if any(k in t for k in ['知っ', 'わかる', '分かる', '大丈夫', 'OK', 'いい']):
            return '知っている'
        if any(k in t for k in ['知ら', 'わから', '分から', '初めて', '何それ']):
            return '知らない'
        if any(k in t for k in ['ある', '使った']):
            return 'ある'
        if any(k in t for k in ['ない', '使ってない']):
            return 'ない'
        return '曖昧'

    def _do_explain(self, explain_id: str):
        text = self.plan_store.get_text(explain_id) or ''
        if text:
            self._say(text)
            self._commit()
            self._wait_tts_gap()

    # --------------- plan loading / transitions ---------------
    def _prepare(self):
        self._say("対話の準備中です。少々お待ちください。")
        try:
            base_dir = os.path.dirname(__file__)
            plan_path = os.path.normpath(os.path.join(base_dir, '../prompt/plan_example.json'))
            plan = self.plan_generator.generate_from_example(plan_path)
            self.plan_store.set_plan(plan, plan_id=f"plan-{int(time.time())}")
            ok, errs = self.plan_store.validate()
            if not ok:
                self._error_and_end("発話計画の生成に失敗しました: " + '; '.join(errs))
                return
            self._commit()
            self._wait_tts_gap()
        except Exception as e:
            self._error_and_end(f"発話計画の生成に失敗しました: {e}")

    def _transition(self, new_state: str):
        print(f"[TRACE] session={self.session_id} state={self.state} -> {new_state}")
        self.state = new_state
        # Reset prepare notice flag whenever entering AWAIT_START_CUE
        try:
            if new_state == 'AWAIT_START_CUE':
                self._prepare_notice_sent = False
        except Exception:
            pass

    # --------------- output helpers ---------------
    def _say(self, text: str):
        if not text:
            return
        iu = self.createIU(text, 'dialogue', RemdisUpdateType.ADD)
        self.printIU(iu)
        self.publish(iu, 'dialogue')

    def _commit(self):
        iu = self.createIU('', 'dialogue', RemdisUpdateType.COMMIT)
        self.printIU(iu)
        self.publish(iu, 'dialogue')

    def _pause_after_tts(self):
        try:
            time.sleep(self.post_tts_wait_sec)
        except Exception:
            pass

    def _wait_tts_gap(self):
        try:
            received = self._tts_commit_event.wait(timeout=max(0.1, self.post_tts_wait_sec * 2))
            if received:
                self._tts_commit_event.clear()
            self._pause_after_tts()
        except Exception:
            self._pause_after_tts()

    def _error_and_end(self, message: str):
        self._say(message)
        self._commit()
        self._wait_tts_gap()
        self._transition('END')


def main():
    Orchestrator().run()


if __name__ == '__main__':
    main()
