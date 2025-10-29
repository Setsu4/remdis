import os
import queue
import threading
import time
import sys
import traceback
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
        # 開発モード時のTTS COMMIT後の遅延秒（任意）
        # Default to 0.0 so that no artificial dev delay is applied unless
        # explicitly configured.
        self.dev_tts_delay_sec = float(self.config.get('DIALOGUE', {}).get('dev_tts_delay_sec', 0.0))

        # If not running in dev_mode, disable the post-TTS artificial pause
        # so normal dialogue proceeds without the extra delay used for
        # development testing.
        if not self.dev_mode:
            try:
                self.post_tts_wait_sec = 0.0
            except Exception:
                pass

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
        # Timestamp of the most recent TTS COMMIT received (time.time()).
        # Used to distinguish stale COMMITs from commits that correspond to
        # the sentence we're currently waiting for.
        self._last_tts_commit_time = 0.0
        # Event indicating TTS is ready to accept a new sentence.
        # Set when no utterance is being synthesized; cleared when we send a new sentence.
        self._tts_ready = threading.Event()
        self._tts_ready.set()
        # Lock to ensure only one sentence is sent to TTS at a time
        self._tts_lock = threading.Lock()
        # Whether we've already announced the prepare-notice in AWAIT_START_CUE
        self._prepare_notice_sent = False

    # --------------- main loop ---------------
    def run(self):
        threading.Thread(target=self.listen_asr_loop, daemon=True).start()
        threading.Thread(target=self.listen_tts_loop, daemon=True).start()

        try:
            # Diagnostic: print the id of the ASR queue and main thread name at startup
            print(f"[{time.time():.6f}] [DIAG] session={self.session_id} asr_queue_id={id(self._asr_queue)} main_thread={threading.current_thread().name}", file=sys.stderr)
        except Exception:
            pass

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

                # Speak main sentence with interruption handling.
                text = node.get('text', '')
                if text:
                    # Send the ADD for the sentence
                    self._say(text)
                    # Send COMMIT to start playback
                    self._commit()
                    # Wait for either normal TTS finish or an interrupting ASR
                    interrupt_text = self._wait_for_tts_or_interrupt(timeout=None)
                    if interrupt_text:
                        # Handle interrupt: attempt to answer via subplan and then re-say
                        handled = self._handle_interruption_during_tts(interrupt_text, self.current_node_id)
                        if handled:
                            # Re-send the same sentence after handling
                            # Only re-send if the node still points to same content
                            # (the handler may update current_node_id to return_to)
                            if text:
                                self._say(text)
                                self._commit()
                                # Wait interruptibly for the re-sent sentence so
                                # users can interrupt it again. If interrupted,
                                # handle and re-send until no more interrupts.
                                while True:
                                    extra_interrupt = self._wait_for_tts_or_interrupt(timeout=None)
                                    if extra_interrupt:
                                        try:
                                            # Handle the nested interruption (may play subplan)
                                            self._handle_interruption_during_tts(extra_interrupt, self.current_node_id)
                                            # After handling, re-send the original sentence again
                                            self._say(text)
                                            self._commit()
                                            continue
                                        except Exception:
                                            # If handler fails, break and proceed
                                            break
                                    # no more interrupts during re-sent sentence
                                    break
                                # After re-sending and handling nested interrupts,
                                # fall through to the normal post-speech logic so
                                # that confirmations for the current node are
                                # evaluated. Do NOT auto-advance the current node here.
                    else:
                        # No interrupt, normal finish
                        self._wait_tts_gap()
                        self.confirm_ctrl.tick_sentence()
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
        try:
            print(f"[{time.time():.6f}] [DIAG] listen_asr_loop thread={threading.current_thread().name} starting subscribe('asr')", file=sys.stderr)
        except Exception:
            pass
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
                try:
                    # Trace that Orchestrator enqueued ASR text (with timestamp)
                    qsize_before = None
                    try:
                        qsize_before = self._asr_queue.qsize()
                    except Exception:
                        pass
                    print(f"[{time.time():.6f}] [ASR-QUEUE] session={self.session_id} enqueuing ASR text: '{text}' asr_queue_id={id(self._asr_queue)} thread={threading.current_thread().name} qsize_before={qsize_before}", file=sys.stderr, flush=True)
                except Exception:
                    pass
                self._asr_queue.put(text)
                try:
                    qsize_after = None
                    try:
                        qsize_after = self._asr_queue.qsize()
                    except Exception:
                        pass
                    print(f"[{time.time():.6f}] [ASR-QUEUE] session={self.session_id} after_put qsize_after={qsize_after}", file=sys.stderr, flush=True)
                except Exception:
                    pass

    def callback_tts(self, ch, method, properties, in_msg):
        iu = self.parse_msg(in_msg)
        if iu.get('update_type') == RemdisUpdateType.COMMIT:
            try:
                # Log arrival time of TTS COMMIT for timing analysis
                print(f"[{time.time():.6f}] [TTS-COMMIT] session={self.session_id} received TTS COMMIT (id={iu.get('id')})", file=sys.stderr)
            except Exception:
                pass
            # Signal that TTS finished the current utterance
            # Record the commit time so waiters can determine whether this
            # COMMIT belongs to the current wait session.
            try:
                self._last_tts_commit_time = time.time()
            except Exception:
                self._last_tts_commit_time = time.time()
            self._tts_commit_event.set()
            # Mark TTS as ready to accept the next sentence.
            # In dev_mode, delay the readiness by dev_tts_delay_sec to allow
            # an artificial pause between sentences for testing.
            def _delayed_set():
                try:
                    time.sleep(float(self.dev_tts_delay_sec))
                except Exception:
                    pass
                try:
                    self._tts_ready.set()
                except Exception:
                    pass

            if getattr(self, 'dev_mode', False):
                t = threading.Thread(target=_delayed_set, daemon=True)
                t.start()
            else:
                try:
                    self._tts_ready.set()
                except Exception:
                    pass

            # Ensure the TTS lock is released so waiting _say() callers can proceed.
            try:
                if self._tts_lock.locked():
                    self._tts_lock.release()
            except Exception:
                pass

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
        # Before starting a new utterance, clear any stale TTS commit event
        # and reset the last-commit timestamp so the waiting routine will
        # treat the subsequent COMMIT as belonging to this new utterance.
        try:
            self._tts_commit_event.clear()
        except Exception:
            pass
        try:
            # resetting to 0.0 ensures comparisons in _wait_for_tts_or_interrupt
            # won't treat an older commit as relevant to the new wait session.
            self._last_tts_commit_time = 0.0
        except Exception:
            pass

        # Acquire lock so concurrent callers will block until this utterance completes
        try:
            self._tts_lock.acquire()
        except Exception:
            pass

        # Wait until TTS is ready to accept a new sentence
        try:
            self._tts_ready.wait()
        except Exception:
            pass

        # Mark TTS as busy for this sentence
        try:
            self._tts_ready.clear()
        except Exception:
            pass

        iu = self.createIU(text, 'dialogue', RemdisUpdateType.ADD)
        self.printIU(iu)
        self.publish(iu, 'dialogue')

    def _commit(self):
        # Clear any stale TTS commit event before issuing a new COMMIT IU
        try:
            self._tts_commit_event.clear()
        except Exception:
            pass
        try:
            self._last_tts_commit_time = 0.0
        except Exception:
            pass

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

    # Send a REVOKE IU on the dialogue exchange to request TTS to stop current playback
    def _send_dialogue_revoke(self):
        try:
            # Visible runtime trace for debugging: indicate we intend to send a REVOKE
            try:
                # Timestamped trace for diagnostic ordering
                print(f"[{time.time():.6f}] [REVOKE] session={self.session_id} node={self.current_node_id} sending dialogue REVOKE", file=sys.stderr)
            except Exception:
                # best-effort; don't fail revoke path if logging fails
                pass

            snd_iu = self.createIU('', 'dialogue', RemdisUpdateType.REVOKE)
            # printIU will emit the IU to stdout/stderr per Remdis conventions
            self.printIU(snd_iu)
            self.publish(snd_iu, 'dialogue')

            try:
                print(f"[{time.time():.6f}] [REVOKE] published dialogue REVOKE id={snd_iu.get('id')}", file=sys.stderr)
            except Exception:
                pass
        except Exception:
            # Surface unexpected errors to stderr to avoid silent failures
            try:
                print('[REVOKE] exception while sending dialogue REVOKE', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            except Exception:
                pass

    # Wait for either TTS COMMIT or an incoming ASR (interrupt). If an ASR is received
    # while waiting, return the text (str). If TTS finishes normally, return None.
    def _wait_for_tts_or_interrupt(self, timeout: Optional[float] = None) -> Optional[str]:
        start = time.time()
        try:
            # Mark when we start waiting so we can see how long wait loops run
            print(f"[{time.time():.6f}] [WAIT-ENTER] session={self.session_id} waiting for TTS or ASR (timeout={timeout})", file=sys.stderr, flush=True)
        except Exception:
            pass
        while True:
            # Quick non-blocking check for ASR that may have arrived just before
            # entering the main wait logic. This ensures we don't miss user
            # interruptions that occurred immediately after sending the IU.
            try:
                try:
                    pre_txt = self._asr_queue.get_nowait()
                except Exception:
                    pre_txt = None
                if pre_txt:
                    try:
                        print(f"[{time.time():.6f}] [ASR-DETECT-PRE] session={self.session_id} detected ASR before waiting: '{pre_txt}'", file=sys.stderr, flush=True)
                    except Exception:
                        pass
                    if 'システムリセット' in pre_txt:
                        try:
                            self._asr_queue.put(pre_txt)
                        except Exception:
                            pass
                        return None
                    return pre_txt
            except Exception:
                pass

            try:
                print(f"[{time.time():.6f}] [WAIT-ITER] session={self.session_id}", file=sys.stderr, flush=True)
            except Exception:
                pass
            # Check if tts finished
            try:
                if self._tts_commit_event.wait(timeout=0.1):
                    # Only treat this COMMIT as relevant if it occurred after
                    # we started waiting in this session. This avoids reacting
                    # to stale commit events from previous sentences.
                    if self._last_tts_commit_time and self._last_tts_commit_time < start:
                        try:
                            print(f"[{time.time():.6f}] [WAIT-IGNORED-STALE-COMMIT] session={self.session_id} commit_time={self._last_tts_commit_time} wait_start={start}", file=sys.stderr, flush=True)
                        except Exception:
                            pass
                        # clear the event and continue waiting
                        self._tts_commit_event.clear()
                        continue
                    self._tts_commit_event.clear()
                    # Before declaring normal finish, do several short blocking
                    # checks of the ASR queue to catch near-simultaneous user input.
                    try:
                        txt_after = None
                        attempts = 3
                        attempt_timeout = 0.05
                        for ai in range(attempts):
                            try:
                                try:
                                    qsz = self._asr_queue.qsize()
                                except Exception:
                                    qsz = None
                                print(f"[{time.time():.6f}] [WAIT-POST-TTS-CHECK] session={self.session_id} attempt={ai+1}/{attempts} about to blocking-get(timeout={attempt_timeout}) asr_queue_id={id(self._asr_queue)} thread={threading.current_thread().name} qsize_before={qsz}", file=sys.stderr, flush=True)
                            except Exception:
                                pass
                            try:
                                txt_after = self._asr_queue.get(timeout=attempt_timeout)
                                break
                            except queue.Empty:
                                txt_after = None
                                continue
                            except Exception as e:
                                try:
                                    print(f"[{time.time():.6f}] [WAIT-POST-TTS-CHECK-EXC] session={self.session_id} get exception: {e}", file=sys.stderr, flush=True)
                                    traceback.print_exc(file=sys.stderr)
                                except Exception:
                                    pass
                                txt_after = None
                                break

                        # If still nothing, fall back to a non-blocking check once
                        if not txt_after:
                            try:
                                try:
                                    qsz2 = self._asr_queue.qsize()
                                except Exception:
                                    qsz2 = None
                                print(f"[{time.time():.6f}] [WAIT-POST-TTS-CHECK] no ASR after {attempts} attempts, trying get_nowait asr_queue_id={id(self._asr_queue)} thread={threading.current_thread().name} qsize_now={qsz2}", file=sys.stderr, flush=True)
                            except Exception:
                                pass
                            try:
                                txt_after = self._asr_queue.get_nowait()
                            except Exception:
                                txt_after = None

                        if txt_after:
                            try:
                                print(f"[{time.time():.6f}] [ASR-DETECT-AFTER-TTS] session={self.session_id} detected ASR immediately after TTS COMMIT: '{txt_after}'", file=sys.stderr, flush=True)
                            except Exception:
                                pass
                            if 'システムリセット' in txt_after:
                                try:
                                    self._asr_queue.put(txt_after)
                                except Exception:
                                    pass
                                return None
                            return txt_after
                    except Exception:
                        # any unexpected error should not prevent normal finish
                        pass
                    # normal finish
                    return None
            except Exception as e:
                try:
                    print(f"[{time.time():.6f}] [WAIT-LOOP-EXC] session={self.session_id} exception: {e}", file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)
                except Exception:
                    pass

            # Check for ASR input (user interruption)
            try:
                try:
                    qsize = self._asr_queue.qsize()
                    print(f"[{time.time():.6f}] [WAIT-LOOP] session={self.session_id} asr_queue_size={qsize}", file=sys.stderr, flush=True)
                except Exception:
                    qsize = None
                try:
                    print(f"[{time.time():.6f}] [WAIT-LOOP-CHECK] session={self.session_id} about to short-block-get(timeout=0.02) asr_queue_id={id(self._asr_queue)} thread={threading.current_thread().name} qsize={qsize}", file=sys.stderr, flush=True)
                except Exception:
                    pass
                try:
                    # Short blocking get reduces tight busy-looping and reduces
                    # races where another thread hasn't yet scheduled the put().
                    txt = self._asr_queue.get(timeout=0.02)
                except Exception as e:
                    # Empty is expected; suppress noisy traceback for normal case
                    if isinstance(e, queue.Empty):
                        txt = None
                    else:
                        try:
                            print(f"[{time.time():.6f}] [WAIT-GET-EXC] session={self.session_id} get exception: {e}", file=sys.stderr, flush=True)
                            traceback.print_exc(file=sys.stderr)
                        except Exception:
                            pass
                        txt = None
                if txt:
                    try:
                        print(f"[{time.time():.6f}] [ASR-DETECT] session={self.session_id} detected ASR while waiting: '{txt}'", file=sys.stderr, flush=True)
                    except Exception:
                        pass
                    # If it's a reset command, put it back for outer loop handling
                    if 'システムリセット' in txt:
                        try:
                            self._asr_queue.put(txt)
                        except Exception:
                            pass
                        return None
                    return txt
            except Exception:
                pass

            # Timeout handling
            if timeout is not None and (time.time() - start) >= timeout:
                return None

    # Handle an interruption utterance that arrived during a system utterance.
    # This will request TTS to stop, attempt to answer the user's question (subplan
    # matching), and keep track so the caller can re-say the original utterance.
    def _handle_interruption_during_tts(self, user_text: str, current_node_id: str) -> bool:
        try:
            # Ask TTS to stop immediately
            self._send_dialogue_revoke()
            # Wait briefly for TTS to acknowledge the REVOKE by sending a
            # TTS COMMIT. If we don't wait, we may start producing a reply
            # while the previous audio is still being streamed which leads
            # to overlapping / continued playback. Use a modest timeout so
            # we don't block the dialog flow for too long.
            try:
                self._wait_for_tts_stop(timeout=1.0)
            except Exception:
                # best-effort: continue even if wait fails
                pass
            # Special-case: user asks "なんて言った" -> reply with the subject
            # We accept common variants including kanji/kanakana
            if isinstance(user_text, str):
                ut = user_text.strip()
                if any(k in ut for k in ['なんて言った', '何て言った', 'なんていった', '何ていった']):
                    # Reply succinctly with the subject of the current sentence
                    self._say("国学院大学です")
                    self._commit()
                    self._wait_tts_gap()
                    # Keep dialog state in MAIN so original sentence can be re-sent
                    self._transition('MAIN')
                    return True

            # Try to match subplans for the current node
            subplans = self.plan_store.get_subplans(current_node_id) if current_node_id else []
            if subplans:
                match = best_subplan_match(user_text, subplans, self.subplan_threshold)
                if match:
                    sp, _ = match
                    self._transition('SUBPLAN')
                    self._say(sp.get('answer', ''))
                    self._commit()
                    # Wait for the answer to finish normally (no nested interruption handling)
                    self._wait_tts_gap()
                    # Return to MAIN. Do NOT change current_node_id here so that the
                    # original sentence can be re-sent and progression continues as before.
                    self._transition('MAIN')
                    return True

            # No matching subplan: give a short fallback response
            self._say("その質問には簡単にはお答えできません。説明に戻ります。")
            self._commit()
            self._wait_tts_gap()
            return True
        except Exception:
            return False

    # Wait for TTS to acknowledge a stop via receiving a TTS COMMIT.
    # Returns True if a commit was observed within the timeout, False otherwise.
    def _wait_for_tts_stop(self, timeout: float = 1.0) -> bool:
        start = time.time()
        try:
            # If TTS already signalled commit recently, consider it stopped.
            if self._tts_commit_event.is_set():
                try:
                    self._tts_commit_event.clear()
                except Exception:
                    pass
                return True
            # Wait in short intervals so we remain responsive to other events
            while (time.time() - start) < timeout:
                # wait with a small timeout so we can loop and remain responsive
                if self._tts_commit_event.wait(timeout=0.05):
                    try:
                        self._tts_commit_event.clear()
                    except Exception:
                        pass
                    return True
            return False
        except Exception:
            return False

    def _error_and_end(self, message: str):
        self._say(message)
        self._commit()
        self._wait_tts_gap()
        self._transition('END')


def main():
    Orchestrator().run()


if __name__ == '__main__':
    main()
