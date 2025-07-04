import sys
import threading
import queue
import time
import re

from base import RemdisModule, RemdisState, RemdisUtil, RemdisUpdateType
from llm import ResponseChatGPT
import prompt.util as prompt_util
from Summary_Generator import SummaryGenerator


class Dialogue(RemdisModule):
    def __init__(self, 
                 pub_exchanges=['dialogue', 'dialogue2'],
                 sub_exchanges=['asr', 'vap', 'tts', 'bc', 'emo_act']):
        super().__init__(pub_exchanges=pub_exchanges,
                         sub_exchanges=sub_exchanges)

        # 設定の読み込み
        self.history_length = self.config['DIALOGUE']['history_length']
        self.response_generation_interval = self.config['DIALOGUE']['response_generation_interval']
        self.prompts = prompt_util.load_prompts(self.config['ChatGPT']['prompts'])

        # ニュース要約の生成
        self.generate_news_summary()

        # 対話履歴
        self.dialogue_history = []

        # IUおよび応答の処理用バッファ
        self.system_utterance_end_time = 0.0
        self.input_iu_buffer = queue.Queue()
        self.bc_iu_buffer = queue.Queue()
        self.emo_act_iu_buffer = queue.Queue()
        self.output_iu_buffer = []
        self.llm_buffer = queue.Queue()

        # 対話状態管理
        self.event_queue = queue.Queue()
        self.state = 'idle'
        self._is_running = True

        # IU処理用の関数
        self.util_func = RemdisUtil()

        self.waiting_for_keyword = True
        self.start_keyword = "どんなニュースがある"

        # 要約文リストとインデックス
        self.summary_sentences = []
        self.summary_index = 0
        self.load_summary_sentences()

    def load_summary_sentences(self):
        """要約文を分割してリストとして保持"""
        generator = SummaryGenerator(self.config, self.prompts)
        self.summary_sentences = generator.load_and_split_summary("../news/summary.txt")
        self.summary_index = 0

    def speak_next_summary(self):
        """次の要約文をインデックス付きで発話"""
        if not self.summary_sentences:
            self.log("No summary sentences loaded.")
            return
        if self.summary_index >= len(self.summary_sentences):
            self.log("All summary sentences have been spoken.")
            return
        total = len(self.summary_sentences)
        idx = self.summary_index + 1
        text = f"[{idx}/{total}] {self.summary_sentences[self.summary_index]}。"
        snd_iu = self.createIU(text, 'dialogue', RemdisUpdateType.ADD)
        self.printIU(snd_iu)
        self.publish(snd_iu, 'dialogue')
        self.summary_index += 1

    # メインループ
    def run(self):
        # 準備中メッセージ
        prep_msg = "対話の準備中です"
        snd_iu = self.createIU(prep_msg, 'dialogue', RemdisUpdateType.ADD)
        self.printIU(snd_iu)
        self.publish(snd_iu, 'dialogue')

        # ニュース分割処理
        self.load_summary_sentences()
        # DEBUG: 分割済み要約文リスト表示（後で削除可）
        print("[DEBUG] 分割済み要約文リスト:")
        for i, s in enumerate(self.summary_sentences):
            print(f"  [{i+1}] {s}")
        time.sleep(0.5)  # 分割処理の安定化のための短い待機

        # 準備完了メッセージ
        ready_msg = "対話開始の合図をお願いします。合図はどんなニュースがある、です。"
        snd_iu2 = self.createIU(ready_msg, 'dialogue', RemdisUpdateType.ADD)
        self.printIU(snd_iu2)
        self.publish(snd_iu2, 'dialogue')

        # 以降は通常のスレッド起動
        t1 = threading.Thread(target=self.listen_asr_loop)
        t2 = threading.Thread(target=self.listen_tts_loop)
        t3 = threading.Thread(target=self.listen_vap_loop)
        t4 = threading.Thread(target=self.listen_bc_loop)
        t5 = threading.Thread(target=self.listen_emo_act_loop)
        t6 = threading.Thread(target=self.parallel_response_generation)
        t7 = threading.Thread(target=self.state_management)
        t8 = threading.Thread(target=self.emo_act_management)
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()
        t7.start()
        t8.start()

    # 音声認識結果受信スレッド
    def listen_asr_loop(self):
        self.subscribe('asr', self.callback_asr)

    # 音声合成結果受信スレッド
    def listen_tts_loop(self):
        self.subscribe('tts', self.callback_tts)

    # VAP情報受信用のコールバックを登録
    def listen_vap_loop(self):
        self.subscribe('vap', self.callback_vap)

    # バックチャネル受信用のコールバックを登録
    def listen_bc_loop(self):
        self.subscribe('bc', self.callback_bc)

    # 表情・行動情報受信用のコールバックを登録
    def listen_emo_act_loop(self):
        self.subscribe('emo_act', self.callback_emo_act)

    # 随時受信される音声認識結果に対して並列に応答を生成
    def parallel_response_generation(self):
        iu_memory = []
        while True:
            input_iu = self.input_iu_buffer.get()
            iu_memory.append(input_iu)

            # IUがREVOKEだった場合はメモリから削除
            if input_iu['update_type'] == RemdisUpdateType.REVOKE:
                iu_memory = self.util_func.remove_revoked_ius(iu_memory)
            # COMMITの場合のみ応答生成
            elif input_iu['update_type'] == RemdisUpdateType.COMMIT:
                user_utterance = self.util_func.concat_ius_body(iu_memory)
                if user_utterance == '':
                    iu_memory = []
                    continue

                # --- ここでキーワード検出 ---
                if self.waiting_for_keyword:
                    if self.start_keyword in user_utterance:
                        self.waiting_for_keyword = False
                        self.log("ニュース伝達モードを開始します。")
                        # 必要ならここで「ニュースをお伝えします」などの初期応答を生成
                    else:
                        self.log("開始キーワード待機中: 入力無視")
                        iu_memory = []
                        continue  # キーワード検出までは無視

                # 通常の応答生成処理
                llm = ResponseChatGPT(self.config, self.prompts, news_path="../news/news.txt")
                last_asr_iu_id = input_iu['id']
                t = threading.Thread(
                    target=llm.run,
                    args=(input_iu['timestamp'],
                          user_utterance,
                          self.dialogue_history,
                          last_asr_iu_id,
                          self.llm_buffer)
                )
                t.start()

                # ASR_COMMITはユーザ発話が前のシステム発話より時間的に後になる場合だけ発出
                if self.system_utterance_end_time < input_iu['timestamp']:
                    self.event_queue.put('ASR_COMMIT')
                iu_memory = []

    # 対話状態を管理
    def state_management(self):
        while True:
            event = self.event_queue.get()
            prev_state = self.state
            self.state = RemdisState.transition[self.state][event]
            self.log(f'********** State: {prev_state} -> {self.state}, Trigger: {event} **********')

            # idle状態でのみASR_COMMITやSYSTEM_TAKE_TURNで応答生成
            if prev_state == 'idle':
                if event == 'SYSTEM_BACKCHANNEL':
                    self.send_backchannel()
                if event == 'SYSTEM_TAKE_TURN':
                    self.send_response()
                if event == 'ASR_COMMIT':
                    self.send_response()

    # 表情・感情を管理
    def emo_act_management(self):
        while True:
            iu = self.emo_act_iu_buffer.get()
            # 感情または行動の送信
            expression_and_action = {}
            if 'emotion' in iu['body']:
                expression_and_action['expression'] = iu['body']['emotion']
            if 'action' in iu['body']:
                expression_and_action['action'] = iu['body']['action']
            
            if expression_and_action:
                snd_iu = self.createIU(expression_and_action, 'dialogue2', RemdisUpdateType.ADD)
                snd_iu['data_type'] = 'expression_and_action'
                self.printIU(snd_iu)
                self.publish(snd_iu, 'dialogue2')


    # システム発話を送信
    def send_response(self):
        if self.llm_buffer.empty():
            # 一瞬スリープしてそれでも応答生成中にならなければシステムから発話を開始
            time.sleep(0.1)
            if self.llm_buffer.empty():
                llm = ResponseChatGPT(self.config, self.prompts)
                t = threading.Thread(
                    target=llm.run,
                    args=(time.time(),
                            None,
                            self.dialogue_history,
                            None,
                            self.llm_buffer)
                )
                t.start()

        # 応答が生成され始めたLLMの中で一番新しい音声認識結果を使っているものを選択して送信
        selected_llm = self.llm_buffer.get()
        latest_asr_time = selected_llm.asr_time
        while not self.llm_buffer.empty():
            llm = self.llm_buffer.get()
            if llm.asr_time > latest_asr_time:
                selected_llm = llm

        # IUに分割して送信
        sys.stderr.write('Resp: Selected user utterance: %s\n' % (selected_llm.user_utterance))
        if selected_llm.response is not None:
            conc_response = ''
            for part in selected_llm.response:
                # 表情・動作を送信
                expression_and_action = {}
                if 'expression' in part and part['expression'] != 'normal':
                    expression_and_action['expression'] = part['expression']
                if 'action' in part and part['action'] != 'wait':
                    expression_and_action['action'] = part['action']
                if expression_and_action:
                    snd_iu = self.createIU(expression_and_action, 'dialogue2', RemdisUpdateType.ADD)
                    snd_iu['data_type'] = 'expression_and_action'
                    self.printIU(snd_iu)
                    self.publish(snd_iu, 'dialogue2')
                    self.output_iu_buffer.append(snd_iu)

                # 生成中に状態が変わることがあるためその確認の後，発話を送信
                if 'phrase' in part:
                    if self.state == 'talking':
                        snd_iu = self.createIU(part['phrase'], 'dialogue', RemdisUpdateType.ADD)
                        self.printIU(snd_iu)
                        self.publish(snd_iu, 'dialogue')
                        self.output_iu_buffer.append(snd_iu)
                        conc_response += part['phrase']

            # 対話コンテキストにユーザ発話を追加
            if selected_llm.user_utterance:
                self.history_management('user', selected_llm.user_utterance)
            else:
                self.history_management('user', '(沈黙)')
            self.history_management('assistant', conc_response)

        # 応答生成終了メッセージ
        sys.stderr.write('End of selected llm response. Waiting next user uttenrance.\n')
        snd_iu = self.createIU('', 'dialogue', RemdisUpdateType.COMMIT)
        self.printIU(snd_iu)
        self.publish(snd_iu, 'dialogue')

    # バックチャネルを送信
    def send_backchannel(self):
        iu = self.bc_iu_buffer.get()

        # 現在の状態がidleの場合のみ後続の処理を実行してバックチャネルを送信
        if self.state != 'idle':
            return

        # 相槌の送信
        snd_iu = self.createIU(iu['body']['bc'], 'dialogue', RemdisUpdateType.ADD)
        self.printIU(snd_iu)
        self.publish(snd_iu, 'dialogue')

    # 応答を中断
    def stop_response(self):
        for iu in self.output_iu_buffer:
            iu['update_type'] = RemdisUpdateType.REVOKE
            self.printIU(iu)
            self.publish(iu, iu['exchange'])
        self.output_iu_buffer = []

    # 音声認識結果受信用のコールバック
    def callback_asr(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        self.input_iu_buffer.put(in_msg)
            
    # 音声合成結果受信用のコールバック
    def callback_tts(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        if in_msg['update_type'] == RemdisUpdateType.COMMIT:
            self.output_iu_buffer = []
            self.system_utterance_end_time = in_msg['timestamp']
            self.event_queue.put('TTS_COMMIT')

    # VAP情報受信用のコールバック
    def callback_vap(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        self.event_queue.put(in_msg['body'])

    # バックチャネル受信用のコールバック
    def callback_bc(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        self.bc_iu_buffer.put(in_msg)
        self.event_queue.put('SYSTEM_BACKCHANNEL')

    # 表情・行動情報受信用のコールバック
    def callback_emo_act(self, ch, method, properties, in_msg):
        in_msg = self.parse_msg(in_msg)
        self.emo_act_iu_buffer.put(in_msg)

    # 対話履歴を更新
    def history_management(self, role, utt):
        self.dialogue_history.append({"role": role, "content": utt})
        if len(self.dialogue_history) > self.history_length:
            self.dialogue_history.pop(0)

    # ニュース要約を生成
    def generate_news_summary(self):
        """起動時にニュース記事の要約を生成する"""
        try:
            summary_generator = SummaryGenerator(
                config=self.config,
                prompts=self.prompts,
                news_path="../news/news.txt"
            )
            
            # 要約生成
            summary = summary_generator.generate_summary()
            
            if summary:
                # 要約をファイルに保存
                summary_generator.save_summary(summary)
                self.log(f"News summary generated and saved successfully")
            else:
                self.log("Failed to generate news summary")
                
        except Exception as e:
            self.log(f"Error in news summary generation: {e}")

    def log(self, *args, **kwargs):
        print(f"[Dialogue][{time.time():.5f}]", *args, flush=True, **kwargs)

def main():
    dialogue = Dialogue()
    dialogue.run()

if __name__ == '__main__':
    main()
