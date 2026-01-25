# Prompt Tuning Harness

OpenAI API を使った既存システムの呼び出し形（プロンプト -> `OK` -> ユーザ発話）をそのまま再現し、ニュース要約やニュース性の抽出、Wikipedia で調べる主題候補の整理といったプロンプトを試すためのサンプルです。

## 使い方
- `config/prompt_tuning.example.yaml` を `config/prompt_tuning.yaml` にコピーし、`OPENAI_API_KEY` 環境変数を設定するか `ChatGPT.api_key` に鍵を入れてください。
- 使用するプロンプトは `prompt/` で編集できます（`news_summary.txt`, `news_reason.txt`, `wiki_topic_summary.txt`）。`scripts/prompt_tuning.py --list-tasks` で定義済みのキーを確認できます。
- 実行例:
  - `python scripts/prompt_tuning.py --task summary --file news/news.txt`
  - `python scripts/prompt_tuning.py --task news_reason --text "記事本文をここに入れる"`
  - `python scripts/prompt_tuning.py --task wiki_topic --file news/news.txt --no-stream`
- 以前の対話履歴を渡したい場合は `--history history.json`（`[{"role": "user", "content": "..."}, ...]` 形式）を指定してください。

## 主なパラメータ（`config/prompt_tuning.yaml`）
- `model`: 利用するモデル名。ここでは `gpt-4o-mini-2024-07-18`。
- `max_tokens`: 生成長の上限。大きいと要約が途切れにくいがコストとレイテンシが増える。
- `max_message_num_in_context`: 過去履歴を何件まで入れるか。入れ過ぎると脱線しやすく、短すぎると文脈が切れる。
- `temperature`: 出力の多様性。低いと決定的に、高いと遊びが増えて表現が多彩になる。
- `top_p`: nucleus sampling の範囲。1.0 で無効。低めにすると高確率トークンに絞り安全寄りになる。
- `presence_penalty`: 新しい話題を出す圧。大きいほど同じ内容の繰り返しを避けやすい。
- `frequency_penalty`: 同じ語の繰り返し抑制。要約の語句重複が気になる場合に上げる。
- `stop`: 生成停止トークンのリスト。不要なら空配列。セクション区切りや記号で出力を切りたいときに使う。
