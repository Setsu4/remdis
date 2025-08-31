import os
import openai

class NewsSummaryGenerator:
    def __init__(self, config):
        self.config = config
        self.prompt_path = config['ChatGPT']['prompts']['NEWS_SUMMARY']
        self.news_path = os.path.join(os.path.dirname(self.prompt_path), '../news/news.txt')
        self.summary_path = os.path.join(os.path.dirname(self.prompt_path), '../news/summary.txt')
        self.api_key = config['ChatGPT']['api_key']
        self.model = config['ChatGPT']['response_generation_model']
        self.max_tokens = config['ChatGPT'].get('summary_max_tokens', 1024)
        openai.api_key = self.api_key

    def generate_summary(self):
        # プロンプトとニュース記事の読み込み
        with open(self.prompt_path, encoding='utf-8') as f:
            prompt = f.read()
        with open(self.news_path, encoding='utf-8') as f:
            news = f.read()
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": news}
        ]
        # ChatGPT APIで要約生成
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens
        )
        summary = response['choices'][0]['message']['content']
        # 要約結果を保存
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        return summary
