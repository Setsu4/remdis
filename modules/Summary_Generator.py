import os
import time
import openai
from base import RemdisModule
from prompt import util as prompt_util


class SummaryGenerator:
    def __init__(self, config, prompts, news_path=None):
        self.config = config
        self.prompts = prompts
        
        # OpenAI APIキーの設定
        openai.api_key = config['ChatGPT']['api_key']
        self.model = config['ChatGPT']['response_generation_model']
        self.max_tokens = config['ChatGPT']['max_tokens']
        # summary_max_tokens（要約生成専用）を追加
        self.summary_max_tokens = config['ChatGPT'].get('summary_max_tokens', self.max_tokens)
        
        # ニュース記事の内容を読み込み
        self.news_content = ""
        if news_path is not None:
            try:
                with open(news_path, 'r', encoding='utf-8') as f:
                    self.news_content = f.read().strip()
                self.log(f"News content loaded from: {news_path}")
            except FileNotFoundError:
                self.log(f"News file not found: {news_path}")
                self.news_content = ""
        
        # 要約プロンプトの内容を読み込み
        self.summary_prompt = ""
        if 'NEWS_SUMMARY' in prompts:
            self.summary_prompt = prompts['NEWS_SUMMARY']
        else:
            self.log("Warning: NEWS_SUMMARY prompt not found in config")

    def generate_summary(self):
        """ニュース記事の要約を生成"""
        if not self.news_content:
            self.log("No news content available for summarization")
            return ""
        
        if not self.summary_prompt:
            self.log("No summary prompt available")
            return ""
        
        try:
            # ChatGPTに入力するメッセージ
            messages = [
                {'role': 'system', 'content': self.summary_prompt},
                {'role': 'user', 'content': self.news_content}
            ]
            
            self.log("Generating news summary with ChatGPT...")
            
            # ChatGPTで要約生成（summary_max_tokensを利用）
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=self.summary_max_tokens,
                temperature=0.7
            )
            
            summary = response.choices[0].message.content.strip()
            self.log(f"News summary generated: {summary}")
            
            return summary
            
        except Exception as e:
            self.log(f"Error generating summary: {e}")
            return ""

    def save_summary(self, summary, output_path="../news/summary.txt"):
        """生成された要約をファイルに保存"""
        try:
            # newsディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            self.log(f"Summary saved to: {output_path}")
            return True
            
        except Exception as e:
            self.log(f"Error saving summary: {e}")
            return False

    def load_and_split_summary(self, summary_path="../news/summary.txt"): 
        """要約ファイルを読み込み、「。」で分割しリストで返す"""
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read().strip()
            # 「。」で分割し、空要素を除去
            sentences = [s for s in summary.split("。") if s.strip()]
            return sentences
        except Exception as e:
            self.log(f"Error loading/splitting summary: {e}")
            return []

    def log(self, *args, **kwargs):
        """デバッグ用のログ出力"""
        print(f"[SummaryGenerator][{time.time():.5f}]", *args, flush=True, **kwargs)


def main():
    """テスト用のメイン関数"""
    # 設定ファイルの読み込み（テスト用）
    import yaml
    
    config_path = "../config/config copy.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # プロンプトの読み込み
    prompts = prompt_util.load_prompts(config['ChatGPT']['prompts'])
    
    # NEWS_SUMMARYプロンプトを追加（テスト用）
    try:
        with open("../prompt/news_summary.txt", 'r', encoding='utf-8') as f:
            prompts['NEWS_SUMMARY'] = f.read().strip()
    except FileNotFoundError:
        print("Warning: news_summary.txt not found")
        prompts['NEWS_SUMMARY'] = "以下のニュース記事を簡潔に要約してください。"
    
    # SummaryGeneratorのインスタンス作成
    generator = SummaryGenerator(
        config=config,
        prompts=prompts,
        news_path="../news/news.txt"
    )
    
    # 要約生成
    summary = generator.generate_summary()
    
    if summary:
        # 要約をファイルに保存
        generator.save_summary(summary)
        print(f"\nGenerated Summary:\n{summary}")
    else:
        print("Failed to generate summary")


if __name__ == '__main__':
    main()
