import json
from typing import Dict, Any


class PlanGenerator:
    def __init__(self, news_text_path: str | None = None):
        self.news_text_path = news_text_path

    def generate_from_example(self, example_json_path: str) -> Dict[str, Any]:
        with open(example_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # future work: parse self.news_text_path and generate plan
