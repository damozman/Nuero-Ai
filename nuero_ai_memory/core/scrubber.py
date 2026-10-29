import re
import json
import os


class MemoryScrubber:
    def __init__(self, rules_path="config/masking_rules.json"):
        self.rules_path = rules_path
        # pattern -> placeholder token; all rules live in masking_rules.json
        self.mask_rules = {}
        # token -> the original text it replaced, filled in as mask() runs
        self.reverse_rules = {}
        self._load_rules()

    def _load_rules(self):
        if os.path.exists(self.rules_path):
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.mask_rules.update(json.load(f))

    def mask(self, text: str) -> str:
        masked_text = text
        for pattern, token in self.mask_rules.items():
            def _swap(match, token=token):
                self.reverse_rules.setdefault(token, match.group(0))
                return token
            masked_text = re.sub(pattern, _swap, masked_text, flags=re.IGNORECASE)
        return masked_text

    def unmask(self, text: str) -> str:
        unmasked_text = text
        for token, original in self.reverse_rules.items():
            unmasked_text = unmasked_text.replace(token, original)
        return unmasked_text
