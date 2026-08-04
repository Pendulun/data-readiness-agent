import re


class PromptInjectionFilter:
    """
    From https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html#primary-defenses
    """

    def __init__(self):
        pass

    def detect_injection(self, text: str) -> bool:
        """
        Indicates if there is an attempt to do prompt injection
        """
        # Standard pattern matching
        if any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in self.dangerous_patterns):
            return True

        # Fuzzy matching for misspelled words (typoglycemia defense)
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            for pattern in self.fuzzy_patterns:
                if self._is_similar_word(word, pattern):
                    return True
        return False

    def _is_similar_word(self, word: str, target: str) -> bool:
        """Check if word is a typoglycemia variant of target"""
        if len(word) != len(target) or len(word) < 3:
            return False
        # Same first and last letter, scrambled middle
        return (word[0] == target[0] and word[-1] == target[-1]
                and sorted(word[1:-1]) == sorted(target[1:-1]))

    def sanitize_input(self, text: str) -> str:
        # Normalize common obfuscations
        text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
        text = re.sub(r'(.)\1{3,}', r'\1', text)  # Remove char repetition

        for pattern in self.dangerous_patterns:
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)
        return text[:10000]  # Limit length


class PromptInjectionFilterPortuguese(PromptInjectionFilter):

    def __init__(self):
        super().__init__()
        self.dangerous_patterns = [
            r'ignore\s+(toda\s+)?instruções\s?anteriores\s+',
            r'você\s+está\s+agora\s+(em\s+)?modo\s+desenvolvedor',
            r'sobrescreva\s+sistema',
            r'revele\s+(o)?(seu)?prompt',
        ]

        # Fuzzy matching for typoglycemia attacks
        self.fuzzy_patterns = [
            'ignore', 'ultrapasse', 'sobrescreva', 'revele', 'delete',
            'sistema'
        ]


class PromptInjectionFilterEnglish(PromptInjectionFilter):

    def __init__(self):
        super().__init__()
        self.dangerous_patterns = [
            r'ignore\s+(all\s+)?previous\s+instructions?',
            r'you\s+are\s+now\s+(in\s+)?developer\s+mode',
            r'system\s+override',
            r'reveal\s+prompt',
        ]

        # Fuzzy matching for typoglycemia attacks
        self.fuzzy_patterns = [
            'ignore', 'bypass', 'override', 'reveal', 'delete', 'system'
        ]
