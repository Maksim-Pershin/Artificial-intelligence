import re
from .base_skill import BaseSkill

class SetNameSkill(BaseSkill):
    """Навык запоминания имени пользователя"""
    
    def __init__(self):
        super().__init__(
            name="set_name",
            description="Запоминает имя пользователя"
        )
    
    def extract_name(self, text: str) -> str:
        """Извлечение имени из текста"""
        patterns = [
            r"(?:меня зовут|меня звать|мое имя|называй меня|зови меня|я)\s+([а-яА-Яa-zA-Z]+)",
            r"([а-яА-Яa-zA-Z]+)\s+(?:меня зовут|меня звать|мое имя)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        return None
    
    def execute(self, text: str, bot_instance=None, **kwargs) -> str:
        name = self.extract_name(text)
        
        if name and bot_instance:
            return bot_instance.set_name(name)
        elif bot_instance:
            return "Как вас зовут? Скажите, например: 'Меня зовут Анна'"
        return "Пожалуйста, представьтесь."