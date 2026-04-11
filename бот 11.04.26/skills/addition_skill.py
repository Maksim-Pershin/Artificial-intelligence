import re
from .base_skill import BaseSkill

class AdditionSkill(BaseSkill):
    """Навык сложения чисел"""
    
    def __init__(self):
        super().__init__(
            name="addition",
            description="Складывает два числа"
        )
    
    def extract_numbers(self, text: str):
        """Извлечение чисел из текста"""
        numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(',', '.'))
        return [float(n) for n in numbers if n]
    
    def execute(self, text: str, bot_instance=None, **kwargs) -> str:
        numbers = self.extract_numbers(text)
        
        if len(numbers) >= 2:
            result = numbers[0] + numbers[1]
            return f"🧮 Результат: {numbers[0]} + {numbers[1]} = {result}"
        else:
            # Запускаем диалог для ввода чисел
            if bot_instance:
                bot_instance.temp_data = {'intent': 'addition'}
                bot_instance.state = bot_instance.state.WAITING_FIRST_NUMBER
                bot_instance.save_state()
                return "Введите первое число:"
            return "Пожалуйста, введите два числа для сложения."