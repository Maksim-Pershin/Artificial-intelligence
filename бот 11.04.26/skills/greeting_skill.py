from .base_skill import BaseSkill

class GreetingSkill(BaseSkill):
    """Навык приветствия"""
    
    def __init__(self):
        super().__init__(
            name="greeting",
            description="Приветствует пользователя"
        )
    
    def execute(self, text: str, bot_instance=None, **kwargs) -> str:
        if bot_instance and bot_instance.name:
            return f"Здравствуйте, {bot_instance.name}! Рад вас снова видеть. 😊"
        return "Здравствуйте! Как я могу к вам обращаться? 😊"