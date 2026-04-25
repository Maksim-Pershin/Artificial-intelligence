from .base_skill import BaseSkill

class GoodbyeSkill(BaseSkill):
    """Навык прощания"""
    
    def __init__(self):
        super().__init__(
            name="goodbye",
            description="Прощается с пользователем"
        )
    
    def execute(self, text: str, bot_instance=None, **kwargs) -> str:
        return "До свидания! Буду рад видеть вас снова! 👋"