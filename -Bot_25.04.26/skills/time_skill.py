from datetime import datetime
from .base_skill import BaseSkill

class TimeSkill(BaseSkill):
    """Навык показа текущего времени"""
    
    def __init__(self):
        super().__init__(
            name="time",
            description="Показывает текущее время"
        )
    
    def execute(self, text: str, bot_instance=None, **kwargs) -> str:
        now = datetime.now()
        return f"🕐 Сейчас {now.strftime('%H:%M')}"