from datetime import datetime
from .base_skill import BaseSkill

class DateSkill(BaseSkill):
    """Навык показа текущей даты"""
    
    def __init__(self):
        super().__init__(
            name="date",
            description="Показывает текущую дату"
        )
    
    def execute(self, text: str, bot_instance=None, **kwargs) -> str:
        now = datetime.now()
        weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        weekday = weekdays[now.weekday()]
        return f"📅 Сегодня {now.strftime('%d.%m.%Y')}, {weekday}"