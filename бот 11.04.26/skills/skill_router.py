from typing import Dict, Optional
from .base_skill import BaseSkill
from .time_skill import TimeSkill
from .date_skill import DateSkill
from .greeting_skill import GreetingSkill
from .goodbye_skill import GoodbyeSkill
from .weather_skill import WeatherSkill
from .addition_skill import AdditionSkill
from .set_name_skill import SetNameSkill
from .smalltalk_skill import SmallTalkSkill
from .help_skill import HelpSkill

class SkillRouter:
    """Маршрутизатор навыков на основе интента"""
    
    def __init__(self):
        self.skills: Dict[str, BaseSkill] = {}
        self._register_skills()
        self.last_intents = {}  # Для контекстного follow_up (если понадобится)
    
    def _register_skills(self):
        """Регистрация всех доступных навыков"""
        self.skills["time"] = TimeSkill()
        self.skills["date"] = DateSkill()
        self.skills["greeting"] = GreetingSkill()
        self.skills["goodbye"] = GoodbyeSkill()
        self.skills["weather"] = WeatherSkill()
        self.skills["addition"] = AdditionSkill()
        self.skills["set_name"] = SetNameSkill()
        self.skills["smalltalk"] = SmallTalkSkill()
        self.skills["help"] = HelpSkill()
    
    def route(self, intent: str, text: str, bot_instance=None, **kwargs) -> str:
        """
        Маршрутизация запроса к соответствующему навыку
        
        Args:
            intent: Определённый интент
            text: Исходный текст пользователя
            bot_instance: Экземпляр ChatBot для доступа к состоянию
            **kwargs: Дополнительные параметры
        
        Returns:
            Ответ от навыка
        """
        # Сохраняем последний интент для контекста
        if bot_instance and bot_instance.user_id:
            self.last_intents[bot_instance.user_id] = intent
        
        # Если навык существует, выполняем его
        if intent in self.skills:
            return self.skills[intent].execute(text, bot_instance=bot_instance, **kwargs)
        
        # Если интент не распознан или unknown
        return self._fallback()
    
    def _fallback(self) -> str:
        """Ответ по умолчанию, если интент не распознан"""
        return ("Я не совсем понял ваш запрос. 🤔\n\n"
                "Я могу:\n"
                "🌤️ Рассказать о погоде\n"
                "🕐 Показать время\n"
                "📅 Показать дату\n"
                "➕ Сложить числа\n"
                "👤 Запомнить ваше имя\n"
                "💬 Поболтать\n\n"
                "Скажите 'что ты умеешь' для полного списка команд!")
    
    def get_skill_description(self, skill_name: str) -> Optional[str]:
        """Получить описание навыка"""
        if skill_name in self.skills:
            return self.skills[skill_name].get_description()
        return None
    
    def list_all_skills(self) -> Dict[str, str]:
        """Список всех навыков с описаниями"""
        return {name: skill.description for name, skill in self.skills.items()}