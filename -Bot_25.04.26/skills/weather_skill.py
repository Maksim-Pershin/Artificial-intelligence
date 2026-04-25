from .base_skill import BaseSkill
from weather_api import get_weather
import re

class WeatherSkill(BaseSkill):
    """Навык получения погоды"""
    
    def __init__(self):
        super().__init__(
            name="weather",
            description="Показывает погоду в городе"
        )
    
    def extract_city(self, text: str) -> str:
        """Извлечение города из текста"""
        text_lower = text.lower()
        common_cities = {
            'москв': 'Москва',
            'спб': 'Санкт-Петербург',
            'питер': 'Санкт-Петербург',
            'новосибирск': 'Новосибирск',
            'екатеринбург': 'Екатеринбург',
            'казан': 'Казань',
            'нижний': 'Нижний Новгород',
            'челябинск': 'Челябинск',
            'омск': 'Омск',
            'самар': 'Самара',
            'ростов': 'Ростов-на-Дону'
        }
        
        for key, city in common_cities.items():
            if key in text_lower:
                return city
        return None
    
    def execute(self, text: str, bot_instance=None, **kwargs) -> str:
        city = self.extract_city(text)
        
        if city:
            weather_response = get_weather(city)
            return weather_response
        else:
            # Запускаем диалог для уточнения города
            if bot_instance:
                bot_instance.temp_data = {'intent': 'weather'}
                bot_instance.state = bot_instance.state.WAITING_CITY
                bot_instance.save_state()
                return "В каком городе вы хотите узнать погоду?"
            return "Пожалуйста, укажите город для прогноза погоды."