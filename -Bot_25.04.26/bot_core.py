# bot_core.py - ОБНОВЛЕННАЯ ВЕРСИЯ с TTS (pyttsx3)
from voice_input import listen, is_whisper_available
from bert_intent import load_bert_model, predict_intent_bert
import re
from datetime import datetime
import sqlite3
from weather_api import get_weather
from enum import Enum
import json
import os
import threading
import time
from skills.skill_router import SkillRouter
from tts_manager import TTSManager

# Определение состояний FSM
class BotState(Enum):
    START = "start"                 
    WAITING_CITY = "waiting_city"    
    WAITING_DATE = "waiting_date"    
    WAITING_FIRST_NUMBER = "waiting_first_number"  
    WAITING_SECOND_NUMBER = "waiting_second_number" 

# Глобальный роутер
skill_router = SkillRouter()

# Глобальный TTS менеджер
tts_manager = None

def init_tts():
    """Инициализация TTS менеджера"""
    global tts_manager
    try:
        tts_manager = TTSManager()
        return True
    except Exception as e:
        print(f"⚠️ TTS не инициализирован: {e}")
        print("   Бот будет работать без озвучивания")
        return False

def load_bert():
    """Загрузка BERT модели"""
    return load_bert_model()

def is_bert_ready():
    """Проверка готовности BERT"""
    from bert_intent import is_bert_available
    return is_bert_available()

def init_db():
    """Инициализация базы данных"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            last_interaction TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_query TEXT,
            city TEXT,
            weather_response TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dialog_states (
            user_id INTEGER PRIMARY KEY,
            state TEXT,
            temp_data TEXT,
            last_updated TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bert_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_text TEXT,
            predicted_intent TEXT,
            confidence REAL,
            timestamp TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)

    conn.commit()
    conn.close()

def save_user(user_id, name):
    """Сохранение пользователя в БД"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO users (user_id, name, last_interaction) VALUES (?, ?, ?)",
        (user_id, name, datetime.now())
    )
    conn.commit()
    conn.close()

def get_user(user_id):
    """Получение пользователя из БД"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_weather_query(user_id, user_query, city, weather_response):
    """Сохранение запроса погоды в БД"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO weather_queries (user_id, user_query, city, weather_response, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, user_query, city, weather_response, datetime.now()))
    conn.commit()
    conn.close()

def save_bert_prediction(user_id, text, intent, confidence):
    """Сохранение BERT-предсказания в БД"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO bert_predictions (user_id, original_text, predicted_intent, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, text, intent, confidence, datetime.now()))
    conn.commit()
    conn.close()

def save_dialog_state(user_id, state, temp_data=None):
    """Сохранение состояния диалога"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    temp_data_json = json.dumps(temp_data) if temp_data else "{}"
    cursor.execute("""
        INSERT OR REPLACE INTO dialog_states (user_id, state, temp_data, last_updated)
        VALUES (?, ?, ?, ?)
    """, (user_id, state.value if isinstance(state, BotState) else state, temp_data_json, datetime.now()))
    conn.commit()
    conn.close()

def load_dialog_state(user_id):
    """Загрузка состояния диалога"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("SELECT state, temp_data FROM dialog_states WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        state_str, temp_data_json = result
        try:
            state = BotState(state_str)
        except ValueError:
            state = BotState.START
        temp_data = json.loads(temp_data_json) if temp_data_json else {}
        return state, temp_data
    return BotState.START, {}

def extract_city_from_text(text):
    """Простое извлечение города из текста"""
    text_lower = text.lower()
    common_cities = {
        'москв': 'Москва', 'спб': 'Санкт-Петербург', 'питер': 'Санкт-Петербург',
        'новосибирск': 'Новосибирск', 'екатеринбург': 'Екатеринбург',
        'казан': 'Казань', 'нижний': 'Нижний Новгород', 'челябинск': 'Челябинск',
        'омск': 'Омск', 'самар': 'Самара', 'ростов': 'Ростов-на-Дону'
    }
    for key, city in common_cities.items():
        if key in text_lower:
            return city
    return None

def extract_date_from_text(text):
    """Извлечение даты из текста"""
    text_lower = text.lower()
    if 'сегодня' in text_lower or 'сейчас' in text_lower:
        return 'сегодня'
    elif 'завтра' in text_lower:
        return 'завтра'
    elif 'послезавтра' in text_lower:
        return 'послезавтра'
    return None

def log_message(user_message, bot_response):
    """Логирование сообщений в файл"""
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] USER: {user_message}\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BOT: {bot_response}\n")
        f.write("-" * 50 + "\n")

class ChatBot:
    def __init__(self, user_id=None):
        self.name = None
        self.user_id = user_id
        self.state = BotState.START
        self.temp_data = {}
        
        if user_id:
            self.name = get_user(user_id)
            self.state, self.temp_data = load_dialog_state(user_id)
    
    def save_state(self):
        if self.user_id:
            save_dialog_state(self.user_id, self.state, self.temp_data)
    
    def reset_state(self):
        self.state = BotState.START
        self.temp_data = {}
        self.save_state()
    
    def set_name(self, name):
        self.name = name
        if self.user_id and self.name:
            save_user(self.user_id, self.name)
        self.reset_state()    
        return f"Приятно познакомиться, {self.name}! Я запомнил ваше имя. 😊"

    def greet(self):
        if self.name:
            return f"Здравствуйте, {self.name}! Рад вас снова видеть. 😊"
        return "Здравствуйте! Как я могу к вам обращаться?"

    def speak_response(self, response_text: str):
        """Асинхронное озвучивание ответа бота"""
        if tts_manager and response_text:
            threading.Thread(target=self._delayed_speak, args=(response_text,), daemon=True).start()
    
    def _delayed_speak(self, text):
        """Задержка перед озвучиванием"""
        time.sleep(0.1)
        tts_manager.speak(text, async_mode=False)

    def start_weather_dialog(self, message):
        city = extract_city_from_text(message)
        date = extract_date_from_text(message)
        
        self.temp_data = {'intent': 'weather'}
        
        if city:
            self.temp_data['city'] = city
            if date:
                weather_response = get_weather(city)
                if self.user_id:
                    save_weather_query(self.user_id, message, city, weather_response)
                self.reset_state()
                response = f"{weather_response}\n\nЧем ещё могу помочь?"
                self.speak_response(response)
                return response
            else:
                self.state = BotState.WAITING_DATE
                self.save_state()
                response = f"Город {city} принят. На какую дату нужен прогноз? (сегодня/завтра)"
                self.speak_response(response)
                return response
        else:
            self.state = BotState.WAITING_CITY
            self.save_state()
            response = "В каком городе вы хотите узнать погоду?"
            self.speak_response(response)
            return response

    def start_addition_dialog(self, message):
        numbers = re.findall(r"[-+]?\d*\.?\d+", message.replace(',', '.'))
        
        if len(numbers) >= 2:
            try:
                a = float(numbers[0])
                b = float(numbers[1])
                result = a + b
                self.reset_state()
                response = f"Результат: {a} + {b} = {result}"
                self.speak_response(response)
                return f"🧮 {response}"
            except ValueError:
                pass
        
        self.temp_data = {'intent': 'addition'}
        self.state = BotState.WAITING_FIRST_NUMBER
        self.save_state()
        response = "Введите первое число:"
        self.speak_response(response)
        return response

    def process_city(self, city_text):
        city = extract_city_from_text(city_text)
        
        if not city:
            city = city_text.strip()
        
        if not city:
            response = "Пожалуйста, укажите название города."
            self.speak_response(response)
            return response
        
        self.temp_data['city'] = city
        
        date = extract_date_from_text(city_text)
        if date:
            weather_response = get_weather(city)
            if self.user_id:
                save_weather_query(self.user_id, city_text, city, weather_response)
            self.reset_state()
            response = f"{weather_response}\n\nЧем ещё могу помочь?"
            self.speak_response(response)
            return response
        
        self.state = BotState.WAITING_DATE
        self.save_state()
        response = f"Город {city} принят. На какую дату нужен прогноз? (сегодня/завтра)"
        self.speak_response(response)
        return response

    def process_date(self, date_text):
        date = extract_date_from_text(date_text)
        if not date:
            date = 'сегодня'
        
        self.temp_data['date'] = date
        city = self.temp_data.get('city')
        
        weather_response = get_weather(city)
        
        if self.user_id:
            save_weather_query(self.user_id, f"погода в {city} на {date}", city, weather_response)
        
        self.reset_state()
        response = f"{weather_response}\n\nЧем ещё могу помочь?"
        self.speak_response(response)
        return response

    def process_first_number(self, number_text):
        try:
            numbers = re.findall(r"[-+]?\d*\.?\d+", number_text.replace(',', '.'))
            if numbers:
                number = float(numbers[0])
            else:
                number = float(number_text.replace(',', '.'))
            
            self.temp_data['first_number'] = number
            
            if len(numbers) >= 2:
                second = float(numbers[1])
                result = number + second
                self.reset_state()
                response = f"Результат: {number} + {second} = {result}"
                self.speak_response(response)
                return f"🧮 {response}"
            
            self.state = BotState.WAITING_SECOND_NUMBER
            self.save_state()
            response = f"Первое число: {number}. Введите второе число:"
            self.speak_response(response)
            return response
        except ValueError:
            response = "Пожалуйста, введите корректное число."
            self.speak_response(response)
            return response

    def process_second_number(self, number_text):
        try:
            numbers = re.findall(r"[-+]?\d*\.?\d+", number_text.replace(',', '.'))
            if numbers:
                second = float(numbers[0])
            else:
                second = float(number_text.replace(',', '.'))
            
            first = self.temp_data.get('first_number', 0)
            result = first + second
            
            self.reset_state()
            response = f"Результат: {first} + {second} = {result}"
            self.speak_response(response)
            return f"🧮 {response}"
        except ValueError:
            response = "Пожалуйста, введите корректное число."
            self.speak_response(response)
            return response

def process_message(message: str, bot_instance: ChatBot) -> str:
    """Обработка сообщения пользователя с BERT-классификацией и Skill Router"""
    message = message.strip()
    
    if not message:
        return "Вы ничего не написали. Чем могу помочь?"

    try:
        # Обработка в зависимости от текущего состояния FSM
        if bot_instance.state == BotState.WAITING_CITY:
            return bot_instance.process_city(message)
        
        elif bot_instance.state == BotState.WAITING_DATE:
            return bot_instance.process_date(message)
        
        elif bot_instance.state == BotState.WAITING_FIRST_NUMBER:
            return bot_instance.process_first_number(message)
        
        elif bot_instance.state == BotState.WAITING_SECOND_NUMBER:
            return bot_instance.process_second_number(message)
        
        # Используем BERT для определения интента
        intent, confidence = predict_intent_bert(message)
        
        if not intent:
            return "Извините, BERT модель не загружена. Пожалуйста, сначала обучите модель."
        
        print(f"[BERT] Интент: {intent}, уверенность: {confidence:.2%}")
        
        # Сохраняем предсказание в БД
        if bot_instance.user_id and intent:
            save_bert_prediction(bot_instance.user_id, message, intent, confidence)
        
        # Проверяем уверенность модели
        if confidence < 0.25:
            response = "Извините, я не уверен, что правильно понял ваш запрос. Пожалуйста, переформулируйте."
            bot_instance.speak_response(response)
            return response
        
        # Используем Skill Router для обработки интента
        response = skill_router.route(intent, message, bot_instance=bot_instance)
        
        # Озвучиваем ответ (кроме прощания - обработаем отдельно в main)
        if intent != "goodbye":
            bot_instance.speak_response(response)
        
        return response
               
    except Exception as e:
        print(f"Ошибка: {e}")
        response = "Произошла ошибка. Попробуйте ещё раз или начните новый диалог."
        bot_instance.speak_response(response)
        return response