import re
from datetime import datetime
import sqlite3
from weather_api import get_weather


def init_db():
    """Создание таблицы пользователей"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            last_interaction TIMESTAMP
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
    """Получение имени пользователя из БД"""
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM users WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None



def handle_greeting(match=None):
    return "Здравствуйте! Чем могу помочь?"


def handle_farewell(match=None):
    return "До свидания! Будет приятно видеть вас снова."


def handle_weather(match):
    """Обработчик запроса погоды"""
    city = match.group(1).strip()
    return get_weather(city)


def handle_addition(match):
    """Сложение двух чисел"""
    a = float(match.group(1))
    b = float(match.group(2))
    return f"Результат: {a} + {b} = {a + b}"


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
        
        
        if user_id:
            self.name = get_user(user_id)

    def set_name(self, match):
        """Установка имени пользователя"""
        self.name = match.group(1)
        
        
        if self.user_id and self.name:
            save_user(self.user_id, self.name)
            
        return f"Приятно познакомиться, {self.name}! Я запомнил ваше имя."

    def greet(self, match=None):
        """Приветствие с учётом имени"""
        if self.name:
            return f"Здравствуйте, {self.name}! Рад вас снова видеть."
        return "Здравствуйте! Как я могу к вам обращаться?"

    



def create_patterns(bot_instance):
    """Создание списка паттернов с обработчиками"""
    return [
        
        (re.compile(r"^(привет|здравствуй|добрый день|доброе утро|добрый вечер)$", re.IGNORECASE), 
         bot_instance.greet),
        
        
        (re.compile(r"^(пока|до свидания|всего хорошего|до встречи)$", re.IGNORECASE), 
         handle_farewell),
        
       
        (re.compile(r"погода в ([а-яА-Яa-zA-Z\- ]+)", re.IGNORECASE), 
         handle_weather),
        
        
        (re.compile(r"(\d+)\s*\+\s*(\d+)"), 
         handle_addition),
        
        
        (re.compile(r"(?:меня зовут|мое имя|называй меня|я) ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), 
         bot_instance.set_name),
        
        
        
    ]


def process_message(message: str, bot_instance: ChatBot) -> str:
    """
    Обработка сообщения пользователя
    """
    message = message.strip()
    
    if not message:
        return "Вы ничего не написали. Чем могу помочь?"

    patterns = create_patterns(bot_instance)

    for pattern, handler in patterns:
        match = pattern.search(message)
        if match:
            return handler(match)

    return "Я не понимаю запрос. Напишите 'помощь', чтобы узнать, что я умею."