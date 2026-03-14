import re
from datetime import datetime
import sqlite3
from weather_api import get_weather
import spacy
from enum import Enum
import json

nlp = spacy.load("ru_core_news_sm")

# Определение состояний FSM
class BotState(Enum):
    START = "start"                 
    WAITING_CITY = "waiting_city"    
    WAITING_DATE = "waiting_date"    
    WAITING_FIRST_NUMBER = "waiting_first_number"  
    WAITING_SECOND_NUMBER = "waiting_second_number" 

def init_db():
   
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    # Таблица пользователей
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            last_interaction TIMESTAMP
        )
    """)

    # Таблица для запросов погоды
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
    
    # Таблица для хранения анализа текста
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS text_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            original_text TEXT,
            tokens TEXT,
            lemmas TEXT,
            pos_tags TEXT,
            entities TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    
    # Таблица для хранения состояния диалога
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dialog_states (
            user_id INTEGER PRIMARY KEY,
            state TEXT,
            temp_data TEXT,
            last_updated TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)

    conn.commit()
    conn.close()

def save_user(user_id, name):
    
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR REPLACE INTO users (user_id, name, last_interaction) VALUES (?, ?, ?)",
        (user_id, name, datetime.now())
    )

    conn.commit()
    conn.close()

def get_user(user_id):
    
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM users WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None

def save_weather_query(user_id, user_query, city, weather_response):
    
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO weather_queries (user_id, user_query, city, weather_response, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, user_query, city, weather_response, datetime.now()))

    conn.commit()
    conn.close()

def save_text_analysis(user_id, text, analysis):
    
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO text_analysis 
        (user_id, original_text, tokens, lemmas, pos_tags, entities, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, 
        text,
        ', '.join(analysis['tokens']),
        ', '.join(analysis['lemmas']),
        ', '.join(analysis['pos_tags']),
        ', '.join([f"{ent}({label})" for ent, label in analysis['entities']]),
        datetime.now()
    ))

    conn.commit()
    conn.close()

def save_dialog_state(user_id, state, temp_data=None):
    
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

def extract_city_with_spacy(text):
    
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.lemma_
    
    return None

def extract_date_with_spacy(text):
    
    text_lower = text.lower()
    
    if 'сегодня' in text_lower or 'сейчас' in text_lower:
        return 'сегодня'
    elif 'завтра' in text_lower:
        return 'завтра'
    elif 'послезавтра' in text_lower:
        return 'послезавтра'
    
    return None

def is_weather_query_with_spacy(text):
    
    text_lower = text.lower()
    
    # Ключевые слова о погоде на русском
    weather_keywords = [
        "погод", "температур", "прогноз", "жарк", "холодн",
        "дожд", "снег", "ветер", "солнечн", "облачн", "град",
        "тепл", "мороз", "осадк", "метео"
    ]
    
    # проверка на наличие ключевых слов
    for keyword in weather_keywords:
        if keyword in text_lower:
            return True
    
    return False

def analyze_text_with_spacy(text):
    """Анализ текста с помощью spaCy"""
    doc = nlp(text)
    
    analysis = {
        "original": text,
        "tokens": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "pos_tags": [token.pos_ for token in doc],
        "entities": [(ent.text, ent.label_) for ent in doc.ents]
    }
    
    return analysis

def handle_greeting(match=None):
    return "Здравствуйте! Чем могу помочь?"

def handle_farewell(match=None):
    return "До свидания! Будет приятно видеть вас снова."

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
            # Загружаем сохраненное состояние
            self.state, self.temp_data = load_dialog_state(user_id)
    
    def save_state(self):
        """Сохраняет текущее состояние в БД"""
        if self.user_id:
            save_dialog_state(self.user_id, self.state, self.temp_data)
    
    def reset_state(self):
        """Сброс состояния после завершения диалога"""
        self.state = BotState.START
        self.temp_data = {}
        self.save_state()
    
    def set_name(self, match):
        """Установка имени пользователя"""
        name = match.group(1)
        self.name = name
        
        if self.user_id and self.name:
            save_user(self.user_id, self.name)
        
        self.reset_state()    
        return f"Приятно познакомиться, {self.name}! Я запомнил ваше имя."

    def greet(self, match=None):
        """Приветствие с учётом имени"""
        if self.name:
            return f"Здравствуйте, {self.name}! Рад вас снова видеть."
        return "Здравствуйте! Как я могу к вам обращаться?"

    def start_weather_dialog(self, message):
        
        # Пытаемся извлечь город из сообщения
        city = extract_city_with_spacy(message)
        date = extract_date_with_spacy(message)
        
        self.temp_data = {'intent': 'weather'}
        
        if city:
            self.temp_data['city'] = city
            if date:
                # Есть и город, и дата - сразу показываем погоду
                weather_response = get_weather(city)
                if self.user_id:
                    save_weather_query(self.user_id, message, city, weather_response)
                self.reset_state()
                return f"{weather_response}\n\nЧем ещё могу помочь?"
            else:
                # Есть город, но нет даты - спрашиваем дату
                self.state = BotState.WAITING_DATE
                self.save_state()
                return f"Город {city} принят. На какую дату нужен прогноз? (сегодня/завтра)"
        else:
            # Нет города - спрашиваем город
            self.state = BotState.WAITING_CITY
            self.save_state()
            return "В каком городе вы хотите узнать погоду?"

    def start_addition_dialog(self, message):
        
        # Пытаемся найти числа в сообщении
        numbers = re.findall(r"[-+]?\d*\.?\d+", message.replace(',', '.'))
        
        if len(numbers) >= 2:
            # Нашли два числа - сразу складываем
            try:
                a = float(numbers[0])
                b = float(numbers[1])
                result = a + b
                self.reset_state()
                return f"Результат: {a} + {b} = {result}"
            except ValueError:
                pass
        
        self.temp_data = {'intent': 'addition'}
        self.state = BotState.WAITING_FIRST_NUMBER
        self.save_state()
        return "Введите первое число:"

    def process_city(self, city_text):
        """Обрабатывает ввод города"""
        # Извлекаем город через spaCy
        city = extract_city_with_spacy(city_text)
        
        if not city:
            # Если не удалось извлечь, пробуем просто взять текст
            city = city_text.strip()
        
        if not city:
            return "Пожалуйста, укажите название города."
        
        self.temp_data['city'] = city
        
        # Проверяем, не указана ли дата в этом же сообщении
        date = extract_date_with_spacy(city_text)
        if date:
            # Есть и город, и дата - сразу показываем погоду
            weather_response = get_weather(city)
            if self.user_id:
                save_weather_query(self.user_id, city_text, city, weather_response)
            self.reset_state()
            return f"{weather_response}\n\nЧем ещё могу помочь?"
        
        self.state = BotState.WAITING_DATE
        self.save_state()
        return f"Город {city} принят. На какую дату нужен прогноз? (сегодня/завтра)"

    def process_date(self, date_text):
        """Обрабатывает ввод даты"""
        date = extract_date_with_spacy(date_text)
        
        if not date:
            date = 'сегодня'  # По умолчанию
        
        self.temp_data['date'] = date
        city = self.temp_data.get('city')
        
        # Получаем погоду сразу
        weather_response = get_weather(city)
        
        # Сохраняем в БД
        if self.user_id:
            save_weather_query(self.user_id, f"погода в {city} на {date}", city, weather_response)
        
        self.reset_state()
        return f"{weather_response}\n\nЧем ещё могу помочь?"

    def process_first_number(self, number_text):
        """Обрабатывает ввод первого числа"""
        try:
            # Пробуем найти число в тексте
            numbers = re.findall(r"[-+]?\d*\.?\d+", number_text.replace(',', '.'))
            if numbers:
                number = float(numbers[0])
            else:
                number = float(number_text.replace(',', '.'))
            
            self.temp_data['first_number'] = number
            
            # Проверяем, есть ли второе число в этом же сообщении
            if len(numbers) >= 2:
                second = float(numbers[1])
                result = number + second
                self.reset_state()
                return f"Результат: {number} + {second} = {result}"
            
            self.state = BotState.WAITING_SECOND_NUMBER
            self.save_state()
            return f"Первое число: {number}. Введите второе число:"
        except ValueError:
            return "Пожалуйста, введите корректное число."

    def process_second_number(self, number_text):
        """Обрабатывает ввод второго числа и выполняет сложение"""
        try:
            # Пробуем найти число в тексте
            numbers = re.findall(r"[-+]?\d*\.?\d+", number_text.replace(',', '.'))
            if numbers:
                second = float(numbers[0])
            else:
                second = float(number_text.replace(',', '.'))
            
            first = self.temp_data.get('first_number', 0)
            result = first + second
            
            self.reset_state()
            return f"Результат: {first} + {second} = {result}"
        except ValueError:
            return "Пожалуйста, введите корректное число."

def create_patterns(bot_instance):
    """Создание списка паттернов с обработчиками"""
    return [
        # Приветствия
        (re.compile(r"^(привет|здравствуй|добрый день|доброе утро|добрый вечер|хай|здарова)$", re.IGNORECASE), 
         bot_instance.greet),
        
        # Прощания
        (re.compile(r"^(пока|до свидания|всего хорошего|до встречи|увидимся)$", re.IGNORECASE), 
         handle_farewell),
        
        # Запрос погоды - запуск диалога с анализом сообщения
        (re.compile(r".*(?:погода|температура|прогноз|холодно|жарко|дождь|снег).*", re.IGNORECASE), 
         lambda m: bot_instance.start_weather_dialog(m.string)),
        
        # Сложение - запуск диалога с анализом сообщения
        (re.compile(r"(?:сложи|плюс|сумма|прибавь|сложение|\+)"), 
         lambda m: bot_instance.start_addition_dialog(m.string)),
        
        # Установка имени
        (re.compile(r"(?:меня зовут|мое имя|называй меня|я) ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), 
         bot_instance.set_name),
    ]

def process_message(message: str, bot_instance: ChatBot) -> str:
    """
    Обработка сообщения пользователя с поддержкой FSM
    """
    message = message.strip()
    
    if not message:
        return "Вы ничего не написали. Чем могу помочь?"

    try:
        # Анализируем текст с помощью spaCy
        analysis = analyze_text_with_spacy(message)
        
        # Сохраняем анализ в базу данных
        if bot_instance.user_id:
            save_text_analysis(bot_instance.user_id, message, analysis)

        # Обработка в зависимости от текущего состояния FSM
        if bot_instance.state == BotState.WAITING_CITY:
            return bot_instance.process_city(message)
        
        elif bot_instance.state == BotState.WAITING_DATE:
            return bot_instance.process_date(message)
        
        elif bot_instance.state == BotState.WAITING_FIRST_NUMBER:
            return bot_instance.process_first_number(message)
        
        elif bot_instance.state == BotState.WAITING_SECOND_NUMBER:
            return bot_instance.process_second_number(message)
        
        # Если нет активного состояния, ищем подходящий паттерн
        patterns = create_patterns(bot_instance)

        for pattern, handler in patterns:
            match = pattern.search(message)
            if match:
                return handler(match)

        # Если ничего не подошло
        return ("Я не понимаю запрос.")
               
    
    except Exception as e:
        print(f"Ошибка: {e}")
        return "Произошла ошибка. Попробуйте ещё раз или начните новый диалог."