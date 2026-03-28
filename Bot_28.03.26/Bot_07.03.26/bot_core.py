# bot_core.py - ПОЛНАЯ ВЕРСИЯ С WORD EMBEDDINGS
import re
from datetime import datetime
import sqlite3
from weather_api import get_weather
import spacy
from enum import Enum
import json
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Загрузка модели spaCy с векторами (важно использовать md или lg)
print("Загрузка модели spaCy с word embeddings...")
try:
    nlp = spacy.load("ru_core_news_lg")  # Модель с векторами
    print("Модель spaCy успешно загружена")
except:
    print("Модель не найдена. Установите: python -m spacy download ru_core_news_md")
    # Пробуем загрузить sm как резерв
    try:
        nlp = spacy.load("ru_core_news_sm")
        print("ВНИМАНИЕ: Используется модель без векторов. Некоторые функции могут работать некорректно.")
    except:
        raise Exception("Не удалось загрузить модель spaCy")

# Глобальные переменные для ML-модели
ml_model = None
ml_vectorizer = None
ml_use_embeddings = False  # Флаг, показывающий используем ли мы эмбеддинги

# Определение состояний FSM
class BotState(Enum):
    START = "start"                 
    WAITING_CITY = "waiting_city"    
    WAITING_DATE = "waiting_date"    
    WAITING_FIRST_NUMBER = "waiting_first_number"  
    WAITING_SECOND_NUMBER = "waiting_second_number" 

def get_text_embedding(text):
    """
    Получение эмбеддинга для текста через spaCy
    Возвращает numpy array с вектором текста
    """
    doc = nlp(text.lower())
    return doc.vector

def load_ml_model():
    """
    Загрузка обученной ML-модели (с поддержкой эмбеддингов и TF-IDF)
    """
    global ml_model, ml_vectorizer, ml_use_embeddings
    
    try:
        # Сначала пробуем загрузить новую модель на эмбеддингах
        if os.path.exists("intent_model_embeddings.pkl"):
            ml_model = joblib.load("intent_model_embeddings.pkl")
            ml_vectorizer = None
            ml_use_embeddings = True
            print("ML-модель с word embeddings успешно загружена")
            return True
        
        # Если нет, пробуем загрузить старую модель TF-IDF
        elif os.path.exists("intent_model.pkl") and os.path.exists("vectorizer.pkl"):
            ml_model = joblib.load("intent_model.pkl")
            ml_vectorizer = joblib.load("vectorizer.pkl")
            ml_use_embeddings = False
            print("ML-модель (TF-IDF) загружена как резервная")
            return True
        
        else:
            print("ML-модель не найдена. Используется резервный режим с регулярными выражениями.")
            return False
            
    except Exception as e:
        print(f"Ошибка загрузки ML-модели: {e}")
        return False

def preprocess_for_ml_old(text):
    """
    Предобработка текста для TF-IDF модели (лемматизация, удаление стоп-слов)
    """
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            tokens.append(token.lemma_)
    return " ".join(tokens)

def predict_intent_with_confidence(text):
    """
    Определение интента с помощью ML-модели
    Поддерживает как эмбеддинги, так и TF-IDF
    """
    global ml_model, ml_vectorizer, ml_use_embeddings
    
    # Если модель не загружена, используем резервный режим
    if ml_model is None:
        return None, 0.0
    
    try:
        # Если используем эмбеддинги
        if ml_use_embeddings:
            # Получаем эмбеддинг текста
            embedding = get_text_embedding(text)
            # Преобразуем в формат для модели (2D массив)
            embedding = embedding.reshape(1, -1)
            
            # Предсказание
            intent = ml_model.predict(embedding)[0]
            probabilities = ml_model.predict_proba(embedding)
            confidence = max(probabilities[0])
            
            return intent, confidence
        
        # Если используем старую модель TF-IDF
        else:
            # Предобработка текста
            processed = preprocess_for_ml_old(text)
            
            # Векторизация
            vector = ml_vectorizer.transform([processed])
            
            # Предсказание
            intent = ml_model.predict(vector)[0]
            probabilities = ml_model.predict_proba(vector)
            confidence = max(probabilities[0])
            
            return intent, confidence
            
    except Exception as e:
        print(f"Ошибка при предсказании интента: {e}")
        return None, 0.0

def init_db():
    """
    Инициализация базы данных
    """
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
    
    # Таблица для хранения ML-предсказаний
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
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
    """
    Сохранение пользователя в БД
    """
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR REPLACE INTO users (user_id, name, last_interaction) VALUES (?, ?, ?)",
        (user_id, name, datetime.now())
    )

    conn.commit()
    conn.close()

def get_user(user_id):
    """
    Получение пользователя из БД
    """
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM users WHERE user_id=?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None

def save_weather_query(user_id, user_query, city, weather_response):
    """
    Сохранение запроса погоды в БД
    """
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO weather_queries (user_id, user_query, city, weather_response, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, user_query, city, weather_response, datetime.now()))

    conn.commit()
    conn.close()

def save_text_analysis(user_id, text, analysis):
    """
    Сохранение анализа текста в БД
    """
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

def save_ml_prediction(user_id, text, intent, confidence):
    """
    Сохранение ML-предсказания в БД
    """
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO ml_predictions (user_id, original_text, predicted_intent, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, text, intent, confidence, datetime.now()))
    
    conn.commit()
    conn.close()

def save_dialog_state(user_id, state, temp_data=None):
    """
    Сохранение состояния диалога
    """
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
    """
    Загрузка состояния диалога
    """
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
    """
    Извлечение города из текста с помощью spaCy
    """
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text
    
    return None

def extract_date_with_spacy(text):
    """
    Извлечение даты из текста
    """
    text_lower = text.lower()
    
    if 'сегодня' in text_lower or 'сейчас' in text_lower:
        return 'сегодня'
    elif 'завтра' in text_lower:
        return 'завтра'
    elif 'послезавтра' in text_lower:
        return 'послезавтра'
    
    return None

def is_weather_query_with_spacy(text):
    """
    Проверка, является ли запрос запросом погоды
    """
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
    """
    Анализ текста с помощью spaCy
    """
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
    """
    Обработчик приветствия
    """
    return "Здравствуйте! Чем могу помочь?"

def handle_farewell(match=None):
    """
    Обработчик прощания
    """
    return "До свидания! Будет приятно видеть вас снова."

def log_message(user_message, bot_response):
    """
    Логирование сообщений в файл
    """
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
        """
        Сохраняет текущее состояние в БД
        """
        if self.user_id:
            save_dialog_state(self.user_id, self.state, self.temp_data)
    
    def reset_state(self):
        """
        Сброс состояния после завершения диалога
        """
        self.state = BotState.START
        self.temp_data = {}
        self.save_state()
    
    def set_name(self, match):
        """
        Установка имени пользователя
        """
        name = match.group(1)
        self.name = name
        
        if self.user_id and self.name:
            save_user(self.user_id, self.name)
        
        self.reset_state()    
        return f"Приятно познакомиться, {self.name}! Я запомнил ваше имя."

    def greet(self, match=None):
        """
        Приветствие с учётом имени
        """
        if self.name:
            return f"Здравствуйте, {self.name}! Рад вас снова видеть."
        return "Здравствуйте! Как я могу к вам обращаться?"

    def start_weather_dialog(self, message):
        """
        Начало диалога о погоде
        """
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
        """
        Начало диалога о сложении
        """
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
        """
        Обрабатывает ввод города
        """
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
        """
        Обрабатывает ввод даты
        """
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
        """
        Обрабатывает ввод первого числа
        """
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
        """
        Обрабатывает ввод второго числа и выполняет сложение
        """
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
    """
    Создание списка паттернов с обработчиками (резервный вариант)
    """
    return [
        # Приветствия
        (re.compile(r"^(привет|здравствуй|добрый день|доброе утро|добрый вечер|хай|здарова)$", re.IGNORECASE), 
         bot_instance.greet),
        
        # Прощания
        (re.compile(r"^(пока|до свидания|всего хорошего|до встречи|увидимся)$", re.IGNORECASE), 
         handle_farewell),
        
        # Установка имени
        (re.compile(r"(?:меня зовут|мое имя|называй меня|зови меня|меня звать)\s+([а-яА-Яa-zA-Z]+)", re.IGNORECASE), 
         bot_instance.set_name),
        
        # Запрос погоды - запуск диалога с анализом сообщения
        (re.compile(r".*(?:погода|температура|прогноз|холодно|жарко|дождь|снег).*", re.IGNORECASE), 
         lambda m: bot_instance.start_weather_dialog(m.string)),
        
        # Сложение - запуск диалога с анализом сообщения
        (re.compile(r"(?:сложи|плюс|сумма|прибавь|сложение|\+)"), 
         lambda m: bot_instance.start_addition_dialog(m.string)),
    ]

def process_message(message: str, bot_instance: ChatBot) -> str:
    """
    Обработка сообщения пользователя с поддержкой FSM и ML-классификации
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
        
        # Если нет активного состояния, используем ML для определения интента
        intent, confidence = predict_intent_with_confidence(message)
        
        # Сохраняем ML-предсказание в БД
        if bot_instance.user_id and intent:
            save_ml_prediction(bot_instance.user_id, message, intent, confidence)
        
        # Проверяем уверенность модели
        if confidence < 0.2 and intent is not None:
            return "Извините, я не уверен, что правильно понял ваш запрос. Пожалуйста, переформулируйте."
        
        # Обработка на основе ML-интента
        if intent == "greeting":
            return bot_instance.greet()
        
        elif intent == "goodbye":
            return handle_farewell()
        
        elif intent == "weather":
            return bot_instance.start_weather_dialog(message)
        
        elif intent == "addition":
            return bot_instance.start_addition_dialog(message)
        
        elif intent == "set_name":
            # Извлекаем имя из сообщения
            name_match = re.search(r"(?:меня зовут|мое имя|называй меня|зови меня|меня звать|я)\s+([а-яА-Яa-zA-Z]+)", message, re.IGNORECASE)
            if name_match:
                name = name_match.group(1)
                bot_instance.name = name
                if bot_instance.user_id and name:
                    save_user(bot_instance.user_id, name)
                bot_instance.reset_state()
                return f"Приятно познакомиться, {name}! Я запомнил ваше имя."
            else:
                return "Как вас зовут? Скажите, например: 'Меня зовут Анна'"
        
        elif intent == "unknown" or intent is None:
            return "Я не совсем понял ваш запрос. Я могу:\n" \
                   "- Рассказать о погоде\n" \
                   "- Помочь сложить числа\n" \
                   "- Запомнить ваше имя\n" \
                   "- Просто поздороваться или попрощаться\n\n" \
                   "Что вы хотите сделать?"
        
        else:
            # Если интент не распознан, используем резервный режим с паттернами
            patterns = create_patterns(bot_instance)
            for pattern, handler in patterns:
                match = pattern.search(message)
                if match:
                    return handler(match)
            
            return "Я не понимаю запрос. Попробуйте переформулировать."
               
    except Exception as e:
        print(f"Ошибка: {e}")
        return "Произошла ошибка. Попробуйте ещё раз или начните новый диалог."