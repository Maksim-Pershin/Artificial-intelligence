import re
from datetime import datetime


def handle_greeting(match=None):
    return "Здравствуйте! Чем могу помочь?"


def handle_farewell(match=None):
    return "До свидания!"


def handle_weather(match):
    city = match.group(1)
    return f"Погода в городе {city}: солнечно (демо-режим)."


def handle_addition(match):
    a = float(match.group(1))
    b = float(match.group(2))
    return f"Результат: {a + b}"


class ChatBot:
    def __init__(self):
        self.name = None

    def set_name(self, match):
        self.name = match.group(1)
        return f"Приятно познакомиться, {self.name}!"

    def greet(self, match=None):
        if self.name:
            return f"Здравствуйте, {self.name}!"
        return "Здравствуйте!"


def log_message(user, bot):
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] USER: {user}\n")
        f.write(f"[{datetime.now()}] BOT: {bot}\n")


bot = ChatBot()

patterns = [
    (re.compile(r"^(привет|здравствуй|добрый день)$", re.IGNORECASE), handle_greeting),
    (re.compile(r"^(пока|до свидания)$", re.IGNORECASE), handle_farewell),
    
    (re.compile(r"погода в ([а-яА-Яa-zA-Z\- ]+)", re.IGNORECASE), handle_weather),

    (re.compile(r"(\d+)\s*\+\s*(\d+)"), handle_addition),
    
    (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), bot.set_name),
    
    (re.compile(r"^(привет|здравствуй)", re.IGNORECASE), bot.greet),
]


def process_message(message: str):
 
    message = message.strip()

    for pattern, handler in patterns:
        match = pattern.search(message) 
        if match:
            return handler(match)

    return "Я не понимаю запрос."


if __name__ == "__main__":
    print("Бот: Здравствуйте! Я бот. Имя, сложить чилса, погода, прощание")
    
    while True:
        user_input = input("Вы: ")
        
        response = process_message(user_input)
        
        print("Бот:", response)
        
        log_message(user_input, response)
        
        if "до свидания" in user_input.lower():
            break