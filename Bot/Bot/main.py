import sys
from bot_core import ChatBot, process_message, log_message, init_db

def main():
    # Инициализация базы данных
    print("Инициализация базы данных...")
    init_db()

    # Создание бота с фиксированным id
    bot = ChatBot(user_id=123456789)
    
    print("=" * 50)
    print("Бот: Здравствуйте!")
    print("Для выхода напишите 'пока' или 'до свидания'")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            if user_input.lower() in ('выход', 'exit', 'quit'):
                print("Бот: До свидания!")
                break
            
            response = process_message(user_input, bot)
            print("Бот:", response)
            
            # Логируем сообщение
            log_message(user_input, response)
            
            # Проверяем, не попрощался ли пользователь
            if any(word in user_input.lower() for word in ['пока', 'до свидания']):
                print("Бот: Всего доброго! Заходите ещё.")
                break
                
        except KeyboardInterrupt:
            print("\nБот: Работа завершена. До свидания!")
            break
        except Exception as e:
            print(f"Бот: Произошла ошибка: {e}")
            print("Бот: Попробуйте ещё раз или напишите 'помощь'")

if __name__ == "__main__":
    main()