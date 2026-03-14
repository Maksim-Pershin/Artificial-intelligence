import sys
from bot_core import ChatBot, process_message, log_message, init_db, analyze_text_with_spacy

def main():
    
    print("Инициализация базы данных...")
    init_db()

    
    user_id = 123456789 
    
   
    bot = ChatBot(user_id=user_id)
    
    print("=" * 70)
    print(" Бот: Здравствуйте!")
    print("=" * 70)
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            # Проверка на выход или отмену
            if user_input.lower() in ('выход', 'exit', 'quit'):
                print(" Бот: До свидания!")
                break
                
            if user_input.lower() in ('отмена', 'стоп', 'cancel'):
                if bot.state.value != "start":
                    bot.reset_state()
                    print(" Бот: Диалог прерван. Чем могу помочь?")
                else:
                    print(" Бот: Нет активного диалога для отмены.")
                continue
            
            if user_input.lower() in ('пока', 'до свидания', 'всего хорошего'):
                response = process_message(user_input, bot)
                print(" Бот:", response)
                print(" Бот: Всего доброго! Заходите ещё.")
                break
            
            # Обработка сообщения
            response = process_message(user_input, bot)
            print(" Бот:", response)
            
            
            log_message(user_input, response)
            
            
                
        except KeyboardInterrupt:
            print("\n Бот: Работа завершена. До свидания!")
            break
        except Exception as e:
            print(f" Бот: Произошла ошибка: {e}")
            print(" Бот: Попробуйте ещё ")

if __name__ == "__main__":
    main()