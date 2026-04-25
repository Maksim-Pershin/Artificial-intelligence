# main.py - Запуск бота с голосовым вводом (с подтверждением)
import sys
import time
from bot_core import ChatBot, process_message, log_message, init_db, load_bert, init_tts
from voice_input import listen, is_whisper_available

def print_commands():
    """Вывод списка команд"""
    print("\n" + "=" * 70)
    print("📌 Управление ботом:")
    print("   • Нажмите Enter, затем говорите (4 секунды)")
    print("   • Введите '!text' — переключиться на текстовый ввод")
    print("   • Введите '!voice' — переключиться на голосовой ввод")
    print("   • Введите '!help' — показать команды")
    print("   • Введите 'выход' — завершить работу")
    print("=" * 70 + "\n")

def main():
    print("=" * 70)
    print("🤖 ЧАТ-БОТ НА BERT + ГОЛОСОВОЙ ВВОД (Whisper) + TTS")
    print("=" * 70)
    
    # Инициализация
    print("📁 Инициализация базы данных...")
    init_db()
    
    print("🧠 Загрузка BERT модели...")
    bert_loaded = load_bert()
    if not bert_loaded:
        print("\n❌ BERT модель не найдена!")
        print("📌 Запустите python train_bert.py для обучения модели")
        return
    print("✅ BERT модель успешно загружена!")
    
    print("\n🎤 Проверка Whisper...")
    whisper_available = is_whisper_available()
    if whisper_available:
        print("✅ Whisper готов к работе!")
    else:
        print("⚠️ Whisper не доступен. Бот будет работать только с текстовым вводом.")
    
    print("\n🔊 Инициализация TTS...")
    tts_loaded = init_tts()
    if tts_loaded:
        print("✅ TTS готов к работе!")
    else:
        print("⚠️ TTS не инициализирован.")
    
    # Создание бота
    user_id = 123456789
    bot = ChatBot(user_id=user_id)
    
    print("\n" + "=" * 70)
    greeting = bot.greet()
    print("🤖 Бот:", greeting)
    if tts_loaded:
        bot.speak_response(greeting)
    
    print_commands()
    
    # Режим ввода (по умолчанию текстовый, чтобы не мешал)
    voice_mode = False
    
    print(f"🎯 Текущий режим: {'ГОЛОСОВОЙ' if voice_mode else 'ТЕКСТОВЫЙ'}")
    print("   (Для голосового ввода введите !voice)\n")
    
    while True:
        try:
            if voice_mode and whisper_available:
                # Голосовой режим - ждём Enter
                print("\n🎤 [Голосовой режим] Нажмите Enter, затем говорите...")
                print("   (или введите '!text' для выхода из голосового режима)")
                
                user_input = input().strip()
                
                # Проверяем команды перед записью
                if user_input.lower() == '!text':
                    voice_mode = False
                    print("\n📝 Переключено на ТЕКСТОВЫЙ режим ввода\n")
                    continue
                
                if user_input.lower() in ('выход', 'exit', 'quit'):
                    goodbye_msg = "До свидания! Буду рад видеть вас снова! 👋"
                    print("🤖 Бот:", goodbye_msg)
                    if tts_loaded:
                        bot.speak_response("До свидания! Буду рад видеть вас снова!")
                    break
                
                # Запись и распознавание
                text = listen(seconds=4)
                
                if not text:
                    print("⚠️ Не распознано. Попробуйте ещё раз.")
                    continue
                
                print(f"📝 Распознано: {text}")
                user_input = text
                
            else:
                # Текстовый режим
                user_input = input("👤 Вы: ").strip()
                if not user_input:
                    continue
            
            # Обработка команд (всегда работают)
            if user_input.lower() in ('выход', 'exit', 'quit'):
                goodbye_msg = "До свидания! Буду рад видеть вас снова! 👋"
                print("🤖 Бот:", goodbye_msg)
                if tts_loaded:
                    bot.speak_response("До свидания! Буду рад видеть вас снова!")
                break
            
            if user_input == "!text" and voice_mode:
                voice_mode = False
                print("\n📝 Переключено на ТЕКСТОВЫЙ режим ввода\n")
                continue
                
            if user_input == "!voice" and not voice_mode and whisper_available:
                voice_mode = True
                print("\n🎤 Переключено на ГОЛОСОВОЙ режим ввода")
                print("   Нажмите Enter, затем говорите\n")
                continue
            
            if user_input == "!help":
                print_commands()
                continue
            
            if user_input == "!mode":
                print(f"📌 Текущий режим: {'ГОЛОСОВОЙ' if voice_mode else 'ТЕКСТОВЫЙ'}")
                continue
            
            # Обработка сообщения
            response = process_message(user_input, bot)
            print("🤖 Бот:", response)
            
            if tts_loaded and response:
                bot.speak_response(response)
            
            log_message(user_input, response)
                
        except KeyboardInterrupt:
            print("\n🤖 Бот: Работа завершена. До свидания! 👋")
            if tts_loaded:
                bot.speak_response("До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("🤖 Бот: Попробуйте ещё раз")

if __name__ == "__main__":
    main()