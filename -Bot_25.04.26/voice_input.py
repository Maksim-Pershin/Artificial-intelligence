
import whisper
import sounddevice as sd
import numpy as np
import re
import threading
import time
import io
import scipy.io.wavfile as wav

# модель Whisper
_whisper_model = None
_model_lock = threading.Lock()

def load_whisper_model(model_size="base"):
    """Загрузка модели Whisper"""
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                print("🎤 Загрузка Whisper модели...")
                try:
                    _whisper_model = whisper.load_model(model_size)
                    print("✅ Whisper модель загружена!")
                except Exception as e:
                    print(f"❌ Ошибка загрузки Whisper: {e}")
                    _whisper_model = None
    return _whisper_model

def record_audio_to_array(seconds=4, fs=16000):
    """Запись с микрофона в numpy массив"""
    print("\n🎤 [Запись] Говорите...", end="", flush=True)
    
    for i in range(3):
        print(".", end="", flush=True)
        time.sleep(0.3)
    print()
    
    try:
        audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.float32)
        sd.wait()
        # Преобразуем в int16 (как нужно для WAV)
        audio_int16 = (audio * 32767).astype(np.int16)
        print("✅ [Запись] Готово!")
        return audio_int16, fs
    except Exception as e:
        print(f"\n❌ Ошибка записи: {e}")
        return None, None

def audio_to_bytes(audio_array, fs):
    
    buffer = io.BytesIO()
    wav.write(buffer, fs, audio_array)
    buffer.seek(0)
    return buffer

def transcribe_audio(audio_array, fs, language="ru"):
    """Распознавание из массива аудиоданных"""
    model = load_whisper_model()
    if model is None or audio_array is None:
        return ""
    
    try:
        # Конвертируем в float32 для Whisper
        if audio_array.dtype == np.int16:
            audio_float = audio_array.astype(np.float32) / 32767.0
        else:
            audio_float = audio_array.astype(np.float32)
        
        # Нормализуем
        if len(audio_float.shape) > 1:
            audio_float = audio_float.flatten()
        
        result = model.transcribe(audio_float, language=language, fp16=False)
        text = result["text"].strip()
        return text
    except Exception as e:
        print(f"❌ Ошибка распознавания: {e}")
        return ""

def clean_voice_text(text: str) -> str:
    """Очистка распознанного текста"""
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^а-яё0-9\s\.\,\!\?\-\—]', '', text)
    text = text.rstrip('.')
    return text.strip()

def listen(seconds=4, clean=True):
    """
    Основная функция голосового ввода
    Без сохранения файлов на диск
    """
    # Записываем аудио в массив
    audio_array, fs = record_audio_to_array(seconds=seconds)
    
    if audio_array is None:
        return ""
    
    # Распознаём напрямую из массива
    raw_text = transcribe_audio(audio_array, fs)
    
    if raw_text:
        print(f"🎤 Распознано: '{raw_text}'")
        if clean:
            return clean_voice_text(raw_text)
        return raw_text
    
    return ""

def is_whisper_available():
    """Проверка, доступен ли Whisper"""
    try:
        model = load_whisper_model()
        return model is not None
    except:
        return False