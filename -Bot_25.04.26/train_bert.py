# train_bert.py - Дообучение BERT (РАСШИРЕННАЯ ВЕРСИЯ с 7 интентами)
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import json

MODEL_NAME = "cointegrated/rubert-tiny2"
OUTPUT_DIR = "bert_intent_model"

def create_dataset():
    """Создание расширенного датасета для 7 интентов"""
    
    # 1. Приветствия
    greeting_examples = [
        "привет", "здравствуй", "здравствуйте", "добрый день", "доброе утро", "добрый вечер",
        "хай", "здарова", "приветствую", "салют", "приветик", "здрасьте", "здорово",
        "привет всем", "всем привет", "хаюшки", "здорова", "прив", "дратути", "о привет",
        "йоу", "хеллоу", "хей", "ку", "дарова", "салам", "привет бот", "здравствуй бот",
        "добрый день бот", "здравствуйте как жизнь", "доброго здоровья", "приветствую вас"
    ]
    
    # 2. Погода
    weather_examples = [
        "какая погода", "погода в москве", "сколько градусов", "температура на улице",
        "будет ли дождь", "прогноз погоды", "что с погодой", "холодно на улице",
        "жарко сегодня", "какая температура завтра", "погода на завтра", "снег будет",
        "ветер сильный", "прогноз на сегодня", "метео", "погода сейчас", "какая сегодня погода",
        "погода на неделю", "прогноз на выходные", "солнечно сегодня", "дождь будет",
        "снег ожидается", "ветер какой", "на улице что творится", "градусник показывает",
        "замерзнешь на улице", "тепло или холодно", "нужна куртка", "зонт нужен", "гроза будет"
    ]
    
    # 3. Время (НОВЫЙ ИНТЕНТ)
    time_examples = [
        "сколько времени", "который час", "текущее время", "скажи время", "время сейчас",
        "какое сейчас время", "часы покажи", "точное время", "сколько сейчас часов",
        "который час сейчас", "время московское", "местное время", "текущий час",
        "покажи время", "во сколько", "который час покажи", "сколько минут", "время суток"
    ]
    
    # 4. Дата (НОВЫЙ ИНТЕНТ)
    date_examples = [
        "какое сегодня число", "сегодняшняя дата", "какое число", "какой сегодня день",
        "число сегодня", "день недели", "какой день недели", "сегодня какой день",
        "дата сегодня", "текущая дата", "какое сегодня число и месяц", "сегодняшнее число",
        "какого числа", "какой сегодня день недели", "день месяц год", "сегодняшний день"
    ]
    
    # 5. Прощания
    goodbye_examples = [
        "пока", "до свидания", "всего хорошего", "до встречи", "увидимся", "прощай", "бывай",
        "счастливо", "удачи", "до скорого", "всего доброго", "покеда", "чао", "до связи",
        "пока пока", "прощайте", "до завтра", "бывай здоров", "до встречи пока", "удачи пока",
        "созвонимся", "до новых встреч", "хорошего дня", "пока бот", "спокойной ночи"
    ]
    
    # 6. Сложение
    addition_examples = [
        "сложи 5 и 3", "2 плюс 2", "сумма 10 и 20", "прибавь 15 к 7", "сколько будет 8 + 4",
        "сложение", "плюс", "прибавить", "калькулятор", "сколько 5+3", "вычисли 10+20",
        "прибавь 15+7", "плюс 2 и 2", "найди сумму 3 и 6", "сложить 4 и 9", "1 плюс 1",
        "100+200", "5+5", "8+4", "два плюс два", "три плюс шесть", "десять плюс двадцать"
    ]
    
    # 7. Установка имени
    set_name_examples = [
        "меня зовут Максим", "меня зовут Анна", "меня зовут Дмитрий", "мое имя Иван",
        "называй меня Саша", "зови меня Катя", "я Максим", "меня звать Олег",
        "можно просто Петя", "обращайся ко мне Лена", "зовут меня Алексей",
        "меня зовут пожалуйста Максим", "давай познакомимся", "будем знакомы меня зовут",
        "меня можно называть", "вот мое имя", "меня зовут Сергей", "меня зовут Алексей",
        "я Антон", "я Денис", "я Кристина", "я Юлия", "называй меня Макс", "зови меня Дима"
    ]
    
    # 8. SmallTalk (НОВЫЙ ИНТЕНТ - разговорные ответы)
    smalltalk_examples = [
        "как дела", "как ты", "как жизнь", "как настроение", "что делаешь", "чем занимаешься",
        "как поживаешь", "как сам", "как у тебя дела", "что нового", "как твои дела",
        "как работаешь", "как настроение сегодня", "ты как", "как себя чувствуешь",
        "что нового у тебя", "как ты поживаешь", "что слышно", "как жизнь молодая",
        "как ты сегодня", "расскажи что-нибудь", "что интересного", "как твоё настроение"
    ]
    
    # 9. Help (НОВЫЙ ИНТЕНТ - справка)
    help_examples = [
        "что ты умеешь", "помощь", "справка", "команды", "что можешь", "какие у тебя функции",
        "что ты можешь делать", "расскажи о себе", "помоги", "нужна помощь", "как пользоваться",
        "твои возможности", "что ты умеешь делать", "покажи команды", "список команд",
        "что я могу спросить", "расскажи о своих навыках", "как ты работаешь", "функции бота", "помоги мне", "подскажи", "что ты можешь сделать",
    "расскажи о функциях", "какие у тебя есть команды",
    "помощь бота", "как работать с ботом"
    ]
    
    # 10. Unknown (всё остальное)
    unknown_examples = [
        "расскажи шутку", "кто твой создатель", "сколько времени", "какой сегодня день",
        "спасибо", "откуда ты", "ты человек", "расскажи анекдот", "спой песню", "потанцуй",
        "что ты любишь", "какой твой любимый цвет", "у тебя есть друзья", "скучно",
        "развлеки меня", "поиграем", "хочешь кофе", "новости мира", "как пройти",
        "где находится", "посоветуй", "расскажи историю", "что происходит", "зачем ты"
    ]
    
    texts = []
    intents = []
    
    # Собираем все примеры
    for ex in greeting_examples:
        texts.append(ex)
        intents.append("greeting")
    for ex in weather_examples:
        texts.append(ex)
        intents.append("weather")
    for ex in time_examples:
        texts.append(ex)
        intents.append("time")
    for ex in date_examples:
        texts.append(ex)
        intents.append("date")
    for ex in goodbye_examples:
        texts.append(ex)
        intents.append("goodbye")
    for ex in addition_examples:
        texts.append(ex)
        intents.append("addition")
    for ex in set_name_examples:
        texts.append(ex)
        intents.append("set_name")
    for ex in smalltalk_examples:
        texts.append(ex)
        intents.append("smalltalk")
    for ex in help_examples:
        texts.append(ex)
        intents.append("help")
    for ex in unknown_examples:
        texts.append(ex)
        intents.append("unknown")
    
    df = pd.DataFrame({"text": texts, "intent": intents})
    df = df.drop_duplicates(subset=["text"])
    df.to_csv("dataset.csv", index=False, encoding='utf-8')
    
    print(f"📊 Создан датасет: {len(df)} примеров")
    print(f"📈 Распределение по интентам:")
    for intent, count in df['intent'].value_counts().items():
        print(f"     {intent}: {count} примеров")
    
    return df

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert():
    print("=" * 70)
    print("Дообучение BERT для классификации 7+ интентов")
    print("=" * 70)
    
    # Создание или загрузка данных
    if not os.path.exists("dataset.csv"):
        df = create_dataset()
    else:
        df = pd.read_csv("dataset.csv")
        print(f"✅ Загружен датасет из dataset.csv: {len(df)} примеров")
        print(f"📊 Распределение по интентам:\n{df['intent'].value_counts()}")
    
    # Создание маппинга интентов
    unique_intents = sorted(df['intent'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_intents)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"\n📋 Интенты ({len(unique_intents)}): {list(label2id.keys())}")
    
    df['label'] = df['intent'].map(label2id)
    
    # Разделение на train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label'].tolist()
    )
    
    # Загрузка токенизатора и модели
    print(f"\n🔄 Загрузка модели {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_intents),
        id2label=id2label,
        label2id=label2id
    )
    
    # Токенизация
    print("📝 Токенизация данных...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Создание датасета PyTorch
    class IntentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)
    
    # Настройка обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=35,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=3e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Обучение
    print("\n🚀 Начало обучения BERT...")
    trainer.train()
    
    # Сохранение модели
    print(f"\n💾 Сохранение модели в {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Сохраняем маппинг в JSON
    with open(f"{OUTPUT_DIR}/label_map.json", 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    
    # Оценка на валидации
    print("\n📊 Оценка модели на валидации:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Тестирование на новых примерах
    print("\n" + "=" * 70)
    print("🧪 Тестирование BERT модели на новых примерах:")
    print("=" * 70)
    
    test_examples = [
        ("привет как дела", "greeting/smalltalk"),
        ("сколько времени", "time"),
        ("какое сегодня число", "date"),
        ("какая погода в москве", "weather"),
        ("что ты умеешь", "help"),
        ("как настроение", "smalltalk"),
        ("сложи 5 и 3", "addition"),
        ("пока до свидания", "goodbye"),
        ("меня зовут Антон", "set_name"),
        ("который час", "time"),
        ("какой сегодня день недели", "date"),
        ("что делаешь", "smalltalk"),
        ("помощь", "help")
    ]
    
    model.eval()
    correct = 0
    for example, expected in test_examples:
        inputs = tokenizer(example, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0][pred_class].item()
        
        intent = id2label[pred_class]
        is_correct = expected.split('/')[0] == intent
        if is_correct:
            correct += 1
        
        mark = "✅" if is_correct else "❌"
        print(f"{mark} '{example}' → {intent} (уверенность: {confidence:.2%}) [ожидалось: {expected}]")
    
    print(f"\n📈 Точность на тестовых примерах: {correct}/{len(test_examples)} = {correct/len(test_examples)*100:.1f}%")
    
    print(f"\n✅ Модель сохранена в {OUTPUT_DIR}")
    return model

if __name__ == "__main__":
    train_bert()