# 🏥 MedGemma 4B - Руководство по использованию

## 📋 Что такое MedGemma?

**MedGemma** - это семейство медицинских AI-моделей от Google, основанных на Gemma 3. Специально обучены для работы с медицинскими данными: изображениями (рентген, КТ, МРТ) и текстом.

## 🎯 Ответы на основные вопросы

### ✅ Готова ли модель к использованию?
**ДА!** MedGemma 4B-it (instruction-tuned) работает "из коробки" без дополнительного обучения.

### 🔄 Нужен ли fine-tuning?
**НЕ ОБЯЗАТЕЛЬНО**, но рекомендуется для:
- Специфических медицинских задач
- Улучшения точности на конкретных типах данных
- Адаптации под ваши данные

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install transformers torch accelerate pillow requests
```

### 2. Настройка Hugging Face
```bash
# Войдите в Hugging Face
huggingface-cli login

# Или установите токен
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

### 3. Принятие лицензии
Перейдите на https://huggingface.co/google/medgemma-4b-it и примите условия использования.

### 4. Запуск примера
```bash
python medgemma_example.py
```

## 📊 Возможности MedGemma

### 🖼️ Анализ медицинских изображений
- **Рентгеновские снимки** грудной клетки
- **КТ и МРТ** сканы
- **Дерматологические** изображения
- **Офтальмологические** снимки
- **Гистопатологические** изображения

### 📝 Медицинские задачи
- Генерация медицинских отчетов
- Ответы на медицинские вопросы
- Визуальный вопросно-ответный анализ
- Классификация медицинских изображений
- Анализ медицинских записей

## 💻 Примеры использования

### Анализ рентгеновского снимка
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device="auto"
)

# Загружаем изображение
image = Image.open("chest_xray.jpg")

# Анализируем
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}]
    },
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "Describe this X-ray"},
            {"type": "image", "image": image}
        ]
    }
]

result = pipe(text=messages, max_new_tokens=200)
print(result[0]["generated_text"][-1]["content"])
```

### Медицинский вопрос-ответ
```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device="auto"
)

question = "What are the symptoms of pneumonia?"
prompt = f"You are a medical expert. Answer: {question}"

result = pipe(prompt, max_new_tokens=150)
print(result[0]["generated_text"])
```

## 🎓 Fine-tuning (опционально)

### Когда нужен fine-tuning?
- Специфические медицинские задачи
- Работа с уникальными типами данных
- Повышение точности на конкретных задачах

### Требования для fine-tuning
- **GPU**: минимум 16GB VRAM
- **Время**: несколько часов
- **Данные**: размеченные медицинские данные

### Запуск fine-tuning
```bash
python medgemma_finetune_example.py
```

## 📈 Производительность

### MedGemma 4B vs Gemma 3 4B
| Задача | Gemma 3 4B | MedGemma 4B |
|--------|------------|-------------|
| MIMIC CXR (F1) | 81.2 | 88.9 |
| CheXpert CXR (F1) | 32.6 | 48.1 |
| MedQA (accuracy) | 50.7 | 64.4 |
| MedMCQA (accuracy) | 45.4 | 55.7 |

## ⚠️ Важные ограничения

### Безопасность
- **НЕ для клинической диагностики** без валидации
- Требуется независимая проверка результатов
- Все выводы - предварительные

### Технические ограничения
- Оценена на задачах с одним изображением
- Не оптимизирована для многоходовых диалогов
- Чувствительна к формулировке промптов

## 🔧 Решение проблем

### Ошибка авторизации
```bash
huggingface-cli login
# Введите ваш токен
```

### Недостаток памяти
```python
# Используйте меньший batch size
model = AutoModelForImageTextToText.from_pretrained(
    'google/medgemma-4b-it',
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
```

### Медленная работа
- Используйте GPU
- Уменьшите max_new_tokens
- Используйте torch.bfloat16

## 📚 Дополнительные ресурсы

- [Официальная документация](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [GitHub репозиторий](https://github.com/google-health/medgemma)
- [Colab примеры](https://github.com/google-health/medgemma/blob/main/notebooks/quick_start_with_hugging_face.ipynb)
- [Технический отчет](https://arxiv.org/abs/2507.05201)

## 🎯 Заключение

MedGemma 4B - это мощная готовая модель для медицинских задач. Она работает "из коробки" и показывает отличные результаты на медицинских данных. Fine-tuning рекомендуется только для специфических задач, требующих максимальной точности.

**Начните с готовой модели, оцените результаты, и при необходимости проведите fine-tuning!**
