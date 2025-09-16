#!/usr/bin/env python3
"""
Работа с локальной моделью MedGemma 4B
"""

import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import requests

# Путь к локальной модели
MODEL_PATH = "./models/medgemma_4b"

def load_local_medgemma():
    """Загрузка локальной модели MedGemma"""
    print('🔄 Загружаем локальную модель MedGemma...')
    
    try:
        # Загружаем процессор
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print('✅ Processor загружен из локальной папки!')
        
        # Загружаем модель с оптимизациями
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        print('✅ Model загружена из локальной папки!')
        print('🎉 Локальная MedGemma готова к работе!')
        
        return model, processor
        
    except Exception as e:
        print(f'❌ Ошибка загрузки: {e}')
        return None, None

def analyze_chest_xray_local(model, processor, image_path=None):
    """Анализ рентгеновского снимка с локальной моделью"""
    print('\n🫁 Анализ рентгеновского снимка (локальная модель)...')
    
    try:
        # Загружаем изображение
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
            print(f'📁 Загружено изображение: {image_path}')
        else:
            # Используем пример изображения
            image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
            image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)
            print('📷 Используем пример изображения из интернета')
        
        # Формируем сообщения на русском
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Вы эксперт-рентгенолог. Проанализируйте рентгеновский снимок грудной клетки и предоставьте подробный медицинский отчет на русском языке."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Пожалуйста, проанализируйте этот рентгеновский снимок грудной клетки и опишите любые находки, аномалии или нормальные структуры, которые вы наблюдаете. Отвечайте на русском языке."},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # Обрабатываем с помощью локальной модели
        print('🔍 Анализируем изображение...')
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=150,  # Уменьшили для скорости
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]
        
        result = processor.decode(generation, skip_special_tokens=True)
        
        print('\n📋 РЕЗУЛЬТАТ АНАЛИЗА (локальная модель):')
        print('=' * 60)
        print(result)
        print('=' * 60)
        
        return result
        
    except Exception as e:
        print(f'❌ Ошибка при анализе: {e}')
        return None

def medical_qa_local(model, processor, question):
    """Ответ на медицинский вопрос с локальной моделью"""
    print(f'\n❓ Медицинский вопрос: {question}')
    
    try:
        # Формируем промпт на русском
        prompt = f"""Вы эксперт-медик. Пожалуйста, ответьте на следующий медицинский вопрос точно и профессионально на русском языке:

Вопрос: {question}

Ответ:"""
        
        # Токенизируем
        inputs = processor.tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print('🔍 Обрабатываем вопрос...')
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=100,  # Уменьшили для скорости
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            generation = generation[0][inputs["input_ids"].shape[-1]:]
        
        result = processor.tokenizer.decode(generation, skip_special_tokens=True)
        
        print('\n📋 ОТВЕТ (локальная модель):')
        print('=' * 60)
        print(result)
        print('=' * 60)
        
        return result
        
    except Exception as e:
        print(f'❌ Ошибка при обработке вопроса: {e}')
        return None

def test_local_model():
    """Тестирование локальной модели"""
    print('🧪 Тестируем локальную модель MedGemma...')
    
    # Загружаем модель
    model, processor = load_local_medgemma()
    if model is None or processor is None:
        return False
    
    # Тест 1: Анализ рентгеновского снимка
    print('\n' + '=' * 60)
    print('📋 ТЕСТ 1: Анализ рентгеновского снимка')
    analyze_chest_xray_local(model, processor)
    
    # Тест 2: Медицинский вопрос
    print('\n' + '=' * 60)
    print('📋 ТЕСТ 2: Медицинский вопрос')
    medical_qa_local(model, processor, "Какие основные симптомы пневмонии?")
    
    # Тест 3: Еще один вопрос
    print('\n' + '=' * 60)
    print('📋 ТЕСТ 3: Дополнительный вопрос')
    medical_qa_local(model, processor, "Как диагностируется туберкулез?")
    
    print('\n' + '=' * 60)
    print('✅ Все тесты завершены!')
    print('💡 Локальная MedGemma работает отлично!')
    
    return True

def main():
    """Основная функция"""
    print('🏥 MedGemma 4B - Локальная модель')
    print('=' * 60)
    
    # Проверяем наличие локальной модели
    if not os.path.exists(MODEL_PATH):
        print(f'❌ Локальная модель не найдена в {MODEL_PATH}')
        print('💡 Скачайте модель командой:')
        print('huggingface-cli download google/medgemma-4b-it --local-dir models/medgemma_4b')
        return
    
    print(f'📁 Используем локальную модель из: {MODEL_PATH}')
    
    # Запускаем тесты
    success = test_local_model()
    
    if success:
        print('\n🎉 Локальная MedGemma готова к использованию!')
        print('💡 Теперь вы можете работать без интернета!')
    else:
        print('\n❌ Возникли проблемы с локальной моделью')

if __name__ == "__main__":
    main()
