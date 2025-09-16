#!/usr/bin/env python3
"""
MedGemma оптимизированная для MacBook Air M3
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import time

MODEL_PATH = "./models/medgemma_4b"

def load_model_m3():
    """Загрузка модели оптимизированная для M3"""
    print('🍎 Загружаем MedGemma для MacBook Air M3...')
    
    try:
        # Для M3 используем MPS (Metal Performance Shaders)
        if torch.backends.mps.is_available():
            device = "mps"
            print('✅ Используем MPS (Metal) ускорение')
        else:
            device = "cpu"
            print('⚠️  MPS недоступен, используем CPU')
        
        # Оптимизации для M3
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,  # float16 для экономии памяти
            device_map=None,  # Ручное управление устройством
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        # Перемещаем модель на устройство
        model = model.to(device)
        
        print(f'✅ Модель загружена на {device}')
        return model, processor, device
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        return None, None, None

def quick_analyze_m3(model, processor, device, image_path=None):
    """Быстрый анализ для M3"""
    print('\n⚡ Быстрый анализ (M3 оптимизация)...')
    
    start_time = time.time()
    
    try:
        # Загружаем изображение
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            # Уменьшаем размер для скорости
            image = image.resize((512, 512))
            print(f'📁 Изображение: {image_path}')
        else:
            # Создаем маленькое тестовое изображение
            image = Image.new('RGB', (256, 256), color='white')
            print('📷 Тестовое изображение (256x256)')
        
        # Очень простой промпт на русском
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Кратко опишите это изображение на русском языке."},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # Обработка
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Перемещаем на устройство
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Очень быстрая генерация
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=30,  # Очень короткий ответ
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                temperature=0.1
            )
            generation = generation[0][input_len:]
        
        result = processor.decode(generation, skip_special_tokens=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f'\n⚡ РЕЗУЛЬТАТ (за {processing_time:.1f} сек):')
        print('=' * 50)
        print(result)
        print('=' * 50)
        
        return result, processing_time
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        return None, 0

def text_only_qa_m3(model, processor, device, question):
    """Только текстовые вопросы (быстрее)"""
    print(f'\n❓ Текстовый вопрос: {question}')
    
    start_time = time.time()
    
    try:
        # Простой промпт на русском
        prompt = f"Ответьте кратко на русском языке: {question}"
        
        inputs = processor.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
            generation = generation[0][inputs["input_ids"].shape[-1]:]
        
        result = processor.tokenizer.decode(generation, skip_special_tokens=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f'\n⚡ ОТВЕТ (за {processing_time:.1f} сек):')
        print('=' * 50)
        print(result)
        print('=' * 50)
        
        return result, processing_time
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        return None, 0

def benchmark_m3():
    """Бенчмарк для M3"""
    print('🏁 Бенчмарк MedGemma на MacBook Air M3...')
    
    model, processor, device = load_model_m3()
    if model is None:
        return
    
    print(f'\n📊 Информация:')
    print(f'   Устройство: {device}')
    print(f'   Тип данных: {model.dtype}')
    print(f'   MPS доступен: {torch.backends.mps.is_available()}')
    
    # Тест 1: Текстовые вопросы (быстрее)
    print('\n🔄 Тест 1: Текстовые вопросы...')
    times_text = []
    questions = [
        "Что такое пневмония?",
        "Симптомы гриппа?",
        "Как лечить температуру?"
    ]
    
    for q in questions:
        result, time_taken = text_only_qa_m3(model, processor, device, q)
        if time_taken > 0:
            times_text.append(time_taken)
    
    # Тест 2: Анализ изображения (медленнее)
    print('\n🔄 Тест 2: Анализ изображения...')
    result, time_image = quick_analyze_m3(model, processor, device)
    
    # Результаты
    print(f'\n📈 РЕЗУЛЬТАТЫ ДЛЯ M3:')
    if times_text:
        avg_text = sum(times_text) / len(times_text)
        print(f'   Текстовые вопросы: {avg_text:.1f} сек')
    print(f'   Анализ изображения: {time_image:.1f} сек')
    
    # Рекомендации
    print(f'\n💡 РЕКОМЕНДАЦИИ ДЛЯ M3:')
    if time_image > 60:
        print('   ⚠️  Анализ изображений очень медленный')
        print('   💡 Используйте только текстовые вопросы')
    elif time_image > 30:
        print('   ⚠️  Анализ изображений медленный')
        print('   💡 Уменьшите размер изображений')
    else:
        print('   ✅ Производительность приемлемая')

def main():
    """Основная функция"""
    print('🍎 MedGemma 4B - MacBook Air M3')
    print('=' * 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f'❌ Модель не найдена в {MODEL_PATH}')
        return
    
    # Проверяем MPS
    if torch.backends.mps.is_available():
        print('✅ MPS (Metal) ускорение доступно')
    else:
        print('⚠️  MPS недоступен - будет медленно')
    
    # Запускаем бенчмарк
    benchmark_m3()
    
    print('\n🎯 СОВЕТЫ ДЛЯ M3:')
    print('   1. Используйте только текстовые вопросы')
    print('   2. Уменьшите размер изображений до 256x256')
    print('   3. max_new_tokens = 30-50 максимум')
    print('   4. Рассмотрите более легкую модель')

if __name__ == "__main__":
    main()
