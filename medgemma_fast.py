#!/usr/bin/env python3
"""
Быстрая версия MedGemma для анализа изображений
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import time

MODEL_PATH = "./models/medgemma_4b"

def load_model_fast():
    """Быстрая загрузка модели с оптимизациями"""
    print('⚡ Быстрая загрузка MedGemma...')
    
    # Проверяем GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'🖥️  Используем устройство: {device}')
    
    try:
        # Загружаем с оптимизациями
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        if device == "cpu":
            model = model.to(device)
        
        print('✅ Модель загружена быстро!')
        return model, processor, device
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        return None, None, None

def quick_analyze(model, processor, device, image_path=None):
    """Быстрый анализ изображения"""
    print('\n⚡ Быстрый анализ изображения...')
    
    start_time = time.time()
    
    try:
        # Загружаем изображение
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            print(f'📁 Изображение: {image_path}')
        else:
            # Создаем тестовое изображение
            image = Image.new('RGB', (512, 512), color='white')
            print('📷 Тестовое изображение')
        
        # Простой промпт на русском для скорости
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Кратко опишите это медицинское изображение на русском языке."},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # Быстрая обработка
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Быстрая генерация
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=50,  # Очень короткий ответ для скорости
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
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

def benchmark_model():
    """Бенчмарк производительности"""
    print('🏁 Бенчмарк MedGemma...')
    
    model, processor, device = load_model_fast()
    if model is None:
        return
    
    print(f'\n📊 Информация о модели:')
    print(f'   Устройство: {device}')
    print(f'   Тип данных: {model.dtype}')
    print(f'   Параметры: ~4B')
    
    # Тест скорости
    times = []
    for i in range(3):
        print(f'\n🔄 Тест {i+1}/3...')
        result, time_taken = quick_analyze(model, processor, device)
        if time_taken > 0:
            times.append(time_taken)
    
    if times:
        avg_time = sum(times) / len(times)
        print(f'\n📈 РЕЗУЛЬТАТЫ БЕНЧМАРКА:')
        print(f'   Среднее время: {avg_time:.1f} сек')
        print(f'   Минимальное: {min(times):.1f} сек')
        print(f'   Максимальное: {max(times):.1f} сек')
        
        if avg_time < 10:
            print('⚡ Отличная скорость!')
        elif avg_time < 30:
            print('✅ Хорошая скорость')
        else:
            print('🐌 Медленно, нужна оптимизация')

def main():
    """Основная функция"""
    print('⚡ MedGemma 4B - Быстрая версия')
    print('=' * 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f'❌ Модель не найдена в {MODEL_PATH}')
        return
    
    # Запускаем бенчмарк
    benchmark_model()
    
    print('\n💡 Советы для ускорения:')
    print('   1. Используйте GPU (CUDA)')
    print('   2. Уменьшите max_new_tokens')
    print('   3. Используйте torch.float16')
    print('   4. Включите use_cache=True')

if __name__ == "__main__":
    main()
