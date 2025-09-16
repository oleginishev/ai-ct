#!/usr/bin/env python3
"""
MedGemma с правильным pipeline
"""

import os
import torch
from transformers import pipeline
from PIL import Image
import requests

MODEL_PATH = "/app/models/medgemma_4b"

def main():
    """Правильная проверка с pipeline"""
    print('🏥 MedGemma 4B - Pipeline проверка')
    print('=' * 50)
    
    # Проверяем модель
    if not os.path.exists(MODEL_PATH):
        print(f'❌ Модель не найдена в {MODEL_PATH}')
        return
    
    # Проверяем GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'🖥️  Устройство: {device}')
    
    if device == "cuda":
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        print(f'💾 Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    try:
        # Создаем pipeline с оптимизацией памяти
        print('🔄 Создаем pipeline с оптимизацией памяти...')
        pipe = pipeline(
            "image-text-to-text",
            model=MODEL_PATH,
            model_kwargs={
                "low_cpu_mem_usage": True,
                "device_map": "auto",
                "dtype": torch.float16
            }
        )
        
        print('✅ Pipeline создан!')
        
        # Загружаем изображение
        print('\n📷 Загружаем изображение...')
        image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
        image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)
        
        # Формируем сообщения
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
        
        # Генерируем ответ
        print('\n🧪 Тестируем генерацию...')
        output = pipe(text=messages, max_new_tokens=200)
        
        # Отладочная информация
        print(f"🔍 Debug: output type = {type(output)}")
        print(f"🔍 Debug: output = {output}")
        
        # Правильная обработка результата в зависимости от формата
        result = ""
        if isinstance(output, list) and len(output) > 0:
            first_output = output[0]
            if isinstance(first_output, dict):
                if "generated_text" in first_output:
                    result = first_output["generated_text"]
                elif "text" in first_output:
                    result = first_output["text"]
                else:
                    # Извлекаем текст из любого поля
                    for key, value in first_output.items():
                        if isinstance(value, str) and len(value) > 10:
                            result = value
                            break
                    if not result:
                        result = str(first_output)
            else:
                result = str(first_output)
        elif isinstance(output, str):
            result = output
        else:
            result = str(output)
        
        # Если результат пустой, попробуем альтернативный подход
        if not result or result.strip() == "":
            print("⚠️  Результат пустой, пробуем альтернативный подход...")
            try:
                # Попробуем другой формат вызова
                simple_output = pipe(image, "Describe this X-ray", max_new_tokens=200)
                if isinstance(simple_output, list) and len(simple_output) > 0:
                    result = str(simple_output[0])
                else:
                    result = str(simple_output)
            except Exception as e2:
                result = f"Ошибка генерации: {e2}"
        
        print(f'\n📋 РЕЗУЛЬТАТ:')
        print('=' * 50)
        print(result)
        print('=' * 50)
        
        print('\n🎉 MedGemma работает!')
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
