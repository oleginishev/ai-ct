#!/usr/bin/env python3
"""
MedGemma - Официальный пример из документации
"""

import os
import torch
from transformers import pipeline
from PIL import Image
import requests

MODEL_PATH = "/app/models/medgemma_4b"

def main():
    """Официальный пример MedGemma"""
    print('🏥 MedGemma 4B - Официальный пример')
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
        # Создаем pipeline точно как в документации
        print('🔄 Создаем pipeline...')
        pipe = pipeline(
            "image-text-to-text",
            model=MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        
        print('✅ Pipeline создан!')
        
        # Загружаем изображение точно как в документации
        print('\n📷 Загружаем изображение...')
        image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
        image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)
        
        # Формируем сообщения точно как в документации
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
        
        # Генерируем ответ точно как в документации
        print('\n🧪 Тестируем генерацию...')
        output = pipe(text=messages, max_new_tokens=200)
        
        # Извлекаем результат точно как в документации
        result = output[0]["generated_text"][-1]["content"]
        
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
