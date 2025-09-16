#!/usr/bin/env python3
"""
MedGemma - Оптимизированная версия для GPU с ограниченной памятью
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests

MODEL_PATH = "/app/models/medgemma_4b"

def main():
    """Оптимизированная версия MedGemma для GPU с ограниченной памятью"""
    print('🏥 MedGemma 4B - Оптимизированная версия')
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
        # Очищаем память
        torch.cuda.empty_cache()
        print('🧹 Очистили память GPU')
    
    try:
        # Загружаем процессор
        print('🔄 Загружаем процессор...')
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print('✅ Процессор загружен!')
        
        # Загружаем модель с максимальной оптимизацией памяти
        print('🔄 Загружаем модель с оптимизацией памяти...')
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,  # float16 для экономии памяти
            device_map="auto",  # Автоматическое распределение
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # Дополнительные параметры для экономии памяти
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        )
        print('✅ Модель загружена!')
        
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
        
        # Обрабатываем с оптимизацией памяти
        print('\n🧪 Тестируем генерацию...')
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        
        # Перемещаем на GPU с float16
        if device == "cuda":
            inputs = {k: v.to(device, dtype=torch.float16) for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Генерируем с оптимизацией памяти
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=150,  # Уменьшаем количество токенов
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                # Дополнительные параметры для экономии памяти
                output_attentions=False,
                output_hidden_states=False,
            )
            generation = generation[0][input_len:]
        
        result = processor.decode(generation, skip_special_tokens=True)
        
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
