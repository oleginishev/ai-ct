#!/usr/bin/env python3
"""
Простая проверка MedGemma в Docker
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = "/app/models/medgemma_4b"

def main():
    """Простая проверка"""
    print('🏥 MedGemma 4B - Простая проверка')
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
        # Загружаем модель
        print('🔄 Загружаем модель...')
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        print('✅ Модель загружена!')
        
        # Простой тест
        print('\n🧪 Тестируем генерацию...')
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello! How are you?"}]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[-1]
        print(f'📏 Длина входа: {input_len} токенов')
        
        # Проверяем входные токены
        input_tokens = inputs["input_ids"][0].tolist()
        print(f'🔍 Входные токены: {input_tokens}')
        print(f'🔍 Входной текст: "{processor.decode(input_tokens, skip_special_tokens=False)}"')
        
        # Проверяем специальные токены
        print(f'🔍 EOS token: {processor.tokenizer.eos_token_id}')
        print(f'🔍 PAD token: {processor.tokenizer.pad_token_id}')
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            generation = generation[0][input_len:]
        
        print(f'📏 Сгенерировано: {len(generation)} токенов')
        
        # Проверяем что генерируется
        print(f'🔍 Первые 10 токенов: {generation[:10].tolist()}')
        
        # Декодируем без пропуска специальных токенов
        result_raw = processor.decode(generation, skip_special_tokens=False)
        print(f'📝 Сырой результат (с токенами): "{result_raw}"')
        
        # Декодируем с пропуском специальных токенов
        result = processor.decode(generation, skip_special_tokens=True)
        print(f'📝 Очищенный результат: "{result}"')
        
        # Если результат пустой, пробуем другой способ
        if not result.strip():
            result = processor.tokenizer.decode(generation, skip_special_tokens=True)
            print(f'📝 Альтернативный результат: "{result}"')
        
        print(f'\n📋 РЕЗУЛЬТАТ:')
        print('=' * 40)
        print(result)
        print('=' * 40)
        
        print('\n🎉 MedGemma работает!')
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
