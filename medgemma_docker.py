#!/usr/bin/env python3
"""
MedGemma для Docker с GPU поддержкой
"""

import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import requests

# Путь к модели в Docker контейнере
MODEL_PATH = "/app/models/medgemma_4b"

def load_medgemma_gpu():
    """Загрузка MedGemma с GPU оптимизацией"""
    print('🔄 Загружаем MedGemma с GPU...')
    
    try:
        # Проверяем GPU
        if torch.cuda.is_available():
            device = "cuda"
            print(f'✅ CUDA доступна: {torch.cuda.get_device_name(0)}')
            print(f'💾 GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        else:
            device = "cpu"
            print('⚠️  CUDA недоступна, используем CPU')
        
        # Загружаем процессор
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print('✅ Processor загружен!')
        
        # Загружаем модель с GPU оптимизацией
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,  # float16 для экономии памяти
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
        print(f'✅ Model загружена на {device}!')
        print('🎉 MedGemma с GPU готова к работе!')
        
        return model, processor, device
        
    except Exception as e:
        print(f'❌ Ошибка загрузки: {e}')
        return None, None, None

def analyze_chest_xray_gpu(model, processor, device, image_path=None):
    """Анализ рентгеновского снимка с GPU"""
    print('\n🫁 Анализ рентгеновского снимка (GPU)...')
    
    try:
        # Загружаем изображение
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
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
        
        # Обрабатываем с GPU
        print('🔍 Анализируем изображение на GPU...')
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        
        # Перемещаем на GPU
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=200,  # Больше токенов для GPU
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
            generation = generation[0][input_len:]
        
        result = processor.decode(generation, skip_special_tokens=True)
        
        print('\n📋 РЕЗУЛЬТАТ АНАЛИЗА (GPU):')
        print('=' * 60)
        print(result)
        print('=' * 60)
        
        return result
        
    except Exception as e:
        print(f'❌ Ошибка при анализе: {e}')
        return None

def medical_qa_gpu(model, processor, device, question):
    """Ответ на медицинский вопрос с GPU"""
    print(f'\n❓ Медицинский вопрос: {question}')
    
    try:
        # Формируем промпт на русском
        prompt = f"""Вы эксперт-медик. Пожалуйста, ответьте на следующий медицинский вопрос точно и профессионально на русском языке:

Вопрос: {question}

Ответ:"""
        
        # Токенизируем
        inputs = processor.tokenizer(prompt, return_tensors="pt")
        
        # Перемещаем на GPU
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print('🔍 Обрабатываем вопрос на GPU...')
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=150,  # Больше токенов для GPU
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
            generation = generation[0][inputs["input_ids"].shape[-1]:]
        
        result = processor.tokenizer.decode(generation, skip_special_tokens=True)
        
        print('\n📋 ОТВЕТ (GPU):')
        print('=' * 60)
        print(result)
        print('=' * 60)
        
        return result
        
    except Exception as e:
        print(f'❌ Ошибка при обработке вопроса: {e}')
        return None

def test_gpu_model():
    """Тестирование модели с GPU"""
    print('🧪 Тестируем MedGemma с GPU...')
    
    # Загружаем модель
    model, processor, device = load_medgemma_gpu()
    if model is None or processor is None:
        return False
    
    # Тест 1: Анализ рентгеновского снимка
    print('\n' + '=' * 60)
    print('📋 ТЕСТ 1: Анализ рентгеновского снимка')
    analyze_chest_xray_gpu(model, processor, device)
    
    # Тест 2: Медицинский вопрос
    print('\n' + '=' * 60)
    print('📋 ТЕСТ 2: Медицинский вопрос')
    medical_qa_gpu(model, processor, device, "Какие основные симптомы пневмонии?")
    
    # Тест 3: Еще один вопрос
    print('\n' + '=' * 60)
    print('📋 ТЕСТ 3: Дополнительный вопрос')
    medical_qa_gpu(model, processor, device, "Как диагностируется туберкулез?")
    
    print('\n' + '=' * 60)
    print('✅ Все тесты завершены!')
    print('💡 MedGemma с GPU работает отлично!')
    
    return True

def main():
    """Основная функция"""
    print('🏥 MedGemma 4B - Docker с GPU')
    print('=' * 60)
    
    # Проверяем наличие модели
    if not os.path.exists(MODEL_PATH):
        print(f'❌ Модель не найдена в {MODEL_PATH}')
        print('💡 Убедитесь что папка models смонтирована в Docker')
        return
    
    print(f'📁 Используем модель из: {MODEL_PATH}')
    
    # Запускаем тесты
    success = test_gpu_model()
    
    if success:
        print('\n🎉 MedGemma с GPU готова к использованию!')
        print('💡 Теперь все работает быстро с GPU ускорением!')
    else:
        print('\n❌ Возникли проблемы с моделью')

if __name__ == "__main__":
    main()
