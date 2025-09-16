#!/usr/bin/env python3
"""
Пример использования MedGemma 4B для медицинских задач
"""

import os
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import requests

def test_medgemma_loading():
    """Тестирование загрузки MedGemma"""
    print('🔄 Тестируем загрузку MedGemma...')
    
    try:
        # Проверяем токен Hugging Face
        if not os.getenv('HUGGINGFACE_HUB_TOKEN'):
            print('⚠️  Токен Hugging Face не найден!')
            print('🔧 Выполните: huggingface-cli login')
            print('📝 Или установите: export HUGGINGFACE_HUB_TOKEN=your_token')
            return False
            
        processor = AutoProcessor.from_pretrained('google/medgemma-4b-it')
        print('✅ Processor загружен успешно!')
        
        model = AutoModelForImageTextToText.from_pretrained(
            'google/medgemma-4b-it',
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        print('✅ Model загружена успешно!')
        print('🎉 MedGemma готова к работе!')
        return True
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        print('🔧 Возможные решения:')
        print('1. Проверьте токен: echo $HUGGINGFACE_HUB_TOKEN')
        print('2. Выполните: huggingface-cli login')
        print('3. Примите условия на: https://huggingface.co/google/medgemma-4b-it')
        return False

def analyze_chest_xray(image_path=None, image_url=None):
    """Анализ рентгеновского снимка грудной клетки"""
    print('\n🫁 Анализ рентгеновского снимка грудной клетки...')
    
    try:
        # Создаем pipeline
        pipe = pipeline(
            "image-text-to-text",
            model="google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device="auto",
        )
        
        # Загружаем изображение
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
            print(f'📁 Загружено изображение: {image_path}')
        elif image_url:
            image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)
            print(f'🌐 Загружено изображение с URL: {image_url}')
        else:
            # Используем пример изображения
            image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
            image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)
            print('📷 Используем пример изображения из интернета')
        
        # Формируем сообщения для анализа на русском
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
        
        # Получаем анализ
        print('🔍 Анализируем изображение...')
        output = pipe(text=messages, max_new_tokens=300)
        result = output[0]["generated_text"][-1]["content"]
        
        print('\n📋 РЕЗУЛЬТАТ АНАЛИЗА:')
        print('=' * 50)
        print(result)
        print('=' * 50)
        
        return result
        
    except Exception as e:
        print(f'❌ Ошибка при анализе: {e}')
        return None

def medical_qa(question):
    """Ответ на медицинский вопрос"""
    print(f'\n❓ Медицинский вопрос: {question}')
    
    try:
        # Создаем pipeline для текстовых вопросов
        pipe = pipeline(
            "text-generation",
            model="google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device="auto",
        )
        
        # Формируем промпт на русском
        prompt = f"""Вы эксперт-медик. Пожалуйста, ответьте на следующий медицинский вопрос точно и профессионально на русском языке:

Вопрос: {question}

Ответ:"""
        
        print('🔍 Обрабатываем вопрос...')
        output = pipe(prompt, max_new_tokens=200, do_sample=False)
        result = output[0]["generated_text"][len(prompt):].strip()
        
        print('\n📋 ОТВЕТ:')
        print('=' * 50)
        print(result)
        print('=' * 50)
        
        return result
        
    except Exception as e:
        print(f'❌ Ошибка при обработке вопроса: {e}')
        return None

def main():
    """Основная функция"""
    print('🏥 MedGemma 4B - Пример использования')
    print('=' * 50)
    
    # Тестируем загрузку
    if not test_medgemma_loading():
        return
    
    print('\n' + '=' * 50)
    print('🎯 ВОЗМОЖНОСТИ MEDGEMMA:')
    print('1. Анализ медицинских изображений (рентген, КТ, МРТ)')
    print('2. Ответы на медицинские вопросы')
    print('3. Генерация медицинских отчетов')
    print('4. Визуальный вопросно-ответный анализ')
    print('5. Классификация медицинских изображений')
    
    # Пример 1: Анализ рентгеновского снимка
    print('\n' + '=' * 50)
    print('📋 ПРИМЕР 1: Анализ рентгеновского снимка')
    analyze_chest_xray()
    
    # Пример 2: Медицинский вопрос
    print('\n' + '=' * 50)
    print('📋 ПРИМЕР 2: Медицинский вопрос')
    medical_qa("Какие основные симптомы пневмонии?")
    
    print('\n' + '=' * 50)
    print('✅ Примеры завершены!')
    print('💡 MedGemma готова к использованию для ваших медицинских задач!')

if __name__ == "__main__":
    main()
