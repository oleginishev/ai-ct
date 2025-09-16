#!/usr/bin/env python3
"""
MedGemma с русскими ответами - оптимизированная версия для MacBook Air M3
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODEL_PATH = "./models/medgemma_4b"

def load_model_russian():
    """Загрузка модели для русских ответов"""
    print('🇷🇺 Загружаем MedGemma для русских ответов...')
    
    try:
        # Определяем устройство
        if torch.backends.mps.is_available():
            device = "mps"
            print('✅ Используем MPS (Metal) ускорение')
        else:
            device = "cpu"
            print('⚠️  Используем CPU')
        
        # Загружаем модель
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        model = model.to(device)
        
        print(f'✅ Модель загружена на {device}')
        return model, tokenizer, device
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        return None, None, None

def ask_medical_question(model, tokenizer, device, question):
    """Задать медицинский вопрос на русском"""
    print(f'\n❓ Вопрос: {question}')
    
    start_time = time.time()
    
    try:
        # Промпт для русских ответов
        prompt = f"""Вы опытный врач-эксперт. Ответьте на медицинский вопрос профессионально и понятно на русском языке.

Вопрос: {question}

Ответ врача:"""
        
        # Токенизация
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Генерация ответа
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,  # Достаточно для развернутого ответа
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Декодирование
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f'⚡ ОТВЕТ (за {processing_time:.1f} сек):')
        print('=' * 60)
        print(answer)
        print('=' * 60)
        
        return answer, processing_time
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        return None, 0

def interactive_russian_qa():
    """Интерактивный режим на русском"""
    print('\n🇷🇺 Интерактивный режим медицинских вопросов на русском')
    print('Введите "выход" для завершения')
    
    model, tokenizer, device = load_model_russian()
    if model is None:
        return
    
    print('\n💡 Примеры вопросов:')
    print('   - Какие симптомы у пневмонии?')
    print('   - Как лечить простуду?')
    print('   - Что такое гипертония?')
    print('   - Признаки инфаркта?')
    
    while True:
        question = input('\n❓ Ваш медицинский вопрос: ').strip()
        
        if question.lower() in ['выход', 'exit', 'quit', 'стоп']:
            print('👋 До свидания!')
            break
        
        if not question:
            continue
        
        ask_medical_question(model, tokenizer, device, question)

def demo_russian_questions():
    """Демонстрация русских вопросов"""
    print('🚀 Демонстрация MedGemma на русском языке...')
    
    model, tokenizer, device = load_model_russian()
    if model is None:
        return
    
    # Популярные медицинские вопросы на русском
    questions = [
        "Какие основные симптомы пневмонии?",
        "Как лечить простуду в домашних условиях?",
        "Что такое гипертония и как ее контролировать?",
        "Какие признаки инфаркта миокарда?",
        "Как правильно измерять артериальное давление?",
        "Что такое диабет и его типы?",
        "Как оказать первую помощь при ожогах?",
        "Какие симптомы гриппа отличают его от простуды?"
    ]
    
    times = []
    for i, question in enumerate(questions, 1):
        print(f'\n--- Вопрос {i}/{len(questions)} ---')
        answer, time_taken = ask_medical_question(model, tokenizer, device, question)
        if time_taken > 0:
            times.append(time_taken)
        
        # Небольшая пауза между вопросами
        if i < len(questions):
            time.sleep(1)
    
    # Статистика
    if times:
        avg_time = sum(times) / len(times)
        print(f'\n📊 СТАТИСТИКА:')
        print(f'   Всего вопросов: {len(times)}')
        print(f'   Среднее время: {avg_time:.1f} сек')
        print(f'   Минимальное: {min(times):.1f} сек')
        print(f'   Максимальное: {max(times):.1f} сек')
        
        if avg_time < 8:
            print('⚡ Отличная скорость!')
        elif avg_time < 20:
            print('✅ Хорошая скорость')
        else:
            print('🐌 Медленно, но работает')

def main():
    """Основная функция"""
    print('🇷🇺 MedGemma - Русские медицинские ответы')
    print('=' * 60)
    print('🍎 Оптимизировано для MacBook Air M3')
    
    if not os.path.exists(MODEL_PATH):
        print(f'❌ Модель не найдена в {MODEL_PATH}')
        print('💡 Скачайте модель: huggingface-cli download google/medgemma-4b-it --local-dir models/medgemma_4b')
        return
    
    print('\nВыберите режим:')
    print('1. Демонстрация (8 примеров вопросов)')
    print('2. Интерактивный режим')
    
    choice = input('\nВаш выбор (1/2): ').strip()
    
    if choice == '1':
        demo_russian_questions()
    elif choice == '2':
        interactive_russian_qa()
    else:
        print('Запускаем демонстрацию...')
        demo_russian_questions()

if __name__ == "__main__":
    main()
