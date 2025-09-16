#!/usr/bin/env python3
"""
Очень легкая версия MedGemma для MacBook Air M3
Только текстовые вопросы - максимальная скорость
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODEL_PATH = "./models/medgemma_4b"

def load_light_model_m3():
    """Загрузка только текстовой части модели"""
    print('🍎 Загружаем легкую версию для M3...')
    
    try:
        # Используем только текстовую модель (без vision)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f'🖥️  Устройство: {device}')
        
        # Загружаем только токенизатор и текстовую модель
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        model = model.to(device)
        
        print('✅ Легкая модель загружена!')
        return model, tokenizer, device
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        return None, None, None

def fast_medical_qa(model, tokenizer, device, question):
    """Очень быстрые медицинские вопросы"""
    print(f'\n⚡ Быстрый вопрос: {question}')
    
    start_time = time.time()
    
    try:
        # Очень простой промпт на русском
        prompt = f"Медицинский вопрос: {question}\nОтвет на русском языке:"
        
        # Токенизация
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Быстрая генерация
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,  # Очень короткий ответ
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
        print('=' * 40)
        print(answer)
        print('=' * 40)
        
        return answer, processing_time
        
    except Exception as e:
        print(f'❌ Ошибка: {e}')
        return None, 0

def interactive_qa():
    """Интерактивный режим вопросов"""
    print('\n🎯 Интерактивный режим медицинских вопросов')
    print('Введите "quit" для выхода')
    
    model, tokenizer, device = load_light_model_m3()
    if model is None:
        return
    
    while True:
        question = input('\n❓ Ваш вопрос: ').strip()
        
        if question.lower() in ['quit', 'exit', 'выход']:
            break
        
        if not question:
            continue
        
        fast_medical_qa(model, tokenizer, device, question)

def quick_demo():
    """Быстрая демонстрация"""
    print('🚀 Быстрая демонстрация для M3...')
    
    model, tokenizer, device = load_light_model_m3()
    if model is None:
        return
    
    # Быстрые вопросы на русском
    questions = [
        "Что такое пневмония?",
        "Симптомы гриппа?",
        "Как лечить температуру?",
        "Что такое диабет?",
        "Признаки инфаркта?"
    ]
    
    times = []
    for i, q in enumerate(questions, 1):
        print(f'\n--- Вопрос {i}/{len(questions)} ---')
        answer, time_taken = fast_medical_qa(model, tokenizer, device, q)
        if time_taken > 0:
            times.append(time_taken)
    
    if times:
        avg_time = sum(times) / len(times)
        print(f'\n📊 СТАТИСТИКА:')
        print(f'   Среднее время: {avg_time:.1f} сек')
        print(f'   Всего вопросов: {len(times)}')
        
        if avg_time < 5:
            print('⚡ Отличная скорость!')
        elif avg_time < 15:
            print('✅ Хорошая скорость')
        else:
            print('🐌 Медленно для M3')

def main():
    """Основная функция"""
    print('🍎 MedGemma Light - MacBook Air M3')
    print('=' * 50)
    print('💡 Только текстовые вопросы для максимальной скорости')
    
    if not os.path.exists(MODEL_PATH):
        print(f'❌ Модель не найдена в {MODEL_PATH}')
        return
    
    print('\nВыберите режим:')
    print('1. Быстрая демонстрация')
    print('2. Интерактивные вопросы')
    
    choice = input('\nВаш выбор (1/2): ').strip()
    
    if choice == '1':
        quick_demo()
    elif choice == '2':
        interactive_qa()
    else:
        print('Запускаем демонстрацию...')
        quick_demo()

if __name__ == "__main__":
    main()
