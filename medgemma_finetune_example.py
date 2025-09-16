#!/usr/bin/env python3
"""
Пример fine-tuning MedGemma 4B для специфических медицинских задач
"""

import torch
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json

def prepare_training_data():
    """Подготовка данных для обучения"""
    print('📊 Подготовка данных для fine-tuning...')
    
    # Пример данных для обучения (замените на ваши данные)
    training_data = [
        {
            "image": "path/to/chest_xray_1.jpg",
            "text": "Normal chest X-ray with clear lung fields and normal cardiac silhouette.",
            "label": "normal"
        },
        {
            "image": "path/to/chest_xray_2.jpg", 
            "text": "Chest X-ray shows bilateral lower lobe consolidation consistent with pneumonia.",
            "label": "pneumonia"
        },
        {
            "image": "path/to/chest_xray_3.jpg",
            "text": "Chest X-ray demonstrates left upper lobe opacity concerning for tuberculosis.",
            "label": "tuberculosis"
        }
    ]
    
    return training_data

def create_dataset(training_data):
    """Создание датасета для обучения"""
    print('🗂️ Создание датасета...')
    
    # Преобразуем данные в формат для Hugging Face
    dataset_dict = {
        "text": [item["text"] for item in training_data],
        "label": [item["label"] for item in training_data]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

def setup_model_and_processor():
    """Настройка модели и процессора"""
    print('🔧 Настройка модели и процессора...')
    
    model_id = "google/medgemma-4b-it"
    
    # Загружаем модель и процессор
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return model, processor

def fine_tune_model(model, processor, dataset):
    """Fine-tuning модели"""
    print('🎯 Начинаем fine-tuning...')
    
    # Настройки обучения
    training_args = TrainingArguments(
        output_dir="./medgemma-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Малый batch size для экономии памяти
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=False
    )
    
    # Создаем тренер
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Запускаем обучение
    print('🚀 Запускаем обучение...')
    trainer.train()
    
    # Сохраняем модель
    print('💾 Сохраняем fine-tuned модель...')
    trainer.save_model()
    processor.save_pretrained("./medgemma-finetuned")
    
    return trainer

def evaluate_model(model, processor, test_data):
    """Оценка модели"""
    print('📊 Оценка модели...')
    
    # Здесь можно добавить код для оценки модели
    # на тестовых данных
    
    pass

def main():
    """Основная функция для fine-tuning"""
    print('🎓 MedGemma 4B - Fine-tuning Example')
    print('=' * 50)
    
    print('⚠️  ВНИМАНИЕ: Fine-tuning требует значительных ресурсов!')
    print('💻 Рекомендуется: GPU с минимум 16GB VRAM')
    print('⏱️  Время обучения: несколько часов')
    
    # Проверяем доступность GPU
    if not torch.cuda.is_available():
        print('❌ CUDA не доступна! Fine-tuning может быть очень медленным.')
        response = input('Продолжить? (y/n): ')
        if response.lower() != 'y':
            return
    
    try:
        # 1. Подготавливаем данные
        training_data = prepare_training_data()
        
        # 2. Создаем датасет
        dataset = create_dataset(training_data)
        
        # 3. Настраиваем модель
        model, processor = setup_model_and_processor()
        
        # 4. Fine-tuning
        trainer = fine_tune_model(model, processor, dataset)
        
        print('✅ Fine-tuning завершен!')
        print('📁 Модель сохранена в: ./medgemma-finetuned')
        
    except Exception as e:
        print(f'❌ Ошибка при fine-tuning: {e}')
        print('💡 Убедитесь, что у вас достаточно ресурсов и правильные данные')

if __name__ == "__main__":
    main()
