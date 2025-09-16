#!/usr/bin/env python3

import os
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

print('🔄 Тестируем загрузку MedGemma...')

try:
    processor = AutoProcessor.from_pretrained('google/medgemma-4b-it')
    print('✅ Processor загружен успешно!')
    
    model = AutoModelForImageTextToText.from_pretrained(
        'google/medgemma-4b-it',
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    print('✅ Model загружена успешно!')
    print('🎉 MedGemma готова к работе!')
    
except Exception as e:
    print(f'❌ Ошибка: {e}')
    print('🔧 Возможные решения:')
    print('1. Проверьте токен: echo $HUGGINGFACE_HUB_TOKEN')
    print('2. Выполните: huggingface-cli login')
    print('3. Примите условия на: https://huggingface.co/google/medgemma-4b-it')
