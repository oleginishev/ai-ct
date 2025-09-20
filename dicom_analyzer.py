#!/usr/bin/env python3
"""
DICOM Analyzer Script (Simplified)
==================================

Простой скрипт для анализа DICOM файлов с использованием MedGemma.
Анализирует все DICOM файлы в указанной папке и выводит общий анализ.

Требования:
- transformers
- torch
- pydicom
- PIL
- numpy
- tqdm (для прогресс-бара)
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
import pydicom
from PIL import Image
import torch
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===== НАСТРОЙКИ =====
# Путь к папке с DICOM файлами (измените на свой путь)
DICOM_FOLDER_PATH = "data"  # По умолчанию папка data

# Debug режим - анализировать только первые N изображений
DEBUG_MODE = False  # Установите True для debug режима
DEBUG_LIMIT = 50    # Количество файлов для анализа в debug режиме

# Доступные модели MedGemma
AVAILABLE_MODELS = {
    "4b": "google/medgemma-4b-it",
    "27b": "google/medgemma-27b-it"
}
DEFAULT_MODEL = "4b"  # По умолчанию используем 4B модель

# ===== НАСТРОЙКИ АНАЛИЗА СНИМКОВ =====
# Параметры CT windowing по умолчанию
DEFAULT_WINDOW_LEVEL = -550   # WL (центр окна)
DEFAULT_WINDOW_WIDTH = 1600   # WW (ширина окна)

# Альтернативные окна для лучшего выявления пневмонии
PNEUMONIA_DETECTION_WINDOWS = {
    "lung_soft": {"wl": -400, "ww": 1400},      # Мягкое легочное окно
    "infection": {"wl": -300, "ww": 1200},      # Окно для инфекций
    "standard_lung": {"wl": -600, "ww": 1600}   # Стандартное легочное
}

# Языковые настройки для промптов
LANGUAGE_PROMPTS = {
    "en": """You are an expert radiologist and pulmonologist with extensive experience in chest CT analysis. 

Analyze this medical image and provide a detailed assessment focusing on:
1. Lung pathology: pneumonia, consolidations, ground-glass opacities
2. Pleural changes: effusions, thickening
3. Airways: air bronchograms, bronchial wall thickening  
4. Overall lung architecture and any abnormalities
5. Clinical recommendations

Be thorough but concise. Report both normal and abnormal findings. If any concerning features are present, describe their location and characteristics in detail.""",
    
    "ru": """Вы - эксперт-рентгенолог и пульмонолог с обширным опытом анализа КТ грудной клетки.

Проанализируйте это медицинское изображение и предоставьте детальную оценку, сосредоточившись на:
1. Патологии легких: пневмония, консолидации, изменения по типу "матового стекла"
2. Плевральные изменения: выпот, утолщение
3. Дыхательные пути: воздушные бронхограммы, утолщение стенок бронхов
4. Общая архитектура легких и любые аномалии
5. Клинические рекомендации

Будьте тщательными, но лаконичными. Сообщайте как о нормальных, так и о патологических находках. Если присутствуют тревожные признаки, опишите их локализацию и характеристики подробно."""
}

# Единый универсальный промпт для всех случаев анализа (по умолчанию английский)
DEFAULT_ANALYSIS_PROMPT = LANGUAGE_PROMPTS["en"]
CURRENT_LANGUAGE = "en"

# Промпты для анализа (можно изменить для разных специализаций)
ANALYSIS_PROMPTS = {
    # Системный промпт - определяет роль эксперта
    "system": "You are an expert pulmonologist and chest radiologist with extensive experience in detecting pneumonia, COVID-19, and other infectious lung diseases.",
    
    # Единый промпт для всех типов анализа
    "universal": DEFAULT_ANALYSIS_PROMPT,
    
    # Промпт для анализа отдельного изображения
    "single_image": DEFAULT_ANALYSIS_PROMPT,
    
    # Промпт для батчевого анализа (краткий)
    "batch_analysis": DEFAULT_ANALYSIS_PROMPT,
    
    # Промпт для общего отчета по серии
    "series_report": "Based on analysis of {count} chest CT images, create a comprehensive pulmonary radiological report. Here are the individual analyses:\n\n{analyses}\n\nProvide a definitive assessment with clinical recommendations.",
    
    # Системный промпт для общего отчета
    "series_system": "You are an expert pulmonologist and chest radiologist specializing in pneumonia detection and infectious lung disease."
}

# Параметры генерации текста
GENERATION_PARAMS = {
    "single_image_tokens": 300,    # Увеличено для детального анализа пневмонии
    "batch_tokens": 200,           # Увеличено для лучшего выявления патологий
    "series_report_tokens": 500    # Увеличено для комплексного отчета
}

# Настройки обработки изображений
IMAGE_PROCESSING = {
    "convert_to_hu": True,         # Конвертировать в Hounsfield Units
    "apply_windowing": True,       # Применять медицинское окно
    "default_modality": "CT",      # Модальность по умолчанию
    "fallback_normalization": True # Использовать min-max если не CT
}

# Настройки производительности и батчей
PERFORMANCE_SETTINGS = {
    "batch_size_cuda": 8,          # Размер батча для CUDA GPU (RTX A5000: 24GB)
    "batch_size_cpu": 1,           # Размер батча для CPU
    "auto_batch_size": True,       # Автоматически определять размер батча
    "max_batch_size": 16,          # Максимальный размер батча
    "memory_safety_margin": 0.8    # Коэффициент безопасности памяти (80%)
}

class DICOMAnalyzer:
    """Класс для анализа DICOM файлов с помощью MedGemma"""
    
    def __init__(self, model_name="4b", window_level=DEFAULT_WINDOW_LEVEL, window_width=DEFAULT_WINDOW_WIDTH, batch_size=None):
        """Инициализация анализатора"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {self.device}")
        
        # Сохраняем параметры windowing
        self.window_level = window_level
        self.window_width = window_width
        print(f"Параметры CT окна: WL={self.window_level}, WW={self.window_width}")
        
        # Настройка размера батча
        if batch_size is not None:
            self.batch_size = batch_size
            print(f"🔧 Пользовательский размер батча: {self.batch_size}")
        else:
            self.batch_size = PERFORMANCE_SETTINGS["batch_size_cuda"] if self.device == "cuda" else PERFORMANCE_SETTINGS["batch_size_cpu"]
            print(f"⚙️  Автоматический размер батча: {self.batch_size}")
        
        # Выбор модели
        if model_name not in AVAILABLE_MODELS:
            print(f"❌ ОШИБКА: Неизвестная модель '{model_name}'")
            print(f"📋 Доступные модели: {list(AVAILABLE_MODELS.keys())}")
            print(f"💡 Используйте: --model=4b или --model=27b")
            raise ValueError(f"Модель '{model_name}' не поддерживается. Доступны: {list(AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_path = AVAILABLE_MODELS[model_name]
        
        # Проверяем, что это MedGemma модель
        if "medgemma" not in self.model_path.lower():
            raise ValueError(f"Поддерживаются только модели MedGemma. Получена: {self.model_path}")
        
        # Инициализация MedGemma pipeline
        print(f"Загружаем MedGemma модель: {self.model_path}")
        
        try:
            # Попробуем загрузить с trust_remote_code
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device=self.device,
                trust_remote_code=True,
                use_fast=False
            )
        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель {self.model_path}")
            print(f"Детали ошибки: {e}")
            print("\n🔧 ВОЗМОЖНЫЕ РЕШЕНИЯ:")
            print("1. Проверьте токен Hugging Face: export HF_TOKEN='your_token'")
            print("2. Запросите доступ к модели: https://huggingface.co/google/medgemma-4b-it")
            print("3. Проверьте интернет-соединение")
            print("4. Убедитесь, что установлены все зависимости: pip install -r requirements_dicom_analyzer.txt")
            
            # Пробуем альтернативный способ загрузки ТОЛЬКО для MedGemma
            print("\n🔄 Пробуем альтернативный способ загрузки MedGemma...")
            
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                self.pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device
                )
                print("✅ MedGemma модель загружена через альтернативный метод")
                
            except Exception as e2:
                print(f"❌ ФАТАЛЬНАЯ ОШИБКА: Невозможно загрузить MedGemma модель")
                print(f"Альтернативная ошибка: {e2}")
                print("\n💀 АНАЛИЗ ОСТАНОВЛЕН. Исправьте проблемы с моделью и повторите запуск.")
                print("📚 Документация: https://huggingface.co/google/medgemma-4b-it")
                raise RuntimeError(f"Не удалось загрузить модель {self.model_path}. Проверьте токен доступа и права.")
        print(f"MedGemma {model_name.upper()} модель загружена успешно!")
        print("Примечание: Предупреждение о 'pipelines sequentially on GPU' - это нормально, мы используем батчи для оптимизации.")
        
        # Результаты анализа
        self.results = []
        
    def apply_ct_windowing(self, image_array, window_center=40, window_width=400):
        """
        Применение окна для CT изображений (стандартный подход в радиологии)
        
        Args:
            image_array (numpy.ndarray): Исходный массив пикселей в HU
            window_center (int): Центр окна (по умолчанию 40 HU для мягких тканей)
            window_width (int): Ширина окна (по умолчанию 400 HU)
            
        Returns:
            numpy.ndarray: Массив пикселей 0-255
        """
        min_hu = window_center - window_width // 2
        max_hu = window_center + window_width // 2
        
        # Обрезаем значения по окну
        windowed = np.clip(image_array, min_hu, max_hu)
        
        # Нормализуем к диапазону 0-255
        if max_hu > min_hu:
            windowed = ((windowed - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
        else:
            windowed = np.zeros_like(windowed, dtype=np.uint8)
        
        return windowed
    
    def load_dicom_as_image(self, dicom_path, silent=False):
        """
        Загрузка DICOM файла и конвертация в PIL Image с правильным CT windowing
        
        Args:
            dicom_path (str): Путь к DICOM файлу
            silent (bool): Не выводить информацию о параметрах (для батчевой обработки)
            
        Returns:
            PIL.Image: Изображение или None при ошибке
        """
        try:
            # Загрузка DICOM файла
            dicom = pydicom.dcmread(dicom_path)
            image_array = dicom.pixel_array.astype(np.float32)
            
            # Применение Rescale Slope и Intercept (конвертация в HU единицы)
            if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
                slope = float(dicom.RescaleSlope)
                intercept = float(dicom.RescaleIntercept)
                image_array = image_array * slope + intercept
                if not silent:
                    print(f"Применены DICOM параметры: Slope={slope}, Intercept={intercept}")
            
            # Определение типа исследования для выбора окна
            modality = getattr(dicom, 'Modality', 'CT')
            body_part = getattr(dicom, 'BodyPartExamined', '').upper()
            
            # Выбор окна в зависимости от типа исследования
            if modality == 'CT':
                # Используем параметры, переданные при инициализации анализатора
                window_center, window_width = self.window_level, self.window_width
                if not silent:
                    print(f"Применено CT окно: WL={window_center}, WW={window_width}")
                
                # Дополнительная логика для специфических частей тела (опционально)
                if 'BRAIN' in body_part or 'HEAD' in body_part:
                    # Мозговое окно (можно переопределить, если нужно)
                    # window_center, window_width = 40, 80
                    # print(f"Переопределено мозговое окно: WL={window_center}, WW={window_width}")
                    pass
                elif 'BONE' in body_part:
                    # Костное окно (можно переопределить, если нужно)  
                    # window_center, window_width = 400, 1800
                    # print(f"Переопределено костное окно: WL={window_center}, WW={window_width}")
                    pass
                
                # Информация о применяемом окне
                min_hu = window_center - window_width // 2
                max_hu = window_center + window_width // 2
                if not silent:
                    print(f"Диапазон HU: {min_hu} до {max_hu} (оптимизировано для анализа легких)")
                if not silent:
                    print(f"Анатомическая область: {body_part or 'CHEST/LUNG (по умолчанию)'}")
            else:
                # Для других модальностей (X-ray и т.д.) используем простую нормализацию
                if image_array.max() > image_array.min():
                    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.zeros_like(image_array, dtype=np.uint8)
                if not silent:
                    print(f"Применена min-max нормализация для модальности: {modality}")
                
                # Конвертация в PIL Image
                image = Image.fromarray(image_array, mode='L')
                image_rgb = Image.new('RGB', image.size)
                image_rgb.paste(image)
                return image_rgb
            
            # Применение CT windowing
            windowed_array = self.apply_ct_windowing(image_array, window_center, window_width)
            
            # Конвертация в PIL Image
            image = Image.fromarray(windowed_array, mode='L')
            
            # Конвертация в RGB (MedGemma ожидает RGB)
            image_rgb = Image.new('RGB', image.size)
            image_rgb.paste(image)
            
            return image_rgb
            
        except Exception as e:
            print(f"Ошибка при загрузке {dicom_path}: {e}")
            return None
    
    def analyze_image(self, image, file_path):
        """
        Анализ изображения с помощью MedGemma
        
        Args:
            image (PIL.Image): Изображение для анализа
            file_path (str): Путь к файлу
            
        Returns:
            dict: Результаты анализа
        """
        try:
            # Адаптивный анализ в зависимости от типа модели
            if "medgemma" in self.model_path.lower():
                # Создание сообщения для MedGemma
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": ANALYSIS_PROMPTS["system"]}]
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": ANALYSIS_PROMPTS["single_image"]},
                            {"type": "image", "image": image}
                        ]
                    }
                ]
                
                # Получение ответа от MedGemma
                try:
                    output = self.pipe(text=messages, max_new_tokens=GENERATION_PARAMS["single_image_tokens"])
                    analysis_text = output[0]["generated_text"][-1]["content"]
                except Exception as e:
                    print(f"Ошибка обработки MedGemma: {e}")
                    # Fallback для MedGemma
                    prompt = f"{ANALYSIS_PROMPTS['system']} {ANALYSIS_PROMPTS['single_image']}"
                    output = self.pipe(prompt, max_new_tokens=GENERATION_PARAMS["single_image_tokens"])
                    analysis_text = output[0]["generated_text"]
            else:
                # Ошибка: должна быть только MedGemma модель
                raise ValueError(f"Неподдерживаемая модель: {self.model_path}. Поддерживаются только модели MedGemma.")
            
            return {
                'file_path': file_path,
                'analysis': analysis_text,
                'file_name': os.path.basename(file_path)
            }
            
        except Exception as e:
            print(f"Ошибка при анализе {file_path}: {e}")
            return {
                'file_path': file_path,
                'analysis': f'Ошибка анализа: {str(e)}',
                'file_name': os.path.basename(file_path)
            }
    
    def analyze_all_images(self, images, file_paths):
        """
        Анализ всех изображений сразу
        
        Args:
            images (list): Список PIL изображений
            file_paths (list): Список путей к файлам
            
        Returns:
            dict: Результаты анализа
        """
        try:
            # Создаем коллаж из всех изображений (простой способ - берем первое изображение)
            # MedGemma может анализировать только одно изображение за раз
            # Поэтому будем анализировать первое изображение как представителя всех
            
            if not images:
                return {
                    'file_paths': file_paths,
                    'analysis': 'Нет изображений для анализа',
                    'total_files': 0
                }
            
            # Берем первое изображение для анализа
            main_image = images[0]
            
            # Создание сообщения для MedGemma (использует настройки из блока конфигурации)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": ANALYSIS_PROMPTS["series_system"]}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": f"Analyze this chest CT image from a series of {len(images)} DICOM files. Focus on lung pathology and provide detailed assessment of this representative image with general recommendations for the entire series."},
                        {"type": "image", "image": main_image}
                    ]
                }
            ]
            
            # Получение ответа от MedGemma
            output = self.pipe(text=messages, max_new_tokens=300)
            analysis_text = output[0]["generated_text"][-1]["content"]
            
            return {
                'file_paths': file_paths,
                'analysis': analysis_text,
                'total_files': len(images),
                'analyzed_image': os.path.basename(file_paths[0]) if file_paths else "unknown"
            }
            
        except Exception as e:
            print(f"Ошибка при анализе всех изображений: {e}")
            return {
                'file_paths': file_paths,
                'analysis': f'Ошибка анализа: {str(e)}',
                'total_files': len(images),
                'analyzed_image': "unknown"
            }
    
    def create_combined_analysis(self, analyses, file_paths):
        """
        Создание общего анализа на основе всех индивидуальных анализов
        
        Args:
            analyses (list): Список анализов каждого изображения
            file_paths (list): Список путей к файлам
            
        Returns:
            dict: Объединенный результат анализа
        """
        try:
            # Объединяем все анализы в один текст
            combined_text = "\n\n".join([f"Analysis {i+1}: {analysis}" for i, analysis in enumerate(analyses)])
            
            # Создаем сообщение для MedGemma для создания общего анализа
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": ANALYSIS_PROMPTS["series_system"]}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": ANALYSIS_PROMPTS["series_report"].format(count=len(analyses), analyses=combined_text)}
                    ]
                }
            ]
            
            # Получение общего анализа от MedGemma
            output = self.pipe(text=messages, max_new_tokens=GENERATION_PARAMS["series_report_tokens"])
            combined_analysis = output[0]["generated_text"][-1]["content"]
            
            return {
                'file_paths': file_paths,
                'analysis': combined_analysis,
                'total_files': len(analyses),
                'individual_analyses': analyses
            }
            
        except Exception as e:
            print(f"Ошибка при создании общего анализа: {e}")
            # Если не удалось создать общий анализ, просто объединяем тексты
            combined_text = "\n\n".join([f"Analysis {i+1}: {analysis}" for i, analysis in enumerate(analyses)])
            return {
                'file_paths': file_paths,
                'analysis': f"Combined Analysis (Individual Reports):\n\n{combined_text}",
                'total_files': len(analyses),
                'individual_analyses': analyses
            }
    
    def analyze_batch(self, images, file_paths):
        """
        Анализ батча изображений
        
        Args:
            images (list): Список PIL изображений
            file_paths (list): Список путей к файлам
            
        Returns:
            list: Список анализов
        """
        batch_analyses = []
        total_images = len(images)
        
        print(f"⚡ Обрабатываем {total_images} изображений батчами по {self.batch_size}...")
        
        # Обрабатываем изображения батчами
        for i in range(0, total_images, self.batch_size):
            batch_end = min(i + self.batch_size, total_images)
            current_batch_size = batch_end - i
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_images + self.batch_size - 1) // self.batch_size
            
            print(f"🔄 Батч {batch_num}/{total_batches}: обрабатываем {current_batch_size} изображений...")
            
            batch_images = images[i:batch_end]
            batch_paths = file_paths[i:batch_end]
            
            for j, (image, file_path) in enumerate(zip(batch_images, batch_paths)):
                try:
                    # Создание сообщения для MedGemma
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": ANALYSIS_PROMPTS["system"]}]
                        },
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": ANALYSIS_PROMPTS["batch_analysis"]},
                                {"type": "image", "image": image}
                            ]
                        }
                    ]
                    
                    # Получение ответа от MedGemma
                    output = self.pipe(text=messages, max_new_tokens=GENERATION_PARAMS["batch_tokens"])
                    analysis_text = output[0]["generated_text"][-1]["content"]
                    
                    batch_analyses.append({
                        'file_path': file_path,
                        'analysis': analysis_text,
                        'file_name': os.path.basename(file_path)
                    })
                    
                except Exception as e:
                    print(f"⚠️  Ошибка при анализе {os.path.basename(file_path)}: {e}")
                    continue
            
            print(f"✅ Батч {batch_num}/{total_batches} завершен: {len(batch_analyses[-current_batch_size:])} анализов")
        
        print(f"🎉 Батчевая обработка завершена! Всего обработано: {len(batch_analyses)} изображений")
        
        # Информация о предупреждении GPU pipelines
        if self.device == "cuda":
            print("ℹ️  Примечание: Сообщение 'pipelines sequentially on GPU' ожидаемо при батчевой обработке")
        
        return batch_analyses
    
    def analyze_directory(self, directory_path):
        """
        Анализ всех DICOM файлов в директории
        
        Args:
            directory_path (str): Путь к директории с DICOM файлами
        """
        if not os.path.exists(directory_path):
            print(f"Директория {directory_path} не найдена!")
            return
        
        print(f"\nАнализируем DICOM файлы в: {directory_path}")
        
        # Поиск всех DICOM файлов
        dicom_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.dcm', '.dicom')):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            print(f"DICOM файлы не найдены в {directory_path}")
            return
        
        print(f"Найдено {len(dicom_files)} DICOM файлов")
        
        # Проверяем debug режим
        if DEBUG_MODE:
            # Берем каждый 5-й файл до конца
            debug_files = []
            for i in range(0, len(dicom_files), 5):  # Каждый 5-й файл
                debug_files.append(dicom_files[i])
            dicom_files = debug_files
            print(f"🐛 DEBUG РЕЖИМ: Анализируем каждый 5-й файл, всего {len(dicom_files)} файлов")
        
        # Загружаем все изображения с прогресс-баром
        images = []
        valid_files = []
        
        print("Загружаем изображения...")
        for dicom_file in tqdm(dicom_files, desc="Загрузка DICOM файлов", unit="файл"):
            image = self.load_dicom_as_image(dicom_file)
            if image is not None:
                images.append(image)
                valid_files.append(dicom_file)
        
        if not images:
            print("Не удалось загрузить ни одного изображения!")
            return
        
        print(f"Успешно загружено {len(images)} изображений")
        
        # Анализ всех изображений батчами для эффективности GPU
        print("Анализируем все изображения...")
        all_analyses = []
        
        # Разбиваем на батчи для эффективной обработки на GPU
        batch_size = self.batch_size
        total_batches = (len(images) + batch_size - 1) // batch_size
        print(f"📊 Всего батчей: {total_batches} (по {batch_size} изображений в батче)")
        
        for batch_idx in tqdm(range(total_batches), desc="Анализ батчей", unit="батч"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(images))
            batch_images = images[start_idx:end_idx]
            batch_files = valid_files[start_idx:end_idx]
            
            # Анализируем батч
            batch_analyses = self.analyze_batch(batch_images, batch_files)
            all_analyses.extend(batch_analyses)
        
        # Создаем общий анализ на основе всех индивидуальных анализов
        combined_result = self.create_combined_analysis(all_analyses, valid_files)
        self.results.append(combined_result)
    
    def analyze_single_file(self, file_path):
        """
        Анализ одного DICOM файла
        
        Args:
            file_path (str): Путь к DICOM файлу
        """
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден!")
            return
        
        print(f"\nАнализируем файл: {file_path}")
        
        # Загрузка изображения
        image = self.load_dicom_as_image(file_path)
        if image is None:
            return
        
        # Анализ изображения
        result = self.analyze_image(image, file_path)
        self.results.append(result)
        
        print(f"Анализ завершен для файла: {result['file_name']}")
    
    def print_results_table(self):
        """Вывод результатов в виде красивой таблицы"""
        if not self.results:
            print("\nНет результатов для отображения!")
            return
        
        print("\n" + "="*100)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА DICOM ФАЙЛОВ")
        print(f"Модель: MedGemma-{self.model_name.upper()} ({self.model_path})")
        print("="*100)
        
        # Вывод общего анализа
        for idx, result in enumerate(self.results, 1):
            print(f"\n📊 АНАЛИЗ СЕРИИ {idx}:")
            print("-" * 100)
            
            if 'total_files' in result:
                # Новый формат - анализ всех файлов
                print(f"📁 Всего файлов проанализировано: {result['total_files']}")
                if 'analyzed_image' in result:
                    print(f"🔍 Представительное изображение: {result['analyzed_image']}")
                print(f"📂 Обработанные файлы:")
                for file_path in result['file_paths'][:5]:  # Показываем первые 5 файлов
                    print(f"   - {os.path.basename(file_path)}")
                if len(result['file_paths']) > 5:
                    print(f"   ... и еще {len(result['file_paths']) - 5} файлов")
                print(f"\n💬 ОБЩИЙ АНАЛИЗ ПО ВСЕМ ФАЙЛАМ:")
                print(f"{result['analysis']}")
            else:
                # Старый формат - анализ одного файла
                print(f"📁 Файл: {result['file_name']}")
                print(f"📂 Путь: {result['file_path']}")
                print(f"💬 Анализ: {result['analysis']}")
            
            print("-" * 100)
        
        # Общий отчет
        total_series = len(self.results)
        total_files = sum(result.get('total_files', 1) for result in self.results)
        print(f"\n📈 ОБЩИЙ ОТЧЕТ:")
        print("-" * 50)
        print(f"   Всего серий проанализировано: {total_series}")
        print(f"   Всего файлов обработано: {total_files}")
        print(f"   Анализ завершен успешно!")
        
        print("\n" + "="*100)

def show_help():
    """Показать справку по использованию"""
    help_text = """
🔬 DICOM АНАЛИЗАТОР С MEDGEMMA - СПРАВКА

ИСПОЛЬЗОВАНИЕ:
    python dicom_analyzer.py [ОПЦИИ] <ПУТЬ_К_ПАПКЕ_ИЛИ_ФАЙЛУ>

ОСНОВНЫЕ КОМАНДЫ:
    /путь/к/папке/         Анализ всех DICOM файлов в папке
    /путь/к/файлу.dcm      Анализ отдельного DICOM файла
    '/путь/к/файлам/*'     Анализ файлов по glob-паттерну
    --help, -h             Показать эту справку

ОПЦИИ МОДЕЛИ:
    --model=4b|27b         Выбор модели MedGemma (по умолчанию: 4b)
    --prompt="текст"       Пользовательский промпт для анализа
    --lang=ЯЗЫК            Язык ответа: en, ru (по умолчанию: en)

ОПЦИИ CT WINDOWING:
    --wl=ЧИСЛО             Window Level (по умолчанию: -550)
    --ww=ЧИСЛО             Window Width (по умолчанию: 1600)
    --window=WL,WW         Установить WL и WW одной командой
    --pneumonia-window=ТИП Специальные окна: lung_soft, infection, standard_lung

ОПЦИИ ОБРАБОТКИ:
    --batch-size=ЧИСЛО     Размер батча для GPU (по умолчанию: 8)
    --debug                Анализировать каждый 5-й файл (для тестирования)

ПРИМЕРЫ:
    # Анализ папки с файлами
    python dicom_analyzer.py /data/dicom_files/
    
    # Анализ одного файла
    python dicom_analyzer.py /data/scan.dcm
    
    # Анализ диапазона файлов (glob-паттерны)
    python dicom_analyzer.py '/data/example/abnormal/1/IMG-000*'
    python dicom_analyzer.py '/data/scans/patient_*/slice_[0-9][0-9].dcm'
    python dicom_analyzer.py '/data/study/IMG-00[1-5]*.dcm'
    
    # Дебаг режим с моделью 27B
    python dicom_analyzer.py --debug --model=27b /data/
    
    # Пользовательский промпт с glob-паттерном
    python dicom_analyzer.py --prompt="Найди признаки пневмонии" '/data/covid_cases/IMG-*.dcm'
    
    # Ответ на русском языке
    python dicom_analyzer.py --lang=ru '/data/example/abnormal/1/IMG-000*'
    
    # Специальное окно для инфекций
    python dicom_analyzer.py --pneumonia-window=infection '/data/pneumonia/IMG-*.dcm'

DOCKER ПРИМЕРЫ:
    # Анализ папки через Docker
    docker-compose run --rm dicom-analyzer /data
    
    # С пользовательскими параметрами
    docker-compose run --rm dicom-analyzer --model=27b --debug /data
"""
    print(help_text)

def expand_glob_pattern(pattern):
    """Расширение glob-паттерна в список файлов"""
    try:
        # Проверяем, содержит ли путь glob-символы
        if any(char in pattern for char in ['*', '?', '[', ']']):
            files = glob.glob(pattern)
            # Фильтруем только DICOM файлы
            dicom_files = []
            for file_path in files:
                if os.path.isfile(file_path):
                    # Проверяем расширение или пытаемся прочитать как DICOM
                    if (file_path.lower().endswith(('.dcm', '.dicom')) or 
                        not os.path.splitext(file_path)[1]):  # файлы без расширения
                        dicom_files.append(file_path)
            
            dicom_files.sort()  # Сортируем для предсказуемого порядка
            print(f"🔍 Найдено {len(dicom_files)} DICOM файлов по паттерну: {pattern}")
            return dicom_files
        else:
            # Обычный путь без glob-символов
            return [pattern] if os.path.exists(pattern) else []
    except Exception as e:
        print(f"❌ Ошибка при обработке паттерна '{pattern}': {e}")
        return []

def analyze_single_file(file_path, analyzer):
    """Анализ отдельного DICOM файла"""
    print(f"\n📁 Анализируем файл: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Файл не найден: {file_path}")
        return
    
    try:
        # Загружаем и обрабатываем DICOM файл
        image = analyzer.load_dicom_as_image(file_path)
        if image is None:
            print(f"❌ Не удалось загрузить DICOM файл: {file_path}")
            return
        
        # Анализируем изображение
        result = analyzer.analyze_image(image, file_path)
        
        # Выводим результат
        print(f"\n🔍 РЕЗУЛЬТАТ АНАЛИЗА:")
        print(f"Файл: {result['file_name']}")
        print(f"Анализ: {result['analysis']}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Ошибка при анализе файла {file_path}: {e}")

def analyze_file_list(file_list, analyzer):
    """Анализ списка DICOM файлов с батчевой обработкой"""
    if not file_list:
        print("❌ Список файлов пуст")
        return
    
    print(f"\n📋 Анализируем {len(file_list)} файлов батчами...")
    
    # Если один файл - используем analyze_single_file
    if len(file_list) == 1:
        analyze_single_file(file_list[0], analyzer)
        return
    
    # Загружаем все изображения с минимальным выводом
    print("🔄 Загрузка и конвертация DICOM файлов...")
    images_and_paths = []
    
    for file_path in tqdm(file_list, desc="Загрузка DICOM", leave=False):
        try:
            if not os.path.exists(file_path):
                continue
                
            # Загружаем DICOM без вывода параметров для каждого файла
            image = analyzer.load_dicom_as_image(file_path, silent=True)
            if image is not None:
                images_and_paths.append((image, file_path))
                
        except Exception:
            continue  # Тихо пропускаем проблемные файлы
    
    if not images_and_paths:
        print("❌ Не удалось загрузить ни одного файла")
        return
    
    print(f"✅ Загружено {len(images_and_paths)} файлов")
    print("🤖 Запуск батчевого анализа с MedGemma...")
    print(f"📦 Размер батча: {analyzer.batch_size}")
    print(f"🔧 Устройство: {analyzer.device.upper()}")
    
    # Вычисляем количество батчей
    total_batches = (len(images_and_paths) + analyzer.batch_size - 1) // analyzer.batch_size
    print(f"📊 Всего батчей для обработки: {total_batches}")
    
    # Батчевый анализ
    try:
        results = analyzer.analyze_batch([img for img, _ in images_and_paths], 
                                       [path for _, path in images_and_paths])
        
        # Создаем общий отчет
        if results:
            combined_report = analyzer.create_combined_analysis(results)
            print(f"\n📊 ОБЩИЙ ОТЧЕТ ПО {len(results)} ФАЙЛАМ:")
            print("="*80)
            print(combined_report)
            print("="*80)
        else:
            print("❌ Не удалось проанализировать файлы")
            
    except Exception as e:
        print(f"❌ Ошибка при батчевом анализе: {e}")
        print("🔄 Переключаемся на поштучный анализ...")
        
        # Fallback - поштучный анализ без лишнего вывода
        results = []
        for image, file_path in tqdm(images_and_paths, desc="Анализ файлов"):
            try:
                result = analyzer.analyze_image(image, file_path)
                results.append(result)
            except Exception:
                continue
        
        if results:
            combined_report = analyzer.create_combined_analysis(results)
            print(f"\n📊 ОБЩИЙ ОТЧЕТ ПО {len(results)} ФАЙЛАМ:")
            print("="*80)
            print(combined_report)
            print("="*80)

def main():
    """Основная функция"""
    print("🔬 DICOM АНАЛИЗАТОР С MEDGEMMA")
    print("="*60)
    
    # Проверяем справку
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        return
    
    # Проверяем флаг debug в аргументах командной строки
    global DEBUG_MODE
    if "--debug" in sys.argv:
        DEBUG_MODE = True
        sys.argv.remove("--debug")
        print("🐛 DEBUG РЕЖИМ ВКЛЮЧЕН - анализируем каждый 5-й файл до конца")
    
    # Проверяем выбор модели
    model_name = DEFAULT_MODEL
    window_level = DEFAULT_WINDOW_LEVEL
    window_width = DEFAULT_WINDOW_WIDTH
    batch_size = None
    custom_prompt = None
    language = "en"
    
    # Обработка аргументов командной строки
    args_to_remove = []
    
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--model="):
            model_name = arg.split("=")[1]
            args_to_remove.append(arg)
        elif arg == "--model" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
            args_to_remove.extend([arg, sys.argv[i + 1]])
        elif arg.startswith("--wl="):
            window_level = int(arg.split("=")[1])
            args_to_remove.append(arg)
        elif arg == "--wl" and i + 1 < len(sys.argv):
            window_level = int(sys.argv[i + 1])
            args_to_remove.extend([arg, sys.argv[i + 1]])
        elif arg.startswith("--ww="):
            window_width = int(arg.split("=")[1])
            args_to_remove.append(arg)
        elif arg == "--ww" and i + 1 < len(sys.argv):
            window_width = int(sys.argv[i + 1])
            args_to_remove.extend([arg, sys.argv[i + 1]])
        elif arg.startswith("--window="):
            # Формат: --window=WL,WW например --window=-550,1600
            wl_ww = arg.split("=")[1].split(",")
            if len(wl_ww) == 2:
                window_level = int(wl_ww[0])
                window_width = int(wl_ww[1])
            args_to_remove.append(arg)
        elif arg.startswith("--pneumonia-window="):
            # Специальные окна для выявления пневмонии
            window_type = arg.split("=")[1]
            if window_type in PNEUMONIA_DETECTION_WINDOWS:
                window_level = PNEUMONIA_DETECTION_WINDOWS[window_type]["wl"]
                window_width = PNEUMONIA_DETECTION_WINDOWS[window_type]["ww"]
                print(f"🔍 Выбрано специальное окно для пневмонии: {window_type}")
            args_to_remove.append(arg)
        elif arg.startswith("--batch-size="):
            # Размер батча
            batch_size = int(arg.split("=")[1])
            args_to_remove.append(arg)
        elif arg == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
            args_to_remove.extend([arg, sys.argv[i + 1]])
        elif arg.startswith("--prompt="):
            # Пользовательский промпт
            custom_prompt = arg.split("=", 1)[1].strip('"').strip("'")
            args_to_remove.append(arg)
        elif arg == "--prompt" and i + 1 < len(sys.argv):
            custom_prompt = sys.argv[i + 1].strip('"').strip("'")
            args_to_remove.extend([arg, sys.argv[i + 1]])
        elif arg.startswith("--lang="):
            # Язык ответа
            language = arg.split("=")[1].lower()
            args_to_remove.append(arg)
        elif arg == "--lang" and i + 1 < len(sys.argv):
            language = sys.argv[i + 1].lower()
            args_to_remove.extend([arg, sys.argv[i + 1]])
    
    # Удаляем обработанные аргументы
    for arg in args_to_remove:
        if arg in sys.argv:
            sys.argv.remove(arg)
    
    print(f"🤖 Выбранная модель: MedGemma-{model_name.upper()}")
    print(f"🪟 Параметры окна: WL={window_level}, WW={window_width}")
    if batch_size:
        print(f"📦 Пользовательский размер батча: {batch_size}")
    
    # Создание анализатора
    analyzer = DICOMAnalyzer(model_name=model_name, window_level=window_level, window_width=window_width, batch_size=batch_size)
    
    # Применение языковых настроек
    global DEFAULT_ANALYSIS_PROMPT, CURRENT_LANGUAGE
    if language in LANGUAGE_PROMPTS:
        DEFAULT_ANALYSIS_PROMPT = LANGUAGE_PROMPTS[language]
        CURRENT_LANGUAGE = language
        ANALYSIS_PROMPTS["universal"] = DEFAULT_ANALYSIS_PROMPT
        ANALYSIS_PROMPTS["single_image"] = DEFAULT_ANALYSIS_PROMPT
        ANALYSIS_PROMPTS["batch_analysis"] = DEFAULT_ANALYSIS_PROMPT
        
        lang_names = {"en": "English", "ru": "Русский"}
        print(f"🌍 Язык ответа: {lang_names.get(language, language.upper())}")
    else:
        print(f"⚠️  Неподдерживаемый язык '{language}'. Доступные: {list(LANGUAGE_PROMPTS.keys())}")
        print("🌍 Используется английский язык по умолчанию")
        language = "en"
        DEFAULT_ANALYSIS_PROMPT = LANGUAGE_PROMPTS["en"]
    
    # Применение пользовательского промпта (переопределяет языковой)
    if custom_prompt:
        DEFAULT_ANALYSIS_PROMPT = custom_prompt
        ANALYSIS_PROMPTS["universal"] = custom_prompt
        ANALYSIS_PROMPTS["single_image"] = custom_prompt
        ANALYSIS_PROMPTS["batch_analysis"] = custom_prompt
        print(f"💬 Используется пользовательский промпт: {custom_prompt[:50]}...")
    
    # Проверка аргументов командной строки
    if len(sys.argv) > 1:
        # Анализ файла, директории или glob-паттерна из аргументов
        path_pattern = sys.argv[1]
        
        # Проверяем, является ли это glob-паттерном
        if any(char in path_pattern for char in ['*', '?', '[', ']']):
            # Обрабатываем как glob-паттерн
            file_list = expand_glob_pattern(path_pattern)
            if file_list:
                analyze_file_list(file_list, analyzer)
            else:
                print(f"❌ Не найдено файлов по паттерну: {path_pattern}")
                return
        elif os.path.isfile(path_pattern):
            # Анализ отдельного файла
            analyze_single_file(path_pattern, analyzer)
        elif os.path.isdir(path_pattern):
            # Анализ директории
            analyzer.analyze_directory(path_pattern)
        else:
            print(f"❌ Путь или паттерн '{path_pattern}' не найден!")
            print("💡 Примеры использования:")
            print("   python dicom_analyzer.py /data/scan.dcm")
            print("   python dicom_analyzer.py /data/scans/")
            print("   python dicom_analyzer.py '/data/example/abnormal/1/IMG-000*'")
            return
    else:
        # Используем путь из переменной DICOM_FOLDER_PATH
        print(f"\nИспользуем путь из настроек: {DICOM_FOLDER_PATH}")
        if DEBUG_MODE:
            print(f"🐛 Debug режим: будем анализировать каждый 5-й файл до конца")
        
        if os.path.exists(DICOM_FOLDER_PATH):
            analyzer.analyze_directory(DICOM_FOLDER_PATH)
        else:
            print(f"Директория {DICOM_FOLDER_PATH} не найдена!")
            print("Измените переменную DICOM_FOLDER_PATH в начале скрипта на правильный путь.")
            return
    
    # Вывод результатов
    analyzer.print_results_table()
    
    print("\n✅ Анализ завершен!")

if __name__ == "__main__":
    main()
