# ai-ct

DICOM анализатор с использованием MedGemma для анализа медицинских CT изображений.

## 🔬 Возможности

- Анализ DICOM файлов с помощью MedGemma-4B/27B
- Специализация на легочной патологии и пневмонии
- Медицински корректное CT windowing
- Батчевая обработка с оптимизацией GPU
- Docker контейнеризация с NVIDIA GPU поддержкой
- Настраиваемые параметры окна и размера батча

## 🚀 Быстрый старт

### Локальный запуск:
```bash
# Установка зависимостей
pip install -r requirements_dicom_analyzer.txt

# Запуск анализа
python dicom_analyzer.py --debug --model=4b /path/to/dicom/files
```

### Docker запуск:
```bash
# Сборка и запуск
docker-compose build
docker-compose run --rm dicom-analyzer --debug --model=4b /data
```

## ⚙️ Параметры

- `--model=4b|27b` - выбор модели MedGemma
- `--wl=-550` - Window Level для CT
- `--ww=1600` - Window Width для CT
- `--pneumonia-window=infection` - специальные окна для пневмонии
- `--batch-size=8` - размер батча для GPU
- `--debug` - анализ каждого 5-го файла

## 📁 Структура

- `dicom_analyzer.py` - основной скрипт анализа
- `requirements_dicom_analyzer.txt` - Python зависимости
- `Dockerfile` - образ для контейнеризации
- `docker-compose.yml` - конфигурация Docker Compose

## 🏥 Специализация

Оптимизирован для анализа легких и выявления:
- Пневмонии и инфекций
- Консолидаций и затемнений
- Ground-glass opacities
- Пневмоторакса
- Плевральных изменений

## 🛠️ Требования

- Python 3.8+
- CUDA GPU (рекомендуется)
- 8GB+ RAM
- Docker + NVIDIA Docker (для контейнеров)
