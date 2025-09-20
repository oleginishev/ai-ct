# 🔬 ПРИМЕРЫ ЗАПУСКА DICOM АНАЛИЗАТОРА

## 📋 Базовые команды

### Справка
```bash
# Показать справку
python dicom_analyzer.py --help
python dicom_analyzer.py -h
```

### Анализ папки с DICOM файлами
```bash
# Анализ всех файлов в папке
python dicom_analyzer.py /path/to/dicom/folder/

# Анализ с дебаг режимом (каждый 5-й файл)
python dicom_analyzer.py --debug /path/to/dicom/folder/
```

### Анализ отдельного файла
```bash
# Анализ одного DICOM файла
python dicom_analyzer.py /path/to/scan.dcm

# С пользовательским промптом
python dicom_analyzer.py --prompt="Найди признаки пневмонии" /path/to/scan.dcm
```

## 🤖 Выбор модели

```bash
# Модель MedGemma-4B (по умолчанию)
python dicom_analyzer.py --model=4b /data/

# Модель MedGemma-27B (более точная)
python dicom_analyzer.py --model=27b /data/
```

## 🪟 Настройка CT Windowing

```bash
# Настройка Window Level и Width отдельно
python dicom_analyzer.py --wl=-550 --ww=1600 /data/

# Настройка одной командой
python dicom_analyzer.py --window=-550,1600 /data/

# Специальные окна для пневмонии
python dicom_analyzer.py --pneumonia-window=infection /data/
python dicom_analyzer.py --pneumonia-window=lung_soft /data/
python dicom_analyzer.py --pneumonia-window=standard_lung /data/
```

## 💬 Пользовательские промпты

```bash
# Поиск конкретной патологии
python dicom_analyzer.py --prompt="Найди признаки пневмонии и COVID-19" /data/scan.dcm

# Фокус на определенной области
python dicom_analyzer.py --prompt="Внимательно осмотри нижние доли легких" /data/

# Кастомный анализ
python dicom_analyzer.py --prompt="Опиши все патологические изменения в легких с указанием локализации" /data/
```

## ⚡ Оптимизация производительности

```bash
# Настройка размера батча для GPU
python dicom_analyzer.py --batch-size=16 /data/

# Комбинирование параметров
python dicom_analyzer.py --model=27b --batch-size=8 --debug /data/
```

## 🐳 Запуск через Docker

### Базовый запуск
```bash
# Сборка и запуск
docker-compose build
docker-compose up

# Запуск с параметрами
docker-compose run --rm dicom-analyzer --debug --model=4b /data
```

### С переменными окружения
```bash
# Установка токена Hugging Face
export HF_TOKEN="hf_your_token_here"

# Запуск с токеном
docker-compose run --rm -e HF_TOKEN=$HF_TOKEN dicom-analyzer /data
```

### Анализ отдельного файла
```bash
# Монтирование отдельного файла
docker run --rm -it \
  -v /path/to/scan.dcm:/data/scan.dcm:ro \
  -e HF_TOKEN=$HF_TOKEN \
  dicom-analyzer /data/scan.dcm
```

## 🔬 Специализированные сценарии

### COVID-19 скрининг
```bash
python dicom_analyzer.py \
  --model=27b \
  --pneumonia-window=infection \
  --prompt="Найди признаки COVID-19: матовое стекло, консолидации, ретикулярные изменения" \
  /data/covid_scans/
```

### Быстрый скрининг
```bash
python dicom_analyzer.py \
  --debug \
  --model=4b \
  --batch-size=16 \
  --prompt="Быстрая оценка: норма или патология?" \
  /data/large_dataset/
```

### Детальный анализ отдельного случая
```bash
python dicom_analyzer.py \
  --model=27b \
  --window=-400,1400 \
  --prompt="Детальный радиологический анализ с описанием всех находок" \
  /data/complex_case.dcm
```

## 📊 Примеры выходных данных

### Анализ одного файла
```
🔍 РЕЗУЛЬТАТ АНАЛИЗА:
Файл: scan001.dcm
Анализ: The chest CT shows bilateral ground-glass opacities primarily in the lower lobes, consistent with pneumonia. No pleural effusion detected. Recommend clinical correlation and follow-up imaging.
```

### Анализ папки (краткий отчет)
```
📈 ОБЩИЙ ОТЧЕТ ПО СЕРИИ:
Проанализировано файлов: 25
Найдены патологические изменения в 3 случаях
Рекомендации: Дополнительное обследование для файлов scan003.dcm, scan015.dcm, scan021.dcm
```

## 🔧 Отладка и диагностика

### Проверка окружения
```bash
# Проверка CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Проверка модели
python dicom_analyzer.py --model=4b --help
```

### Решение проблем
```bash
# Если модель не загружается
export HF_TOKEN="your_token"
python dicom_analyzer.py --model=4b /data/

# Если не хватает памяти
python dicom_analyzer.py --batch-size=1 /data/

# Если нужна CPU версия
docker build -f Dockerfile.cpu -t dicom-analyzer-cpu .
```

## 📝 Полезные комбинации

```bash
# Производственный анализ
python dicom_analyzer.py --model=27b --batch-size=8 --pneumonia-window=infection /data/

# Быстрое тестирование
python dicom_analyzer.py --debug --model=4b --batch-size=16 /data/test/

# Детальная диагностика
python dicom_analyzer.py --model=27b --window=-550,1600 --prompt="Детальный анализ с рекомендациями" /data/patient001.dcm
```
