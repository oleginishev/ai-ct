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

# Пример вывода:
# 📋 Анализируем всю папку: /data/dicom_files/
# 🔍 Найдено 50 DICOM файлов
# 📦 Размер батча: 8
# 🔧 Устройство: CUDA
# ⚡ Обрабатываем 50 изображений батчами по 8...
```

### Анализ отдельного файла
```bash
# Анализ одного DICOM файла
python dicom_analyzer.py /path/to/scan.dcm

# С пользовательским промптом
python dicom_analyzer.py --prompt="Найди признаки пневмонии" /path/to/scan.dcm

# Пример вывода:
# 📁 Анализируем файл: /data/scan.dcm
# Применены DICOM параметры: Slope=1.0, Intercept=0.0
# Применено CT окно: WL=-550, WW=1600
# 🔍 РЕЗУЛЬТАТ АНАЛИЗА:
# Файл: scan.dcm
# Анализ: Normal chest CT scan. No signs of pneumonia...
```

### 🎯 Анализ диапазона файлов (GLOB-паттерны)
```bash
# Все файлы, начинающиеся с IMG-000
python dicom_analyzer.py '/data/example/abnormal/1/IMG-000*'

# Файлы с определенными номерами (от 1 до 5)
python dicom_analyzer.py '/data/study/IMG-00[1-5]*.dcm'

# Сложные паттерны для множественных пациентов
python dicom_analyzer.py '/data/patients/patient_*/CT_*.dcm'

# Пример вывода с батчевой обработкой:
# 🔍 Найдено 15 DICOM файлов по паттерну: /data/example/abnormal/1/IMG-000*
# 📋 Анализируем 15 файлов батчами...
# 🔄 Загрузка и конвертация DICOM файлов...
# ✅ Загружено 15 файлов
# 🤖 Запуск батчевого анализа с MedGemma...
# 📦 Размер батча: 8
# 🔧 Устройство: CUDA
# 📊 Всего батчей для обработки: 2
# ⚡ Обрабатываем 15 изображений батчами по 8...
# 🔄 Батч 1/2: обрабатываем 8 изображений...
# ✅ Батч 1/2 завершен: 8 анализов за 23.4с (0.3 изображений/сек)
# 🔄 Батч 2/2: обрабатываем 7 изображений...
# ✅ Батч 2/2 завершен: 7 анализов за 19.8с (0.4 изображений/сек)
# 🎉 Батчевая обработка завершена! Всего обработано: 15 изображений
# ⏱️  Общее время обработки: 43.2 секунд
# 📈 Средняя скорость: 0.3 изображений/сек (2.9с на изображение)
```

## 🤖 Выбор модели

```bash
# Модель MedGemma-4B (по умолчанию, быстрая)
python dicom_analyzer.py --model=4b /data/

# Модель MedGemma-27B (более точная, медленнее)
python dicom_analyzer.py --model=27b /data/

# Пример вывода при загрузке модели:
# 🤖 Выбранная модель: MedGemma-4B
# Загружаем MedGemma модель: google/medgemma-4b-it
# ✅ MedGemma 4B модель загружена успешно!

# Сравнение результатов разных моделей
python dicom_analyzer.py --model=4b '/data/test_case.dcm' > results_4b.txt
python dicom_analyzer.py --model=27b '/data/test_case.dcm' > results_27b.txt

# Обработка в дебаг режиме с быстрой моделью
python dicom_analyzer.py --debug --model=4b --batch-size=16 '/data/large_dataset/IMG-*.dcm'

# Точный анализ важного случая с большой моделью  
python dicom_analyzer.py --model=27b --batch-size=4 /data/complex_case.dcm
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
python dicom_analyzer.py --prompt="Внимательно осмотри нижние доли легких" '/data/IMG-*.dcm'

# Кастомный анализ с glob-паттерном
python dicom_analyzer.py --prompt="Опиши все патологические изменения в легких с указанием локализации" '/data/patient_001/CT-*.dcm'

# Экстренная диагностика
python dicom_analyzer.py --prompt="СРОЧНО: есть ли признаки пневмоторакса или тромбоэмболии?" /data/emergency.dcm

# Детальный научный анализ
python dicom_analyzer.py --prompt="Детальный морфометрический анализ для научного исследования" --model=27b '/data/research/*.dcm'

# Пример вывода с промптом:
# 💬 Используется пользовательский промпт: Найди признаки пневмонии и COVID-19...
# 🔍 РЕЗУЛЬТАТ АНАЛИЗА:
# Анализ: Based on the chest CT analysis focusing on pneumonia and COVID-19 signs...
```

## 🌍 Языковые настройки

```bash
# Английский (по умолчанию)
python dicom_analyzer.py --lang=en /data/scan.dcm

# Русский язык
python dicom_analyzer.py --lang=ru /data/scan.dcm

# Пример вывода на русском:
# 🌍 Язык ответа: Русский
# 🔍 РЕЗУЛЬТАТ АНАЛИЗА:
# Анализ: КТ грудной клетки: определяются двусторонние инфильтративные изменения...

# Комбинирование языка с другими параметрами
python dicom_analyzer.py --lang=ru --model=27b --pneumonia-window=infection '/data/covid_cases/IMG-*.dcm'

# Пользовательский промпт на русском
python dicom_analyzer.py --lang=ru --prompt="Найди признаки пневмонии в нижних долях" /data/scan.dcm
```

## ⚡ Оптимизация производительности

```bash
# Настройка размера батча для мощных GPU (RTX 4090, A100)
python dicom_analyzer.py --batch-size=16 '/data/large_dataset/IMG-*.dcm'

# Средний батч для обычных GPU (RTX 3080, 4070)
python dicom_analyzer.py --batch-size=8 '/data/screening/*.dcm'

# Малый батч для слабых GPU (GTX 1080, RTX 2060)
python dicom_analyzer.py --batch-size=4 '/data/single_patient/*.dcm'

# CPU режим (медленный, но работает без GPU)
python dicom_analyzer.py --batch-size=1 /data/scan.dcm

# Пример вывода с оптимизацией:
# 📦 Размер батча: 16
# 🔧 Устройство: CUDA  
# 📊 Всего батчей для обработки: 4
# ⚡ Обрабатываем 60 изображений батчами по 16...
# 🔄 Батч 1/4: обрабатываем 16 изображений...
# ✅ Батч 1/4 завершен: 16 анализов за 15.2с (1.1 изображений/сек)
# 📈 Средняя скорость: 1.0 изображений/сек (1.0с на изображение)

# Оптимальные комбинации для разных задач:

# Массовый скрининг (быстро)  
python dicom_analyzer.py --debug --model=4b --batch-size=16 '/data/screening/IMG-*.dcm'

# Точная диагностика (медленно, но качественно)
python dicom_analyzer.py --model=27b --batch-size=4 --pneumonia-window=infection '/data/critical_cases/*.dcm'

# Сбалансированный режим
python dicom_analyzer.py --model=4b --batch-size=8 '/data/routine/*.dcm'
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

## 🏥 Полные клинические сценарии с выводом

### 🚨 Экстренная диагностика COVID-19
```bash
python dicom_analyzer.py \
  --lang=ru \
  --model=27b \
  --pneumonia-window=infection \
  --batch-size=8 \
  '/data/emergency/covid_suspect/IMG-*.dcm'

# Пример полного вывода:
# 🔍 Найдено 12 DICOM файлов по паттерну: /data/emergency/covid_suspect/IMG-*
# 📋 Анализируем 12 файлов батчами...
# 🔄 Загрузка и конвертация DICOM файлов...
# ✅ Загружено 12 файлов
# 🤖 Запуск батчевого анализа с MedGemma...
# 📦 Размер батча: 8
# 🔧 Устройство: CUDA
# 📊 Всего батчей для обработки: 2
# ⚡ Обрабатываем 12 изображений батчами по 8...
# 🔄 Батч 1/2: обрабатываем 8 изображений...
# ✅ Батч 1/2 завершен: 8 анализов за 18.5с (0.4 изображений/сек)
# 🔄 Батч 2/2: обрабатываем 4 изображения...
# ✅ Батч 2/2 завершен: 4 анализа за 9.2с (0.4 изображений/сек)
# 🎉 Батчевая обработка завершена! Всего обработано: 12 изображений
# ⏱️  Общее время обработки: 27.7 секунд
# 📈 Средняя скорость: 0.4 изображений/сек (2.3с на изображение)
# 
# 📊 ОБЩИЙ ОТЧЕТ ПО 12 ФАЙЛАМ:
# КТ грудной клетки выявляет двусторонние изменения по типу "матового стекла"...
```

### 📊 Массовый скрининг с дебаг режимом
```bash
python dicom_analyzer.py \
  --debug \
  --model=4b \
  --batch-size=16 \
  --lang=en \
  '/data/screening/batch_2024_01/IMG-*.dcm'

# Ожидаемый вывод:
# 🔍 Найдено 384 DICOM файлов по паттерну
# 🐛 DEBUG РЕЖИМ ВКЛЮЧЕН - анализируем каждый 5-й файл до конца
# 📋 Анализируем 77 файлов батчами...
# 🤖 Запуск батчевого анализа с MedGemma...
# 📦 Размер батча: 16
# 📊 Всего батчей для обработки: 5
# ⚡ Обрабатываем 77 изображений батчами по 16...
# 🎉 Батчевая обработка завершена! Всего обработано: 77 изображений
# ⏱️  Общее время обработки: 61.4 секунд
# 📈 Средняя скорость: 1.3 изображений/сек (0.8с на изображение)
```

### 🔬 Точная диагностика одного случая
```bash
python dicom_analyzer.py \
  --model=27b \
  --lang=ru \
  --pneumonia-window=lung_soft \
  --prompt="Детальный анализ: все патологические изменения с локализацией" \
  /data/complex_case/patient_001.dcm

# Вывод для одного файла:
# 🤖 Выбранная модель: MedGemma-27B
# Загружаем MedGemma модель: google/medgemma-27b-it
# ✅ MedGemma 27B модель загружена успешно!
# 🌍 Язык ответа: Русский
# 💬 Используется пользовательский промпт: Детальный анализ: все патологические изменения...
# 📁 Анализируем файл: /data/complex_case/patient_001.dcm
# Применены DICOM параметры: Slope=1.0, Intercept=0.0
# Применено CT окно: WL=-400, WW=1400
# Диапазон HU: -1100 до 300 (оптимизировано для анализа легких)
# Анатомическая область: CHEST/LUNG (по умолчанию)
# 
# 🔍 РЕЗУЛЬТАТ АНАЛИЗА:
# Файл: patient_001.dcm
# Анализ: В правой нижней доле визуализируется участок консолидации размером...
```

## 🎯 Практические рекомендации по использованию

### 📈 Оптимальные настройки для разных GPU:
- **RTX 4090/A100**: `--batch-size=16` для максимальной скорости
- **RTX 3080/4070**: `--batch-size=8` оптимальный баланс
- **RTX 2060/3060**: `--batch-size=4` стабильная работа
- **GTX 1080/CPU**: `--batch-size=1` консервативный режим

### ⏱️ Время обработки (примерные значения):
- **MedGemma-4B**: ~1-2 сек/изображение на RTX 4090
- **MedGemma-27B**: ~3-5 сек/изображение на RTX 4090
- **CPU режим**: ~30-60 сек/изображение

### 🎪 Выбор модели в зависимости от задачи:
- **Скрининг**: `--model=4b --debug` для быстрой обработки
- **Клиника**: `--model=4b --batch-size=8` сбалансированный режим  
- **Исследования**: `--model=27b --batch-size=4` максимальная точность
