# DICOM Analyzer Dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements
COPY requirements_dicom_analyzer.txt .

# Установка Python зависимостей
RUN pip3 install --no-cache-dir -r requirements_dicom_analyzer.txt

# Копирование скрипта анализатора
COPY dicom_analyzer.py .

# Создание директории для данных
RUN mkdir -p /data

# Создание пользователя для безопасности
RUN groupadd -r dicom && useradd -r -g dicom dicom
RUN chown -R dicom:dicom /app /data

# Переключение на пользователя dicom
USER dicom

# Установка переменных окружения
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface

# Создание кэш-директорий
RUN mkdir -p /app/.cache/transformers /app/.cache/huggingface

# Точка входа
ENTRYPOINT ["python3", "dicom_analyzer.py"]

# По умолчанию анализируем папку /data
CMD ["/data"]