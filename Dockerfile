# Dockerfile для MedGemma с GPU поддержкой
FROM nvidia/cuda:12.0.1-base-ubuntu22.04

# Устанавливаем Python и зависимости
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python пакеты
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install transformers accelerate bitsandbytes
RUN pip3 install pillow requests

# Создаем рабочую директорию
WORKDIR /app

# Копируем только скрипты (модели монтируем отдельно)
COPY *.py /app/

# Устанавливаем переменные окружения
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Команда по умолчанию
CMD ["python3", "medgemma_docker.py"]
