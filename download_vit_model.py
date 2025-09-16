#!/usr/bin/env python3

import os
from huggingface_hub import snapshot_download

def download_vit_model():
    """Скачивание ViT модели для COVID-19 классификации"""
    
    print("🔄 Скачиваем ViT модель...")
    print("📦 Модель: DunnBC22/vit-base-patch16-224-in21k_covid_19_ct_scans")
    
    try:
        # Создаем директорию для модели
        local_dir = os.path.join(os.getcwd(), "models", "vit_covid")
        os.makedirs(local_dir, exist_ok=True)
        
        # Скачиваем модель
        snapshot_download(
            repo_id="DunnBC22/vit-base-patch16-224-in21k_covid_19_ct_scans",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        print("✅ ViT модель скачана успешно!")
        print(f"📁 Путь: {local_dir}")
        
        # Проверяем скачанные файлы
        files = os.listdir(local_dir)
        print(f"📋 Скачано файлов: {len(files)}")
        for file in files:
            file_path = os.path.join(local_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Ошибка скачивания: {e}")
        print("🔧 Возможные решения:")
        print("1. Проверьте интернет соединение")
        print("2. Убедитесь что huggingface_hub установлен: pip install huggingface_hub")
        print("3. Проверьте доступ к модели на: https://huggingface.co/DunnBC22/vit-base-patch16-224-in21k_covid_19_ct_scans")

if __name__ == "__main__":
    download_vit_model()
