"""
Configurador de Dataset para el Juego de Clasificación
=====================================================

Este script te ayuda a configurar tu dataset de imágenes para el juego.

Uso:
1. Coloca tus imágenes en una carpeta (ej: 'mi_dataset', 'data', 'images')
2. Ejecuta este script: python setup_dataset.py
3. El script detectará automáticamente las imágenes y las preparará para el juego

"""

import os
import glob
import shutil
from pathlib import Path

def find_images_in_directory(directory):
    """Busca todas las imágenes en un directorio"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    images = []
    
    for ext in extensions:
        # Búsqueda recursiva
        pattern = os.path.join(directory, '**', ext)
        images.extend(glob.glob(pattern, recursive=True))
        
        # Búsqueda directa
        pattern = os.path.join(directory, ext)
        images.extend(glob.glob(pattern))
    
    return list(set(images))  # Remover duplicados

def setup_dataset():
    """Configura el dataset para el juego"""
    print("🔍 Buscando imágenes en el directorio actual...")
    
    # Directorios comunes donde buscar imágenes
    search_dirs = [
        ".",
        "dataset", 
        "data", 
        "images", 
        "fotos", 
        "imagenes",
        "train",
        "test",
        "validation"
    ]
    
    all_images = []
    
    for directory in search_dirs:
        if os.path.exists(directory):
            print(f"📁 Buscando en: {directory}")
            images = find_images_in_directory(directory)
            if images:
                print(f"   ✅ Encontradas {len(images)} imágenes")
                all_images.extend(images)
            else:
                print(f"   ❌ No se encontraron imágenes")
    
    if not all_images:
        print("\n❌ No se encontraron imágenes en ningún directorio")
        print("💡 Soluciones:")
        print("   1. Copia tus imágenes a la carpeta actual")
        print("   2. Crea una carpeta 'dataset' o 'images' con tus imágenes")
        print("   3. Ejecuta este script desde la carpeta donde están tus imágenes")
        return
    
    # Remover duplicados y ordenar
    all_images = sorted(list(set(all_images)))
    
    print(f"\n🖼️ Total de imágenes encontradas: {len(all_images)}")
    
    # Crear carpeta game_images si no existe
    game_images_dir = "game_images"
    if not os.path.exists(game_images_dir):
        os.makedirs(game_images_dir)
        print(f"📁 Creada carpeta: {game_images_dir}")
    
    # Copiar hasta 10 imágenes a game_images
    images_to_copy = all_images[:10] if len(all_images) >= 10 else all_images
    
    print(f"\n📋 Copiando {len(images_to_copy)} imágenes a '{game_images_dir}'...")
    
    for i, image_path in enumerate(images_to_copy):
        # Obtener extensión original
        _, ext = os.path.splitext(image_path)
        
        # Nuevo nombre
        new_name = f"game_image_{i+1:02d}{ext}"
        new_path = os.path.join(game_images_dir, new_name)
        
        try:
            shutil.copy2(image_path, new_path)
            print(f"   ✅ {image_path} → {new_name}")
        except Exception as e:
            print(f"   ❌ Error copiando {image_path}: {e}")
    
    print(f"\n🎉 ¡Configuración completada!")
    print(f"📂 {len(images_to_copy)} imágenes listas en '{game_images_dir}'")
    print("🎮 Ahora puedes ejecutar tu aplicación Streamlit")
    print("   Comando: streamlit run app.py")

def list_current_images():
    """Lista las imágenes actualmente disponibles"""
    print("📋 Imágenes disponibles para el juego:")
    
    if os.path.exists("game_images"):
        images = find_images_in_directory("game_images")
        if images:
            for i, img in enumerate(sorted(images), 1):
                print(f"   {i:2d}. {os.path.basename(img)}")
        else:
            print("   ❌ No hay imágenes en 'game_images'")
    else:
        print("   ❌ La carpeta 'game_images' no existe")

if __name__ == "__main__":
    print("=" * 60)
    print("🎮 CONFIGURADOR DE DATASET PARA JUEGO DE CLASIFICACIÓN")
    print("=" * 60)
    
    while True:
        print("\n¿Qué quieres hacer?")
        print("1. 🔍 Buscar y configurar imágenes automáticamente")
        print("2. 📋 Ver imágenes actualmente disponibles")
        print("3. ❌ Salir")
        
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == "1":
            setup_dataset()
        elif choice == "2":
            list_current_images()
        elif choice == "3":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida. Intenta de nuevo.")
