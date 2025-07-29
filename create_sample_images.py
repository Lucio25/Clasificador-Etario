"""
Script para crear imágenes de muestra para el juego
Este script genera imágenes placeholder con diferentes características visuales
"""
from PIL import Image, ImageDraw, ImageFont
import os
import random

def create_sample_images():
    """Crea 10 imágenes de muestra para el juego"""
    
    # Crear carpeta si no existe
    if not os.path.exists("game_images"):
        os.makedirs("game_images")
    
    # Datos de las imágenes de muestra
    sample_data = [
        {"name": "joven_masculino_1.jpg", "age_range": "13-23", "gender": "Masculino", "real_age": 19},
        {"name": "adulto_femenino_1.jpg", "age_range": "31-44", "gender": "Femenino", "real_age": 35},
        {"name": "joven_adulto_masculino_1.jpg", "age_range": "24-30", "gender": "Masculino", "real_age": 27},
        {"name": "adulto_mayor_femenino_1.jpg", "age_range": "45-60", "gender": "Femenino", "real_age": 52},
        {"name": "anciano_masculino_1.jpg", "age_range": ">60", "gender": "Masculino", "real_age": 72},
        {"name": "adolescente_femenino_1.jpg", "age_range": "13-23", "gender": "Femenino", "real_age": 16},
        {"name": "adulto_masculino_1.jpg", "age_range": "31-44", "gender": "Masculino", "real_age": 38},
        {"name": "joven_adulto_femenino_1.jpg", "age_range": "24-30", "gender": "Femenino", "real_age": 26},
        {"name": "adulto_mayor_masculino_1.jpg", "age_range": "45-60", "gender": "Masculino", "real_age": 48},
        {"name": "anciana_femenino_1.jpg", "age_range": ">60", "gender": "Femenino", "real_age": 68}
    ]
    
    colors = [
        (173, 216, 230),  # Azul claro
        (255, 182, 193),  # Rosa claro
        (144, 238, 144),  # Verde claro
        (255, 218, 185),  # Durazno
        (221, 160, 221),  # Ciruela
        (255, 255, 224),  # Amarillo claro
        (250, 235, 215),  # Blanco antiguo
        (230, 230, 250),  # Lavanda
        (255, 228, 225),  # Rosa misty
        (240, 248, 255)   # Azul alice
    ]
    
    for i, data in enumerate(sample_data):
        # Crear imagen
        width, height = 400, 400
        color = colors[i % len(colors)]
        img = Image.new('RGB', (width, height), color=color)
        draw = ImageDraw.Draw(img)
        
        # Intentar usar fuente del sistema, sino usar fuente por defecto
        try:
            font_large = ImageFont.truetype("arial.ttf", 32)
            font_medium = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Dibujar cara simple
        face_size = 200
        face_x = (width - face_size) // 2
        face_y = (height - face_size) // 2 - 30
        
        # Cara (círculo)
        draw.ellipse([face_x, face_y, face_x + face_size, face_y + face_size], 
                    fill=(255, 220, 177), outline=(0, 0, 0), width=3)
        
        # Ojos
        eye_size = 20
        left_eye_x = face_x + 50
        right_eye_x = face_x + face_size - 70
        eye_y = face_y + 60
        
        draw.ellipse([left_eye_x, eye_y, left_eye_x + eye_size, eye_y + eye_size], 
                    fill=(0, 0, 0))
        draw.ellipse([right_eye_x, eye_y, right_eye_x + eye_size, eye_y + eye_size], 
                    fill=(0, 0, 0))
        
        # Nariz
        nose_x = face_x + face_size // 2
        nose_y = face_y + 100
        draw.line([nose_x, nose_y, nose_x, nose_y + 20], fill=(0, 0, 0), width=2)
        
        # Boca
        mouth_y = face_y + 140
        mouth_width = 40
        draw.arc([nose_x - mouth_width//2, mouth_y, nose_x + mouth_width//2, mouth_y + 20], 
                0, 180, fill=(0, 0, 0), width=3)
        
        # Texto informativo
        text_y = face_y + face_size + 40
        
        # Información principal
        main_text = f"Persona {i+1}"
        draw.text((width//2 - 60, text_y), main_text, fill=(0, 0, 0), font=font_large)
        
        # Información detallada
        detail_text = f"Rango: {data['age_range']}"
        draw.text((width//2 - 80, text_y + 40), detail_text, fill=(50, 50, 50), font=font_medium)
        
        genre_text = f"Género: {data['gender']}"
        draw.text((width//2 - 70, text_y + 65), genre_text, fill=(50, 50, 50), font=font_medium)
        
        age_text = f"Edad real: {data['real_age']} años"
        draw.text((width//2 - 90, text_y + 90), age_text, fill=(100, 100, 100), font=font_small)
        
        # Guardar imagen
        file_path = os.path.join("game_images", data['name'])
        img.save(file_path, 'JPEG', quality=95)
        print(f"✅ Creada: {file_path}")
    
    print(f"\n🎉 Se crearon {len(sample_data)} imágenes de muestra en la carpeta 'game_images'")
    print("🎮 Ahora puedes ejecutar el juego sin problemas!")

if __name__ == "__main__":
    create_sample_images()
