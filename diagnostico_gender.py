#!/usr/bin/env python3
"""
Script para diagnosticar el problema de predicción de género
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from utils.model_loader import cargar_modelo_gender, obtener_transformaciones

def probar_modelo_gender():
    """Prueba el modelo de género con diferentes configuraciones"""
    
    print("🔍 Diagnóstico del modelo de género")
    print("="*50)
    
    # Cargar modelo
    try:
        modelo, device = cargar_modelo_gender("models/Gender_Model.pth")
        print(f"✅ Modelo cargado correctamente en: {device}")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return
    
    # Obtener transformaciones
    transform = obtener_transformaciones()
    
    # Probar con una imagen de ejemplo (si existe)
    test_images = [
        "test_male.jpg",
        "test_female.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📸 Probando con: {img_path}")
            
            # Cargar imagen
            imagen = Image.open(img_path).convert("RGB")
            imagen_tensor = transform(imagen).unsqueeze(0).to(device)
            
            # Hacer predicción
            with torch.no_grad():
                outputs = modelo(imagen_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                print(f"   📊 Outputs raw: {outputs}")
                print(f"   📊 Probabilities: {probabilities}")
                print(f"   📊 Predicted class: {predicted_class}")
                print(f"   📊 Confidence: {confidence:.4f}")
                
                # Mapeo actual
                clases_gender = {0: "Femenino", 1: "Masculino"}
                resultado = clases_gender[predicted_class]
                print(f"   🎯 Resultado: {resultado}")
    
    # Mostrar información del modelo
    print(f"\n🧠 Información del modelo:")
    print(f"   📊 Número de parámetros: {sum(p.numel() for p in modelo.parameters())}")
    print(f"   📊 Arquitectura: {modelo}")
    
    # Probar mapeo inverso
    print(f"\n🔄 Prueba de mapeo inverso:")
    print(f"   Si el modelo predice 0 → debería ser Femenino")
    print(f"   Si el modelo predice 1 → debería ser Masculino")
    
    # Sugerir mapeo alternativo
    print(f"\n💡 Si los resultados están invertidos, prueba este mapeo:")
    print(f"   clases_gender = {{0: 'Masculino', 1: 'Femenino'}}")

def mostrar_transformaciones():
    """Muestra las transformaciones aplicadas"""
    print(f"\n🔄 Transformaciones aplicadas:")
    transform = obtener_transformaciones()
    for i, t in enumerate(transform.transforms):
        print(f"   {i+1}. {t}")

if __name__ == "__main__":
    probar_modelo_gender()
    mostrar_transformaciones()
