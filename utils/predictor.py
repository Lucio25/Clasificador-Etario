import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from .model_loader import (cargar_modelo_gender, obtener_transformaciones)

def cargar_modelo_edad_directo(ruta_modelo):
    """Carga un modelo de edad guardado como modelo completo"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        modelo = torch.load(ruta_modelo, map_location=device, weights_only=False)
        if hasattr(modelo, 'eval'):
            modelo.eval()
            modelo.to(device)
            return modelo, device
        else:
            raise Exception("El objeto cargado no es un modelo válido")
    except Exception as e:
        raise Exception(f"Error al cargar modelo: {e}")

class Predictor:
    def __init__(self, ruta_modelo_gender, ruta_modelo_edad_hombres, ruta_modelo_edad_mujeres):
        """
        Inicializa el predictor con los 3 modelos entrenados
        """
        self.modelo_gender, self.device = cargar_modelo_gender(ruta_modelo_gender)
        self.modelo_edad_hombres, _ = cargar_modelo_edad_directo(ruta_modelo_edad_hombres)
        self.modelo_edad_mujeres, _ = cargar_modelo_edad_directo(ruta_modelo_edad_mujeres)
        self.transform = obtener_transformaciones()
        
        # Mapeo de clases - CORREGIDO: Si detecta hombres como mujeres, el mapeo estaba invertido
        self.clases_gender = {0: "Masculino", 1: "Femenino"}  # Mapeo invertido
        self.clases_edad = {
            0: "Bebé (0-3)",
            1: "Niño/a (4-12)", 
            2: "Adolescente (13-23)",
            3: "Joven Adulto (24-30)",
            4: "Adulto (31-44)",
            5: "Adulto Mayor (45-60)",
            6: "Anciano/a (>60)"
        }
    
    def preprocesar_imagen(self, imagen_pil):
        """
        Preprocesa la imagen para la predicción
        """
        imagen_tensor = self.transform(imagen_pil).unsqueeze(0)
        return imagen_tensor.to(self.device)
    
    def predecir_gender(self, imagen_pil):
        """
        Predice el género de la persona en la imagen
        """
        imagen_tensor = self.preprocesar_imagen(imagen_pil)
        
        with torch.no_grad():
            outputs = self.modelo_gender(imagen_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        gender_predicho = self.clases_gender[predicted_class]
        return gender_predicho, confidence
    
    def predecir_edad(self, imagen_pil, gender_predicho):
        """
        Predice el rango etario de la persona según su género
        """
        imagen_tensor = self.preprocesar_imagen(imagen_pil)
        
        # Seleccionar el modelo correcto según el género
        if gender_predicho == "Masculino":
            modelo_edad = self.modelo_edad_hombres
        else:  # Femenino
            modelo_edad = self.modelo_edad_mujeres
        
        with torch.no_grad():
            outputs = modelo_edad(imagen_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        edad_predicha = self.clases_edad[predicted_class]
        return edad_predicha, confidence
    
    def predecir_completo(self, imagen_pil):
        """
        Realiza la predicción completa: primero género, luego edad según el género
        """
        # Paso 1: Predecir género
        gender_predicho, confianza_gender = self.predecir_gender(imagen_pil)
        
        # Paso 2: Predecir edad usando el modelo específico del género
        edad_predicha, confianza_edad = self.predecir_edad(imagen_pil, gender_predicho)
        
        return {
            'sexo': gender_predicho,
            'confianza_sexo': confianza_gender,
            'edad': edad_predicha,
            'confianza_edad': confianza_edad
        }
    
    def cambiar_mapeo_genero(self, invertir=False):
        """
        Cambia el mapeo de género dinámicamente
        """
        if invertir:
            self.clases_gender = {0: "Femenino", 1: "Masculino"}
        else:
            self.clases_gender = {0: "Masculino", 1: "Femenino"}
