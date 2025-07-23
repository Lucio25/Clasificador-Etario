#!/usr/bin/env python3
"""
Script para probar ambos mapeos de género y determinar cuál es correcto
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from utils.predictor import Predictor

# Definir la clase AgeClassifierResNet
class AgeClassifierResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(AgeClassifierResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def main():
    st.set_page_config(page_title="Prueba de Mapeo de Género", layout="centered")
    
    st.title("🔍 Prueba de Mapeo de Género")
    st.markdown("Sube una imagen para probar si el mapeo de género está correcto")
    
    # Cargar modelos
    RUTA_MODELO_GENDER = "models/Gender_Model.pth"
    RUTA_MODELO_EDAD_HOMBRES = "models/Male_Model.pt"
    RUTA_MODELO_EDAD_MUJERES = "models/Female_Model.pt"
    
    @st.cache_resource
    def cargar_predictor():
        try:
            predictor = Predictor(RUTA_MODELO_GENDER, RUTA_MODELO_EDAD_HOMBRES, RUTA_MODELO_EDAD_MUJERES)
            return predictor
        except Exception as e:
            st.error(f"Error al cargar modelos: {e}")
            return None
    
    predictor = cargar_predictor()
    
    if predictor is None:
        st.error("No se pudieron cargar los modelos")
        return
    
    # Subir imagen
    imagen = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    if imagen:
        imagen_pil = Image.open(imagen).convert("RGB")
        st.image(imagen_pil, caption="Imagen cargada", use_container_width=True)
        
        # Pregunta al usuario qué género es realmente
        st.markdown("### ¿Qué género es realmente la persona en la imagen?")
        genero_real = st.radio("Selecciona el género real:", ["Masculino", "Femenino"])
        
        if st.button("🔍 Probar Predicción"):
            with st.spinner("Analizando..."):
                try:
                    # Hacer predicción
                    resultado = predictor.predecir_gender(imagen_pil)
                    genero_predicho = resultado[0]
                    confianza = resultado[1]
                    
                    st.markdown("---")
                    st.markdown(f"**Género predicho:** {genero_predicho}")
                    st.markdown(f"**Confianza:** {confianza:.2%}")
                    st.markdown(f"**Género real:** {genero_real}")
                    
                    # Verificar si es correcto
                    if genero_predicho == genero_real:
                        st.success("✅ ¡Predicción correcta!")
                    else:
                        st.error("❌ Predicción incorrecta")
                        st.warning("🔄 El mapeo de género podría estar invertido")
                        
                        # Mostrar qué cambiar
                        st.markdown("### 💡 Solución:")
                        st.code("""
# En utils/predictor.py, cambia el mapeo:
# De:
self.clases_gender = {0: "Masculino", 1: "Femenino"}

# A:
self.clases_gender = {0: "Femenino", 1: "Masculino"}
                        """)
                
                except Exception as e:
                    st.error(f"Error en predicción: {e}")

if __name__ == "__main__":
    main()
