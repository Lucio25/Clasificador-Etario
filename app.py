import streamlit as st
from PIL import Image
import time
import os
import torch
import torch.nn as nn
import torchvision.models as models
from utils.predictor import Predictor

# Definir la clase AgeClassifierResNet aquí para que esté disponible en __main__
class AgeClassifierResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(AgeClassifierResNet, self).__init__()
        # Usar ResNet50 como base - ajustar según tu modelo
        self.model = models.resnet50(pretrained=False)
        # Modificar la capa final para el número de clases
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

st.set_page_config(page_title="Clasificador Edad y Sexo", layout="centered")

# Configuración de rutas de modelos
RUTA_MODELO_GENDER = "models/Gender_Model.pth"
RUTA_MODELO_EDAD_HOMBRES = "models/Male_Model.pt"
RUTA_MODELO_EDAD_MUJERES = "models/Female_Model.pt"

# Inicializar predictor (con cache para evitar cargar modelos múltiples veces)
@st.cache_resource
def cargar_predictor():
    """Carga el predictor con los modelos entrenados"""
    try:
        if (os.path.exists(RUTA_MODELO_GENDER) and 
            os.path.exists(RUTA_MODELO_EDAD_HOMBRES) and 
            os.path.exists(RUTA_MODELO_EDAD_MUJERES)):
            predictor = Predictor(RUTA_MODELO_GENDER, RUTA_MODELO_EDAD_HOMBRES, RUTA_MODELO_EDAD_MUJERES)
            return predictor
        else:
            st.error("❌ No se encontraron los modelos entrenados. Por favor, coloca los archivos .pth en la carpeta 'models/'")
            st.info("Archivos necesarios: modelo_gender.pth, modelo_edad_hombres.pth, modelo_edad_mujeres.pth")
            return None
    except Exception as e:
        st.error(f"❌ Error al cargar los modelos: {str(e)}")
        return None

# Cargar predictor
predictor = cargar_predictor()

navbar = st.sidebar
navbar.title("Clasificador de Edad y Sexo")
navbar.markdown("---")
navbar.markdown("### Redes Neuronales Profundas")
navbar.markdown("### Integrantes")
navbar.markdown("""
- **Lucio Malgioglio**
- **Juan Manuel Valdivia**
- **Leonardo Secotaro**
""")

st.title("Clasificador de Edad y Sexo con IA")

# Subida de imagen
imagen = st.file_uploader("📸 Subí una imagen de rostro humano", type=["jpg", "jpeg", "png"])

if imagen:
    imagen_pil = Image.open(imagen).convert("RGB")
    st.image(imagen_pil, caption="Imagen cargada", use_container_width=True)

    st.markdown("### Selecciona la edad real de la persona, o un valor aproximado")
    edad = st.slider(
        "Edad de la persona",
        min_value=0,
        max_value=100,
        value=30,
        step=1
        )

    # Botón de predicción
    if st.button("🔍 Clasificar Sexo y Rango Etario"):
        if predictor is None:
            st.error("❌ No se pueden realizar predicciones. Los modelos no están cargados.")
        else:
            with st.spinner("Analizando la imagen..."):
                try:
                    # Realizar predicción con los modelos reales
                    resultado = predictor.predecir_completo(imagen_pil)
                    
                    sexo_predicho = resultado['sexo']
                    edad_predicha = resultado['edad']
                    confianza_sexo = resultado['confianza_sexo']
                    confianza_edad = resultado['confianza_edad']
                    
                    # Determinar el rango etario real según el slider
                    if edad <= 3:
                        rango_real = "Bebé (0-3)"
                    elif edad <= 12:
                        rango_real = "Niño/a (4-12)"
                    elif edad <= 23:
                        rango_real = "Adolescente (13-23)"
                    elif edad <= 30:
                        rango_real = "Joven Adulto (24-30)"
                    elif edad <= 44:
                        rango_real = "Adulto (31-44)"
                    elif edad <= 60:
                        rango_real = "Adulto Mayor (45-60)"
                    else:
                        rango_real = "Anciano/a (>60)"
                        
                    st.markdown("---")
                    st.success("✅ Clasificación completada")
                    
                    # Mostrar resultados con confianza
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**🚹🚺 Género detectado:** {sexo_predicho}")
                        st.markdown(f"*Confianza: {confianza_sexo:.2%}*")
                    with col2:
                        st.markdown(f"**🎂 Rango etario estimado:** {edad_predicha}")
                        st.markdown(f"*Confianza: {confianza_edad:.2%}*")
                    
                    # Mostrar el flujo de predicción
                    st.info(f"🔄 **Flujo de predicción:** Género ({sexo_predicho}) → Modelo de edad específico → Rango etario")
                    
                    st.markdown("---")
                    st.markdown(f"👤 **Edad real seleccionada:** {edad} años")
                    st.markdown(f"📊 **Rango real según edad ingresada:** {rango_real}")
                    
                    # Comparación de resultados
                    if edad_predicha == rango_real:
                        st.success("🎯 ¡La predicción coincidió con la edad real!")
                    else:
                        st.warning("⚠️ La predicción no coincidió con la edad real.")
                
                except Exception as e:
                    st.error(f"❌ Error durante la predicción: {str(e)}")
                    st.info("🔄 Asegúrate de que los modelos estén correctamente entrenados y guardados.")

else:
    st.info("📂 Esperando imagen...")
