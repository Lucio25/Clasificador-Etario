import streamlit as st
from PIL import Image
import time
import os
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

def crear_grafico_genero(probabilidades_dict, prediccion):
    """Crea un gráfico de barras para las probabilidades de género"""
    generos = list(probabilidades_dict.keys())
    probabilidades = [prob * 100 for prob in probabilidades_dict.values()]  # Convertir a porcentajes
    
    # Colores: verde para la predicción, azul para el resto
    colores = ['#2E8B57' if genero == prediccion else '#4682B4' for genero in generos]
    
    fig = go.Figure(data=[
        go.Bar(
            x=generos,
            y=probabilidades,
            marker_color=colores,
            text=[f'{prob:.1f}%' for prob in probabilidades],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'🚹🚺 Probabilidades de Género (Predicción: {prediccion})',
        xaxis_title='Género',
        yaxis_title='Probabilidad (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False
    )
    
    # Agregar línea de referencia en 50%
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="50% - Umbral de decisión")
    
    return fig

def crear_grafico_edad(probabilidades_dict, prediccion):
    """Crea un gráfico de barras para las probabilidades de edad"""
    # Ordenar por probabilidad (de mayor a menor)
    items_ordenados = sorted(probabilidades_dict.items(), key=lambda x: x[1], reverse=True)
    rangos = [item[0] for item in items_ordenados]
    probabilidades = [item[1] * 100 for item in items_ordenados]  # Convertir a porcentajes
    
    # Colores: verde para la predicción, azul para el resto
    colores = []
    for rango in rangos:
        if rango == prediccion:
            colores.append('#2E8B57')  # Verde para la predicción
        else:
            colores.append('#4682B4')  # Azul para el resto
    
    fig = go.Figure(data=[
        go.Bar(
            x=rangos,
            y=probabilidades,
            marker_color=colores,
            text=[f'{prob:.1f}%' for prob in probabilidades],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'🎂 Probabilidades de Rangos Etarios (Predicción: {prediccion})',
        xaxis_title='Rango de Edad',
        yaxis_title='Probabilidad (%)',
        yaxis=dict(range=[0, max(probabilidades) * 1.1]),
        height=500,
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig

st.set_page_config(page_title="Clasificador Edad y Sexo", layout="centered")
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
                    
                    # DEBUG: Mostrar resultado completo en consola
                    print("\n" + "="*50)
                    print("DEBUG - RESULTADO COMPLETO DE LA PREDICCIÓN:")
                    print("="*50)
                    for key, value in resultado.items():
                        print(f"{key}: {value}")
                        if isinstance(value, torch.Tensor):
                            print(f"  -> Tipo: Tensor, Shape: {value.shape}, Valores: {value}")
                        elif isinstance(value, (list, tuple)):
                            print(f"  -> Tipo: {type(value).__name__}, Longitud: {len(value)}")
                        print()
                    print("="*50)
                    
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
                    
                    # Mostrar resultados principales con confianza
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**🚹🚺 Género detectado:** {sexo_predicho}")
                        st.markdown(f"*Confianza: {confianza_sexo:.2%}*")
                    with col2:
                        st.markdown(f"**🎂 Rango etario estimado:** {edad_predicha}")
                        st.markdown(f"*Confianza: {confianza_edad:.2%}*")
                    
                    # Mostrar el flujo de predicción
                    st.info(f"🔄 **Flujo de predicción:** Género ({sexo_predicho}) → Modelo {resultado.get('modelo_usado', 'N/A')} → Rango etario")
                    
                    # SECCIÓN NUEVA: Mostrar todas las probabilidades
                    st.markdown("---")
                    st.markdown("## 📊 Probabilidades Detalladas")
                    
                    # Probabilidades de Género
                    st.markdown("### 🚹🚺 Probabilidades de Género")
                    if 'probabilidades_sexo_dict' in resultado:
                        col_masc, col_fem = st.columns(2)
                        probs_sexo = resultado['probabilidades_sexo_dict']
                        
                        with col_masc:
                            prob_masc = probs_sexo.get('Masculino', 0)
                            delta_masc = "👑 PREDICHO" if sexo_predicho == 'Masculino' else ""
                            st.metric(
                                label="👨 Masculino",
                                value=f"{prob_masc:.2%}",
                                delta=delta_masc,
                                delta_color="normal" if sexo_predicho == 'Masculino' else "off"
                            )
                            
                        with col_fem:
                            prob_fem = probs_sexo.get('Femenino', 0)
                            delta_fem = "👑 PREDICHO" if sexo_predicho == 'Femenino' else ""
                            st.metric(
                                label="👩 Femenino",
                                value=f"{prob_fem:.2%}",
                                delta=delta_fem,
                                delta_color="normal" if sexo_predicho == 'Femenino' else "off"
                            )
                        
                        # Barra de progreso visual para género
                        st.markdown("**Comparación visual:**")
                        st.progress(prob_masc, text=f"Masculino: {prob_masc:.1%}")
                        st.progress(prob_fem, text=f"Femenino: {prob_fem:.1%}")
                    
                    # Probabilidades de Edad
                    st.markdown("### 🎂 Probabilidades de Rangos Etarios")
                    if 'probabilidades_edad_dict' in resultado:
                        probs_edad = resultado['probabilidades_edad_dict']
                        
                        # Crear DataFrame y ordenar por probabilidad
                        df_edad = pd.DataFrame([
                            {
                                'Rango': rango, 
                                'Probabilidad': prob,
                                'Porcentaje': f"{prob:.2%}",
                                'Es_Predicho': rango == edad_predicha
                            }
                            for rango, prob in probs_edad.items()
                        ]).sort_values('Probabilidad', ascending=False)
                        
                        # Mostrar las 3 más altas como métricas destacadas
                        st.markdown("**🏆 Top 3 rangos más probables:**")
                        cols = st.columns(3)
                        for idx, (_, row) in enumerate(df_edad.head(3).iterrows()):
                            with cols[idx]:
                                delta_text = "👑 PREDICHO" if row['Es_Predicho'] else f"#{idx+1}"
                                st.metric(
                                    label=row['Rango'],
                                    value=row['Porcentaje'],
                                    delta=delta_text,
                                    delta_color="normal" if row['Es_Predicho'] else "off"
                                )
                        
                        # Mostrar todas las probabilidades como barras de progreso
                        st.markdown("**📈 Todas las probabilidades:**")
                        for _, row in df_edad.iterrows():
                            emoji = "👑" if row['Es_Predicho'] else "📊"
                            color = "#1f77b4" if not row['Es_Predicho'] else "#ff7f0e"
                            st.progress(
                                row['Probabilidad'], 
                                text=f"{emoji} {row['Rango']}: {row['Porcentaje']}"
                            )
                        
                        # Tabla completa ordenada
                        st.markdown("**📋 Tabla completa ordenada por probabilidad:**")
                        
                        # Mostrar tabla sin la columna auxiliar
                        tabla_mostrar = df_edad[['Rango', 'Porcentaje']].copy()
                        
                        # Aplicar estilo a la tabla de forma más simple
                        def color_predicho(val):
                            # Buscar si este rango es el predicho
                            rango_actual = val.name if hasattr(val, 'name') else None
                            fila_actual = df_edad[df_edad['Rango'] == tabla_mostrar.iloc[val.name]['Rango']].iloc[0] if hasattr(val, 'name') and val.name < len(tabla_mostrar) else None
                            
                            if fila_actual is not None and fila_actual['Es_Predicho']:
                                return ['background-color: #90EE90; font-weight: bold'] * len(val)
                            else:
                                return [''] * len(val)
                        
                        # Forma alternativa más simple: colorear solo por contenido
                        tabla_final = tabla_mostrar.copy()
                        
                        # Agregar indicador visual directamente en el texto
                        for idx, row in tabla_final.iterrows():
                            fila_original = df_edad[df_edad['Rango'] == row['Rango']].iloc[0]
                            if fila_original['Es_Predicho']:
                                tabla_final.at[idx, 'Rango'] = f"👑 {row['Rango']}"
                        
                        st.dataframe(
                            tabla_final,
                            use_container_width=True,
                            hide_index=True
                        )
                    
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
