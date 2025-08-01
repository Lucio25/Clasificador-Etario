import streamlit as st
from PIL import Image
import time
import os
import torch
import torch.nn as nn
import torchvision.models as models
import efficientnet_pytorch  # Para los modelos de edad
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import gdown
import tempfile
from utils.predictor import Predictor

class AgeClassifier(nn.Module):
    """
    ARQUITECTURA CORRECTA: EfficientNet-B0 para clasificación de edad
    Usado en: Male_Model.pt, Female_Model.pt
    """
    def __init__(self, num_classes=7):
        super(AgeClassifier, self).__init__()

        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        # Modificar la capa final para el número de clases
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class AgeClassifierResNet(nn.Module):
    """
    ARQUITECTURA ALTERNATIVA: ResNet50 (para compatibilidad con modelos antiguos)
    """
    def __init__(self, num_classes=7):
        super(AgeClassifierResNet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def crear_grafico_genero(probabilidades_dict, prediccion):
    """Crea un gráfico de barras para las probabilidades de género"""
    generos = list(probabilidades_dict.keys())
    probabilidades = [prob * 100 for prob in probabilidades_dict.values()]  # Convertir a porcentajes
    
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
    # Ordenar por probabilidad (de mayor a menor)
    items_ordenados = sorted(probabilidades_dict.items(), key=lambda x: x[1], reverse=True)
    rangos = [item[0] for item in items_ordenados]
    probabilidades = [item[1] * 100 for item in items_ordenados]  # Convertir a porcentajes
    
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
@st.cache_data
def load_predictor():
    """Carga el predictor con cache para evitar recargas innecesarias"""
    try:
        if (os.path.exists(RUTA_MODELO_GENDER) and 
            os.path.exists(RUTA_MODELO_EDAD_HOMBRES) and 
            os.path.exists(RUTA_MODELO_EDAD_MUJERES)):
            return Predictor(RUTA_MODELO_GENDER, RUTA_MODELO_EDAD_HOMBRES, RUTA_MODELO_EDAD_MUJERES)
        else:
            st.error("❌ No se encontraron los modelos entrenados en la carpeta 'models/'")
            st.info("Archivos necesarios: Gender_Model.pth, Male_Model.pt, Female_Model.pt")
            return None
    except Exception as e:
        st.error(f"❌ Error al cargar los modelos: {str(e)}")
        return None

def age_to_range(age):
    """Convierte una edad numérica a rango etario"""
    if age <= 3:
        return "Bebé (0-3)"
    elif age <= 12:
        return "Niño/a (4-12)"
    elif age <= 23:
        return "Adolescente (13-23)"
    elif age <= 30:
        return "Joven Adulto (24-30)"
    elif age <= 44:
        return "Adulto (31-44)"
    elif age <= 60:
        return "Adulto Mayor (45-60)"
    else:
        return "Anciano/a (>60)"

def extract_age_from_filename(image_path):
    """Extrae la edad del nombre del archivo"""
    try:
        # Obtener solo el nombre del archivo sin la ruta
        filename = os.path.basename(image_path)
        
        # El formato es: edad_genero_raza_timestamp
        age_str = filename.split('_')[0]
        age = int(age_str)
        
        return age
    except (ValueError, IndexError):
        # Si no se puede extraer la edad, devolver una edad aleatoria
        import random
        return random.randint(20, 60)

def get_dataset_images():
    """Busca imágenes en múltiples ubicaciones posibles del dataset"""
    import glob
    import os
    
    # Lista de carpetas donde buscar imágenes
    possible_folders = [
        "game_images",
        "dataset",
        "data",
        "images",
        "fotos",
        "imagenes",
        "dataset/images",
        "data/images",
        ".",  # Carpeta actual
    ]
    
    # Extensiones de imagen soportadas
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
    
    all_images = []
    
    # Buscar en cada carpeta posible
    for folder in possible_folders:
        if os.path.exists(folder):
            for ext in extensions:
                pattern = os.path.join(folder, "**", ext)
                images = glob.glob(pattern, recursive=True)
                all_images.extend(images)
            
            # También buscar directamente en la carpeta
            for ext in extensions:
                pattern = os.path.join(folder, ext)
                images = glob.glob(pattern)
                all_images.extend(images)
    
    # Remover duplicados y ordenar
    all_images = list(set(all_images))
    all_images.sort()
    
    if all_images:
        
        # Si hay más de 10 imágenes, seleccionar 10 al azar
        if len(all_images) > 10:
            import random
            selected_images = random.sample(all_images, 10)
        else:
            selected_images = all_images
            st.info(f"📋 Usando todas las {len(all_images)} imágenes disponibles")
        
        # Extraer edades reales de los nombres de archivo
        image_data = []
        for img_path in selected_images:
            real_age = extract_age_from_filename(img_path)
            age_range = age_to_range(real_age)
            
            image_data.append((img_path, real_age, age_range))
        
        # Debug: mostrar información de algunas imágenes de muestra
        sample_size = min(5, len(image_data))  # Mostrar máximo 5 ejemplos
        for i in range(sample_size):
            img_path, real_age, age_range = image_data[i]
            filename = os.path.basename(img_path)
        
        return image_data
    
    return []

def get_google_drive_file_ids():
    """Lista de IDs de archivos de la carpeta de Google Drive o dataset local"""
    
    # Primero intentar encontrar imágenes en el dataset local
    dataset_images = get_dataset_images()
    if dataset_images:
        return [(img_path, real_age) for img_path, real_age, _ in dataset_images]
    
    # Si no hay dataset local, mostrar mensaje informativo
    st.warning("📂 No se encontraron imágenes en el dataset local")
    st.info("💡 **Para usar tu dataset:**")
    st.info("1. Copia tus imágenes a la carpeta 'game_images'")
    st.info("2. O colócalas en carpetas como 'dataset', 'data', 'images'")
    st.info("3. El sistema las detectará automáticamente")
    
    # Crear imágenes placeholder mientras tanto
    return create_placeholder_images()

def create_placeholder_images():
    """Crea imágenes placeholder como último recurso"""
    from PIL import Image, ImageDraw
    import tempfile
    
    placeholder_data = []
    
    for i in range(10):
        # Crear imagen placeholder
        img = Image.new('RGB', (300, 300), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Dibujar un rostro simple
        # Cara
        draw.ellipse([75, 75, 225, 225], fill=(255, 220, 177), outline=(0, 0, 0), width=2)
        # Ojos
        draw.ellipse([110, 120, 130, 140], fill=(0, 0, 0))
        draw.ellipse([170, 120, 190, 140], fill=(0, 0, 0))
        # Nariz
        draw.line([150, 150, 150, 170], fill=(0, 0, 0), width=2)
        # Boca
        draw.arc([130, 180, 170, 200], 0, 180, fill=(0, 0, 0), width=2)
        
        # Texto
        draw.text((120, 240), f"Persona {i+1}", fill=(0, 0, 0))
        
        # Guardar imagen temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            img.save(tmp_file.name, 'JPEG')
            age = [25, 34, 18, 67, 42, 29, 55, 31, 73, 22][i]
            placeholder_data.append((tmp_file.name, age))
    
    st.warning("⚠️ Usando imágenes placeholder temporales")
    return placeholder_data

@st.cache_data
def download_game_image(file_id):
    """Descarga una imagen de Google Drive usando su ID"""
    # Si es una ruta local (contiene / o \), usarla directamente
    if '/' in file_id or '\\' in file_id:
        if os.path.exists(file_id):
            return file_id
        else:
            st.error(f"❌ No se encontró la imagen local: {file_id}")
            return None
    
    try:
        # Intentar con Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            gdown.download(url, tmp_file.name, quiet=True)
            return tmp_file.name
    except Exception as e:
        # Si falla, mostrar error pero continuar el juego con imagen placeholder
        st.warning(f"⚠️ No se pudo descargar la imagen de Google Drive. ID: {file_id}")
        st.info("💡 **Soluciones posibles:**")
        st.info("1. Obtén los IDs reales de tus imágenes de Google Drive")
        st.info("2. Configura los permisos como 'Cualquier persona con el enlace'")
        st.info("3. O copia imágenes a la carpeta 'game_images/' para usar imágenes locales")
        
        # Crear una imagen placeholder simple
        from PIL import Image, ImageDraw, ImageFont
        
        # Crear imagen placeholder
        img = Image.new('RGB', (300, 300), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # Agregar texto
        text = f"Imagen de prueba\nID: {file_id[:15]}..."
        draw.text((30, 130), text, fill='black')
        
        # Guardar imagen temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            img.save(tmp_file.name, 'JPEG')
            return tmp_file.name

def initialize_game_state():
    """Inicializa el estado del juego"""
    if 'game_mode' not in st.session_state:
        st.session_state.game_mode = False
    
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
    
    if 'game_round' not in st.session_state:
        st.session_state.game_round = 0
    
    if 'user_score' not in st.session_state:
        st.session_state.user_score = 0
    
    if 'ai_score' not in st.session_state:
        st.session_state.ai_score = 0
    
    if 'game_images' not in st.session_state:
        st.session_state.game_images = get_google_drive_file_ids()
        random.shuffle(st.session_state.game_images)
    
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None
    
    if 'user_prediction' not in st.session_state:
        st.session_state.user_prediction = None
    
    if 'ai_prediction' not in st.session_state:
        st.session_state.ai_prediction = None
    
    if 'real_age' not in st.session_state:
        st.session_state.real_age = None
    
    if 'real_age_range' not in st.session_state:
        st.session_state.real_age_range = None
    
    if 'round_finished' not in st.session_state:
        st.session_state.round_finished = False

def start_new_game():
    """Inicia un nuevo juego"""
    st.session_state.game_round = 0
    st.session_state.user_score = 0
    st.session_state.ai_score = 0
    st.session_state.game_images = get_google_drive_file_ids()
    random.shuffle(st.session_state.game_images)
    st.session_state.current_image_path = None
    st.session_state.user_prediction = None
    st.session_state.ai_prediction = None
    st.session_state.real_age = None
    st.session_state.real_age_range = None
    st.session_state.round_finished = False
    st.session_state.game_started = True

def range_to_numeric(age_range):
    """Convierte un rango etario a edad numérica promedio para cálculos"""
    range_mapping = {
        "Bebé (0-3)": 2,
        "Niño/a (4-12)": 8,
        "Adolescente (13-23)": 18,
        "Joven Adulto (24-30)": 27,
        "Adulto (31-44)": 37,
        "Adulto Mayor (45-60)": 52,
        "Anciano/a (>60)": 70
    }
    return range_mapping.get(age_range, 30)

def calculate_score_by_range(user_range, ai_range, real_range):
    """Calcula el puntaje basado en la precisión de predicción de rangos"""
    # Puntuación máxima si coincide exactamente
    if user_range == real_range:
        user_score = 100
    else:
        user_score = 0
    
    if ai_range == real_range:
        ai_score = 100
    else:
        ai_score = 0
    
    return user_score, ai_score

def load_next_round():
    """Carga la siguiente ronda del juego"""
    if st.session_state.game_round < 10:
        if len(st.session_state.game_images) > st.session_state.game_round:
            file_id, real_age = st.session_state.game_images[st.session_state.game_round]
            st.session_state.real_age = real_age
            st.session_state.real_age_range = age_to_range(real_age)
            
            # Descargar imagen
            image_path = download_game_image(file_id)
            if image_path:
                st.session_state.current_image_path = image_path
                st.session_state.round_finished = False
                return True
    return False

def calculate_score(prediction, real_age):
    """Calcula el puntaje basado en la precisión de la predicción"""
    error = abs(prediction - real_age)
    if error == 0:
        return 100
    elif error <= 2:
        return 80
    elif error <= 5:
        return 60
    elif error <= 10:
        return 40
    elif error <= 15:
        return 20
    else:
        return 0

def game_mode():
    """Interfaz del modo de juego"""
    st.title("🎮 Modo Juego: Usuario vs IA")
    
    initialize_game_state()
    
    # Botón para nuevo juego
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Nuevo Juego", type="primary", use_container_width=True):
            start_new_game()
            st.rerun()
    
    if not st.session_state.get('game_started', False):
        st.info("👆 Haz clic en 'Nuevo Juego' para comenzar")
        return
    
    # Mostrar estadísticas del juego
    st.markdown("### 📊 Estadísticas")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ronda", f"{st.session_state.game_round + 1}/10")
    with col2:
        st.metric("👤 Usuario", st.session_state.user_score)
    with col3:
        st.metric("🤖 IA", st.session_state.ai_score)
    with col4:
        diferencia = st.session_state.user_score - st.session_state.ai_score
        st.metric("Diferencia", f"{diferencia:+d}")
    
    # Cargar imagen si no hay una cargada
    if st.session_state.current_image_path is None and st.session_state.game_round < 10:
        if load_next_round():
            st.rerun()
    
    # Mostrar imagen actual
    if st.session_state.current_image_path:
        st.markdown("### 🖼️ Imagen Actual")
        
        try:
            image = Image.open(st.session_state.current_image_path)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption=f"Ronda {st.session_state.game_round + 1}", use_container_width=True)
        except Exception as e:
            st.error(f"Error cargando imagen: {e}")
            return
        
        if not st.session_state.round_finished:
            # Entrada del usuario - SELECCIÓN POR RANGOS
            st.markdown("### 🎯 Tu Predicción")
            st.markdown("**Selecciona el rango etario que crees que corresponde a esta persona:**")
            
            age_ranges = [
                "-- Seleccione el rango etario --",
                "Bebé (0-3)",
                "Niño/a (4-12)", 
                "Adolescente (13-23)",
                "Joven Adulto (24-30)",
                "Adulto (31-44)",
                "Adulto Mayor (45-60)",
                "Anciano/a (>60)"
            ]
            
            user_age_range = st.selectbox(
                "🎂 Rango etario:",
                age_ranges,
                index=0  # Valor por defecto: "Seleccione el rango etario"
            )
            
            if st.button("🚀 Hacer Predicción", type="primary"):
                # Validar que se haya seleccionado un rango válido
                if user_age_range == "-- Seleccione el rango etario --":
                    st.error("❌ Por favor selecciona un rango etario antes de hacer la predicción")
                    return
                
                # Predicción del usuario
                st.session_state.user_prediction = user_age_range
                
                # Predicción de la IA
                predictor = load_predictor()
                if predictor is None:
                    st.error("❌ No se pueden realizar predicciones. Los modelos no están cargados.")
                    return
                    
                try:
                    # Cargar la imagen como PIL Image antes de pasarla al predictor
                    imagen_pil = Image.open(st.session_state.current_image_path).convert("RGB")
                    resultado = predictor.predecir_completo(imagen_pil)
                    # La predicción de la IA ya viene como rango
                    ai_age_range = resultado['edad']  # Esto ya es un rango como string
                    
                    st.session_state.ai_prediction = ai_age_range
                    
                    # Calcular puntajes basados en rangos
                    user_round_score, ai_round_score = calculate_score_by_range(
                        user_age_range, 
                        ai_age_range, 
                        st.session_state.real_age_range
                    )
                    
                    st.session_state.user_score += user_round_score
                    st.session_state.ai_score += ai_round_score
                    st.session_state.round_finished = True
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error en predicción de IA: {e}")
        
        else:
            # Mostrar resultados de la ronda
            st.markdown("### 📋 Resultados de la Ronda")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"👤 **Tu predicción:** {st.session_state.user_prediction}")
                user_correct = st.session_state.user_prediction == st.session_state.real_age_range
                user_round_score = 100 if user_correct else 0
                st.write(f"✅ Correcto" if user_correct else "❌ Incorrecto")
                st.write(f"Puntos: {user_round_score}")
            
            with col2:
                st.success(f"✅ **Rango real:** {st.session_state.real_age_range}")
                st.write(f"**Edad específica:** {st.session_state.real_age} años")
            
            with col3:
                st.warning(f"🤖 **IA predijo:** {st.session_state.ai_prediction}")
                ai_correct = st.session_state.ai_prediction == st.session_state.real_age_range
                ai_round_score = 100 if ai_correct else 0
                st.write(f"✅ Correcto" if ai_correct else "❌ Incorrecto")
                st.write(f"Puntos: {ai_round_score}")
            
            # Mostrar comparación detallada
            st.markdown("---")
            st.markdown("### 🎯 Comparación de Predicciones")
            
            comparison_data = {
                "Predictor": ["👤 Usuario", "🤖 IA", "✅ Real"],
                "Rango Etario": [
                    st.session_state.user_prediction,
                    st.session_state.ai_prediction,
                    st.session_state.real_age_range
                ],
                "Resultado": [
                    "✅ Correcto" if st.session_state.user_prediction == st.session_state.real_age_range else "❌ Incorrecto",
                    "✅ Correcto" if st.session_state.ai_prediction == st.session_state.real_age_range else "❌ Incorrecto",
                    "📍 Referencia"
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            # Botón para siguiente ronda
            if st.session_state.game_round < 9:
                if st.button("➡️ Siguiente Ronda", type="primary"):
                    st.session_state.game_round += 1
                    st.session_state.current_image_path = None
                    st.session_state.user_prediction = None
                    st.session_state.ai_prediction = None
                    st.session_state.real_age = None
                    st.session_state.real_age_range = None
                    st.session_state.round_finished = False
                    st.rerun()
            else:
                # Juego terminado
                st.markdown("### 🏆 ¡Juego Terminado!")
                
                if st.session_state.user_score > st.session_state.ai_score:
                    st.balloons()
                    st.success(f"🎉 ¡Ganaste! Puntuación final: Usuario {st.session_state.user_score} - IA {st.session_state.ai_score}")
                elif st.session_state.user_score < st.session_state.ai_score:
                    st.error(f"😔 La IA ganó esta vez. Puntuación final: Usuario {st.session_state.user_score} - IA {st.session_state.ai_score}")
                else:
                    st.info(f"🤝 ¡Empate! Puntuación final: Usuario {st.session_state.user_score} - IA {st.session_state.ai_score}")

def prediction_mode():
    """Interfaz del modo predicción original"""
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
            predictor_instance = load_predictor()
            if predictor_instance is None:
                st.error("❌ No se pueden realizar predicciones. Los modelos no están cargados.")
            else:
                with st.spinner("Analizando la imagen..."):
                    try:
                        # Realizar predicción con los modelos reales
                        resultado = predictor_instance.predecir_completo(imagen_pil)
                        
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
                            
                            # Agregar indicador visual directamente en el texto
                            for idx, row in tabla_mostrar.iterrows():
                                fila_original = df_edad[df_edad['Rango'] == row['Rango']].iloc[0]
                                if fila_original['Es_Predicho']:
                                    tabla_mostrar.at[idx, 'Rango'] = f"👑 {row['Rango']}"
                            
                            st.dataframe(
                                tabla_mostrar,
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

navbar = st.sidebar
navbar.title("Clasificador de Edad y Sexo")
navbar.markdown("---")

# MODO DE APLICACIÓN
st.sidebar.markdown("### 🎮 Modo de Aplicación")
modo = st.sidebar.radio(
    "Selecciona el modo:",
    ["🔍 Modo Predicción", "🎯 Modo Juego"],
    key="app_mode"
)

navbar.markdown("---")
navbar.markdown("### Redes Neuronales Profundas")
navbar.markdown("### Integrantes")
navbar.markdown("""
- **Lucio Malgioglio**
- **Juan Manuel Valdivia**
- **Leonardo Secotaro**
""")

# Mostrar la interfaz según el modo seleccionado
if modo == "🔍 Modo Predicción":
    prediction_mode()
else:
    game_mode()
