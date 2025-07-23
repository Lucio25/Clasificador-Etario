# Clasificador de Edad y Género con IA

## Descripción del Proyecto

Este proyecto utiliza **3 modelos de PyTorch** para clasificar edad y género de personas en imágenes:

1. **Modelo de Género** → Clasifica si es masculino o femenino
2. **Modelo de Edad Hombres** → Predice rangos etarios específicos para hombres
3. **Modelo de Edad Mujeres** → Predice rangos etarios específicos para mujeres

## Flujo de Predicción

```
Imagen → Modelo Gender → Género detectado → Modelo específico de edad → Rango etario
```

## Estructura del Proyecto

```
Proyecto/
├── app.py                          # Aplicación Streamlit principal
├── models/                         # Modelos entrenados
│   ├── modelo_gender.pth          # Modelo de género
│   ├── modelo_edad_hombres.pth    # Modelo de edad para hombres
│   └── modelo_edad_mujeres.pth    # Modelo de edad para mujeres
├── utils/                          # Código de soporte
│   ├── __init__.py
│   ├── model_loader.py            # Carga los modelos PyTorch
│   └── predictor.py               # Lógica de predicción
├── requirements.txt               # Dependencias
└── README.md                      # Este archivo
```

## Rangos Etarios Utilizados

- **0**: Bebé (0-3)
- **1**: Niño/a (4-12)
- **2**: Adolescente (13-23)
- **3**: Joven Adulto (24-30)
- **4**: Adulto (31-44)
- **5**: Adulto Mayor (45-60)
- **6**: Anciano/a (>60)

# Instalación y Configuración

## Uso de Git LFS para archivos grandes

Este proyecto contiene modelos grandes (**`.pth`**) que se manejan con **Git Large File Storage (Git LFS)**. Para clonar y trabajar con el repositorio correctamente, sigue estos pasos:

1. **Instalar Git LFS:**

- **macOS (Homebrew):**

```bash
brew install git-lfs
```
- **Ubuntu:**

```
sudo apt-get install git-lfs
```

- **Windows:**
```
Descarga e instala desde https://git-lfs.github.com/
```

2. **Inicializar Git LFS (solo una vez):**

```
git lfs install
```

3. **Clonar el repositorio normalmente:**

```
git clone https://github.com/Lucio25/Clasificador-Etario.git
```

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Colocar los modelos entrenados
Coloca tus archivos `.pth` en la carpeta `models/` con los nombres exactos:
- `modelo_gender.pth`
- `modelo_edad_hombres.pth`
- `modelo_edad_mujeres.pth`

### 3. Ajustar las arquitecturas de los modelos
En `utils/model_loader.py`, modifica las clases:
- `ModeloGender`
- `ModeloEdadHombres`
- `ModeloEdadMujeres`

Para que coincidan exactamente con la arquitectura que usaste en tus Colabs.

### 4. Ejecutar la aplicación
```bash
streamlit run app.py
```

## Uso

1. Sube una imagen de rostro humano
2. Selecciona la edad real de la persona (para comparar)
3. Haz clic en "Clasificar Sexo y Rango Etario"
4. Ve los resultados con confianza/probabilidad

## Características

- **Predicción en 3 pasos**: Género → Modelo específico → Edad
- **Interfaz intuitiva** con Streamlit
- **Mostrar confianza** de las predicciones
- **Comparación** con edad real ingresada
- **Manejo de errores** robusto
- **Cache inteligente** para cargar modelos una sola vez

## Integrantes

- **Lucio Malgioglio**
- **Juan Manuel Valdivia**
- **Leonardo Secotaro**

## Dataset

Utiliza el dataset **UTKFace** donde cada imagen tiene información de edad y género en el nombre del archivo.

## Tecnologías Utilizadas

- **PyTorch** para los modelos de deep learning
- **Streamlit** para la interfaz web
- **PIL/Pillow** para procesamiento de imágenes
- **NumPy** para operaciones numéricas
