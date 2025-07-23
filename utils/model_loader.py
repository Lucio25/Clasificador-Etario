import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models

# Definir la clase AgeClassifierResNet que está en tus modelos guardados
class AgeClassifierResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(AgeClassifierResNet, self).__init__()
        # Usar ResNet18 como base - ajustar según tu modelo
        self.model = models.resnet18(pretrained=False)
        # Modificar la capa final para el número de clases
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class ModeloGender(nn.Module):
    """
    Modelo para clasificar género (masculino/femenino) - ResNet50
    """
    def __init__(self, num_classes=2):
        super(ModeloGender, self).__init__()
        # Usar ResNet50 como base (basado en el error, tu modelo es ResNet50)
        self.resnet = models.resnet50(pretrained=False)
        # Modificar la capa final para 2 clases (masculino/femenino)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

def cargar_modelo_gender(ruta_modelo):
    """Carga el modelo de género - ResNet50"""
    modelo = ModeloGender()
    
    # Verificar si hay GPU disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(ruta_modelo):
        # Cargar el checkpoint con weights_only=False
        checkpoint = torch.load(ruta_modelo, map_location=device, weights_only=False)
        
        # Verificar si el checkpoint tiene la estructura completa o solo el state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Mapear las claves del state_dict si es necesario
        new_state_dict = {}
        for key, value in state_dict.items():
            # Si la clave no tiene prefijo "resnet.", agregarla
            if key.startswith('resnet.'):
                new_key = key
            else:
                new_key = 'resnet.' + key
            new_state_dict[new_key] = value
        
        # Cargar el state_dict con strict=False para ignorar claves incompatibles
        modelo.load_state_dict(new_state_dict, strict=False)
        
        modelo.to(device)
        modelo.eval()
        return modelo, device
    else:
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo}")

def cargar_modelo_edad_hombres(ruta_modelo):
    """Carga el modelo de edad para hombres"""
    # Verificar si hay GPU disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(ruta_modelo):
        try:
            # Intentar cargar el modelo completo (AgeClassifierResNet)
            modelo = torch.load(ruta_modelo, map_location=device, weights_only=False)
            if hasattr(modelo, 'eval'):
                modelo.eval()
                modelo.to(device)
                return modelo, device
            else:
                raise Exception("El objeto cargado no es un modelo válido")
        except Exception as e:
            print(f"❌ Error al cargar modelo completo: {e}")
            raise Exception(f"Error al cargar modelo de edad para hombres: {e}")
    else:
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo}")

def cargar_modelo_edad_mujeres(ruta_modelo):
    """Carga el modelo de edad para mujeres"""
    # Verificar si hay GPU disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(ruta_modelo):
        try:
            # Intentar cargar el modelo completo (AgeClassifierResNet)
            modelo = torch.load(ruta_modelo, map_location=device, weights_only=False)
            if hasattr(modelo, 'eval'):
                modelo.eval()
                modelo.to(device)
                return modelo, device
            else:
                raise Exception("El objeto cargado no es un modelo válido")
        except Exception as e:
            print(f"❌ Error al cargar modelo completo: {e}")
            raise Exception(f"Error al cargar modelo de edad para mujeres: {e}")
    else:
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo}")

def obtener_transformaciones():
    """Transformaciones para preprocesar las imágenes"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ajusta según el tamaño de entrada de tu modelo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def diagnosticar_modelo(ruta_modelo):
    """Función para diagnosticar qué contiene el archivo del modelo"""
    if not os.path.exists(ruta_modelo):
        print(f"❌ No se encontró el archivo: {ruta_modelo}")
        return
    
    try:
        # Intentar cargar con weights_only=False para modelos que tienen clases personalizadas
        checkpoint = torch.load(ruta_modelo, map_location='cpu', weights_only=False)
        print(f"📋 Diagnóstico del modelo: {ruta_modelo}")
        print(f"📦 Tipo de objeto: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"🔑 Claves encontradas: {list(checkpoint.keys())}")
            
            # Buscar el state_dict
            state_dict = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("✅ Encontrado 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✅ Encontrado 'state_dict'")
            elif all(isinstance(k, str) and ('weight' in k or 'bias' in k) for k in checkpoint.keys()):
                state_dict = checkpoint
                print("✅ Parece ser un state_dict directo")
            
            if state_dict:
                print(f"🧠 Capas en el modelo: {list(state_dict.keys())[:10]}...")  # Solo las primeras 10
                print(f"📊 Total de parámetros: {len(state_dict)}")
            else:
                print("❌ No se encontró state_dict válido")
                
        else:
            print("❌ El checkpoint no es un diccionario")
            print(f"📝 Contenido: {str(checkpoint)[:200]}...")
            
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        # Intentar con weights_only=True
        try:
            checkpoint = torch.load(ruta_modelo, map_location='cpu', weights_only=True)
            print("✅ Cargado con weights_only=True")
        except Exception as e2:
            print(f"❌ También falló con weights_only=True: {e2}")
