import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

try:
    import efficientnet_pytorch
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("⚠️ EfficientNet no disponible. Instala con: pip install efficientnet_pytorch")

# Modelo para clasificar género
class ModeloGender(nn.Module):
   
    def __init__(self, num_classes=2):
        super(ModeloGender, self).__init__()

        self.model = resnet50(pretrained=False)
        
        # Basado en el error: model.fc.1.weight, model.fc.4.weight
        # Esto indica: Dropout(0), Linear(1), ReLU(2), Dropout(3), Linear(4)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),           # índice 0 - sin parámetros
            nn.Linear(self.model.fc.in_features, 512),  # índice 1 - model.fc.1.weight/bias
            nn.ReLU(),                 # índice 2 - sin parámetros  
            nn.Dropout(0.5),           # índice 3 - sin parámetros
            nn.Linear(512, num_classes) # índice 4 - model.fc.4.weight/bias
        )

    def forward(self, x):
        return self.model(x)

# Modelo para clasificar edad
class ModeloEdad(nn.Module):
   
    def __init__(self, num_classes=7, base_model='efficientnet_b0'):
        super(ModeloEdad, self).__init__()
        
        if base_model == 'resnet50':
            self.model = resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
        elif base_model == 'efficientnet_b0':

            try:
                import efficientnet_pytorch
                self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
                self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
            except ImportError:
                print("⚠️ EfficientNet no disponible, usando ResNet50 como fallback")
                self.model = resnet50(pretrained=False)
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            raise ValueError(f'Modelo base {base_model} no soportado. Usa: resnet50 o efficientnet_b0')

    def forward(self, x):
        return self.model(x)

# Cargar modelo de género
def cargar_modelo_gender(pesos_path):
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Cargar el archivo
        checkpoint = torch.load(pesos_path, map_location=device, weights_only=False)
        print(f"📦 Cargando {pesos_path}")
        
        # Crear el modelo
        model = ModeloGender(num_classes=2)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:

                print(f"📊 Checkpoint completo detectado")
                model_weights = checkpoint['model_state_dict']
                
                new_state_dict = {}
                for key, value in model_weights.items():
                    new_key = f'model.{key}'  # Agregar prefijo 'model.'
                    new_state_dict[new_key] = value
                
                print(f"🔧 Agregando prefijo 'model.' a {len(model_weights)} claves")
                model.load_state_dict(new_state_dict)
                
                # Mostrar info adicional del checkpoint
                if 'epoch' in checkpoint:
                    print(f"📈 Epoch: {checkpoint['epoch']}")
                if 'val_acc' in checkpoint:
                    print(f"🎯 Val Accuracy: {checkpoint['val_acc']:.4f}")
                    
            elif 'model.fc.weight' in checkpoint:
                # State_dict ya tiene prefijo 'model.'
                print(f"📊 State_dict con prefijo 'model.' detectado")
                model.load_state_dict(checkpoint)
                
            else:
                raise Exception(f"Formato de checkpoint no reconocido. Claves: {list(checkpoint.keys())}")
                
        elif hasattr(checkpoint, 'forward'):
            # Es un modelo completo
            print(f"🧠 Modelo completo detectado")
            model = checkpoint
        else:
            raise Exception(f"Formato no soportado: {type(checkpoint)}")
            
        model.eval()
        model.to(device)
        print(f"✅ Modelo de género cargado exitosamente en {device}")
        return model, device
        
    except Exception as e:
        raise Exception(f"Error al cargar modelo de género desde {pesos_path}: {e}")

# Cargar modelo de edad
def cargar_modelo_edad(pesos_path, base_model='efficientnet_b0', num_classes=7):
    
    model = ModeloEdad(num_classes=num_classes, base_model=base_model)
    model.load_state_dict(torch.load(pesos_path, map_location=torch.device('cpu')))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device

# Transformaciones
def obtener_transformaciones():
    transformaciones = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transformaciones
