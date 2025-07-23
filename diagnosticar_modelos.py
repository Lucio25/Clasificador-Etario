#!/usr/bin/env python3
"""
Script para diagnosticar los modelos guardados
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import diagnosticar_modelo, AgeClassifierResNet

def main():
    print("🔍 Diagnóstico de modelos guardados")
    print("="*50)
    
    # Rutas de los modelos
    modelos = [
        "models/Gender_Model.pth",
        "models/Male_Model.pt",
        "models/Female_Model.pt"
    ]
    
    for modelo in modelos:
        print(f"\n📁 Analizando: {modelo}")
        print("-" * 30)
        diagnosticar_modelo(modelo)
        print()

if __name__ == "__main__":
    main()
