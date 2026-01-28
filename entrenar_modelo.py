#!/usr/bin/env python3
"""
Script para entrenar el modelo desde cero.
"""

import sys
sys.path.append('.')

from model import entrenar_y_guardar_modelo_completo

if __name__ == '__main__':
    print("🚀 Iniciando entrenamiento del modelo...")
    
    modelo, scaler, info = entrenar_y_guardar_modelo_completo()
    
    if modelo is not None:
        print("\n🎉 ¡Modelo entrenado y guardado exitosamente!")
        print(f"📊 Accuracy del modelo: {info['accuracy']:.4f}")
        print(f"🎯 Características: {len(info['features_para_modelo'])}")
        print("\n✅ Ahora puedes ejecutar: python app.py")
    else:
        print("\n❌ Error al entrenar el modelo. Revisa los errores anteriores.")
        sys.exit(1)