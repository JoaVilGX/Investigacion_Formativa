import joblib
import numpy as np

modelo = joblib.load('models/modelo_entrenado.pkl')

print("🔍 ANÁLISIS DEL MODELO")
print("="*60)

if hasattr(modelo, 'feature_names_in_'):
    features = list(modelo.feature_names_in_)
    print(f"📊 El modelo tiene {len(features)} características:")
    
    # Contar ocurrencias
    from collections import Counter
    contador = Counter(features)
    
    print("\n📋 Conteo de características:")
    for feature, count in contador.items():
        if count > 1:
            print(f"  ❌ {feature}: {count} veces (DUPLICADO!)")
        else:
            print(f"  ✅ {feature}: {count} vez")
    
    # Mostrar orden completo
    print("\n📋 Orden completo de características:")
    for i, feat in enumerate(features, 1):
        print(f"  {i:2}. {feat}")
else:
    print("❌ El modelo no tiene 'feature_names_in_'")
    
print("="*60)