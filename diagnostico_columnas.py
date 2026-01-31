import json
import pandas as pd

# 1. Verificamos características del modelo
with open('models/info_flask.json', 'r') as f:
    info = json.load(f)

features_modelo = info['features_para_modelo']
print(f"📋 Modelo espera {len(features_modelo)} características:")
for i, f in enumerate(features_modelo, 1):
    print(f"  {i:2}. {f}")

# 2. Verificamos que no haya duplicados en la lista
if len(features_modelo) != len(set(features_modelo)):
    print(f"\n❌ ¡HAY DUPLICADOS EN LA LISTA DE FEATURES!")
    from collections import Counter
    counts = Counter(features_modelo)
    duplicates = [item for item, count in counts.items() if count > 1]
    print(f"   Duplicados: {duplicates}")
else:
    print(f"\n✅ No hay duplicados en la lista de features")
