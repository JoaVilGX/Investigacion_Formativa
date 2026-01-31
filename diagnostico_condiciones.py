import pandas as pd
import numpy as np

# Cargamos datos
df = pd.read_csv('data/car_price_cleaned.csv')
print("Valores únicos en Condition:", df['Condition'].unique())
print("\nConteo:")
print(df['Condition'].value_counts())
