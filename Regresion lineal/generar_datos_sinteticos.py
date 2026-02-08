import pandas as pd
import numpy as np

def generar_datos_sinteticos(n_samples=10000, p_ruido=0.15):
    areas = ['Desarrollo', 'Diseño', 'Marketing', 'Ventas', 'Soporte', 'QA']
    data = {
        'nombre': [f'Empleado_{i}' for i in range(n_samples)],
        'area': np.random.choice(areas, n_samples),
        'jerarquia': np.random.choice(['trainee', 'junior', 'senior'], n_samples, p=[0.3, 0.4, 0.3]),
        'desempenio': np.random.choice(['bajo', 'medio', 'alto'], n_samples, p=[0.2, 0.5, 0.3]),
        'años_experiencia': np.random.normal(5, 2, n_samples).clip(0, 15),
        'proyectos_completados': np.random.poisson(5, n_samples),
        'horas_extra': np.random.normal(10, 5, n_samples).clip(0, 40),
        'capacitaciones': np.random.poisson(3, n_samples),
        'feedback_positivo': np.random.normal(4, 1, n_samples).clip(1, 5),
        'feedback_negativo': np.random.normal(2, 1, n_samples).clip(1, 5),
        'tiempo_en_empresa': np.random.normal(3, 1.5, n_samples).clip(0, 10),
        'motivacion': np.random.normal(3, 1, n_samples).clip(1, 5)
    }
    df = pd.DataFrame(data)

    def calcular_desempenio_futuro(row):
        score = 0
        if row['jerarquia'] == 'senior': score += 1.5
        elif row['jerarquia'] == 'junior': score += 1.0
        else: score += 0.5
        if row['desempenio'] == 'alto': score += 1.5
        elif row['desempenio'] == 'medio': score += 1.0
        else: score += 0.5
        score += min(row['años_experiencia'] / 10, 1.0)
        score += min(row['proyectos_completados'] / 10, 1.0)
        score += min(row['horas_extra'] / 40, 0.5)
        score += min(row['capacitaciones'] / 5, 0.5)
        score += (row['feedback_positivo'] - row['feedback_negativo']) / 5
        score += min(row['tiempo_en_empresa'] / 10, 0.5)
        score += (row['motivacion'] - 3) * 0.5
        score += np.random.normal(0, 0.2)
        if np.random.rand() < p_ruido:
            score += np.random.uniform(-2, 2)
        if score >= 4.5: return 2
        elif score >= 3.0: return 1
        else: return 0

    df['desempenio_futuro'] = df.apply(calcular_desempenio_futuro, axis=1)
    df['jerarquia'] = df['jerarquia'].map({'trainee': 0, 'junior': 1, 'senior': 2})
    df['desempenio'] = df['desempenio'].map({'bajo': 0, 'medio': 1, 'alto': 2})
    return df

if __name__ == "__main__":
    df = generar_datos_sinteticos()
    df.to_csv('prediccion_rendimiento_training_completo.csv', index=False)
    print("Datos sintéticos generados y guardados en 'prediccion_rendimiento_training_completo.csv'") 