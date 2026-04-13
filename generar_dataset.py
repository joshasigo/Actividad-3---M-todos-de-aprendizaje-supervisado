"""
=============================================================
  GENERADOR DE DATASET - TRANSMILENIO BOGOTÁ
  Aprendizaje Supervisado: Predicción de Tiempo de Viaje
=============================================================

Curso: Inteligencia Artificial
Actividad 4 - Métodos Supervisados (Árbol de Decisión)

Este script genera un dataset sintético realista basado en:
  - La red de conexiones definida en la Actividad 3 (A*)
  - Variables contextuales que afectan el tiempo de viaje
  - Ruido aleatorio controlado para simular variabilidad real
"""

import pandas as pd
import numpy as np
import math
import random
import os

random.seed(42)
np.random.seed(42)

# ─── RED DE TRANSMILENIO (tomada de la actividad anterior) ───────────

ESTACIONES = {
    "Portal Norte":       (4.7602, -74.0457),
    "Toberin":            (4.7440, -74.0480),
    "Cardio Infantil":    (4.7280, -74.0500),
    "Mazuren":            (4.7170, -74.0510),
    "Alcalá":             (4.7080, -74.0520),
    "Pepe Sierra":        (4.6990, -74.0530),
    "Calle 100":          (4.6880, -74.0540),
    "Calle 72":           (4.6670, -74.0520),
    "Calle 63":           (4.6570, -74.0510),
    "Flores":             (4.6480, -74.0500),
    "Calle 45":           (4.6380, -74.0490),
    "Marly":              (4.6290, -74.0480),
    "Calle 26":           (4.6180, -74.0840),
    "Ricaurte":           (4.6100, -74.0960),
    "Calle 8":            (4.6010, -74.0960),
    "General Santander":  (4.5920, -74.0960),
    "Sevillana":          (4.5830, -74.0960),
    "Portal Sur":         (4.5680, -74.0970),
    "Portal 80":          (4.6930, -74.1140),
    "Avenida Rojas":      (4.6820, -74.1130),
    "Gratamira":          (4.6720, -74.1120),
    "Portal Américas":    (4.6280, -74.1770),
    "Banderas":           (4.6320, -74.1600),
    "Patio Bonito":       (4.6360, -74.1440),
    "Tintal":             (4.6390, -74.1310),
    "Alquería":           (4.6290, -74.1070),
    "Marsella":           (4.6290, -74.1020),
    "Museo del Oro":      (4.6010, -74.0720),
    "Las Aguas":          (4.6040, -74.0670),
    "Universidades":      (4.6100, -74.0800),
    "El Tiempo Maloka":   (4.6570, -74.1130),
    "El Dorado":          (4.6620, -74.1070),
    "Av. Eldorado":       (4.6580, -74.1000),
    "Portal Usme":        (4.5140, -74.1090),
    "Molinos":            (4.5370, -74.1020),
    "Biblioteca":         (4.5540, -74.0990),
    "Parque El Tunal":    (4.5640, -74.1000),
    "Venecia":            (4.5750, -74.1000),
    "Niza Calle 127":     (4.7060, -74.0700),
    "Calle 76":           (4.6690, -74.0700),
    "Escuela Militar":    (4.6570, -74.0490),
}

LINEAS = ["B11", "C12", "H19", "G14", "P14", "TRANSV"]

# Factor de congestión por hora (hora -> multiplicador)
CONGESTION_HORA = {
    0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
    5: 1.1, 6: 1.5, 7: 2.0, 8: 2.2, 9: 1.8,  # hora pico mañana
    10: 1.3, 11: 1.2, 12: 1.4, 13: 1.3, 14: 1.2,
    15: 1.3, 16: 1.6, 17: 2.1, 18: 2.3, 19: 1.9,  # hora pico tarde
    20: 1.4, 21: 1.2, 22: 1.1, 23: 1.0,
}


def haversine_km(est1, est2):
    lat1, lon1 = ESTACIONES[est1]
    lat2, lon2 = ESTACIONES[est2]
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def es_hora_pico(hora):
    return 1 if (6 <= hora <= 9) or (16 <= hora <= 19) else 0


def es_fin_de_semana(dia_semana):
    return 1 if dia_semana >= 5 else 0


def generar_muestra(idx):
    estaciones_lista = list(ESTACIONES.keys())
    origen = random.choice(estaciones_lista)
    destino = random.choice([e for e in estaciones_lista if e != origen])

    num_paradas = random.randint(1, 15)
    linea = random.choice(LINEAS)
    hora = random.randint(5, 23)
    dia_semana = random.randint(0, 6)
    trasbordos = random.randint(0, 3)

    dist_km = haversine_km(origen, destino)
    hora_pico = es_hora_pico(hora)
    fin_semana = es_fin_de_semana(dia_semana)
    congestion = CONGESTION_HORA[hora]

    # Tiempo base: distancia / velocidad base (30 km/h) en minutos
    tiempo_base = (dist_km / 30) * 60

    # Ajustes por paradas (cada parada ~2.5 min en promedio)
    tiempo_paradas = num_paradas * 2.5

    # Ajuste por congestión
    tiempo_congestion = tiempo_base * (congestion - 1.0)

    # Penalización por trasbordo (cada trasbordo ~4 min)
    tiempo_trasbordo = trasbordos * 4.0

    # Reducción en fin de semana (~15% menos tráfico)
    reduccion_fds = -tiempo_base * 0.15 if fin_semana else 0

    # Ruido aleatorio (variabilidad real: ±20%)
    ruido = np.random.normal(0, tiempo_base * 0.1)

    tiempo_total = max(3.0, tiempo_base + tiempo_paradas +
                       tiempo_congestion + tiempo_trasbordo +
                       reduccion_fds + ruido)

    return {
        "id":              idx,
        "estacion_origen": origen,
        "estacion_destino": destino,
        "distancia_km":    round(dist_km, 3),
        "num_paradas":     num_paradas,
        "trasbordos":      trasbordos,
        "linea":           linea,
        "hora_salida":     hora,
        "dia_semana":      dia_semana,
        "hora_pico":       hora_pico,
        "fin_de_semana":   fin_semana,
        "factor_congestion": round(congestion, 2),
        "tiempo_viaje_min": round(tiempo_total, 2),   # ← VARIABLE OBJETIVO
    }


def generar_dataset(n=500):
    print(f"Generando dataset con {n} muestras...")
    muestras = [generar_muestra(i) for i in range(n)]
    df = pd.DataFrame(muestras)
    return df


if __name__ == "__main__":
    df = generar_dataset(500)
    os.makedirs("datos", exist_ok=True)
    df.to_csv("datos/dataset_transmilenio.csv", index=False)
    print(f"\n✅ Dataset guardado en: datos/dataset_transmilenio.csv")
    print(f"   Filas: {len(df)}  |  Columnas: {len(df.columns)}")
    print(f"\nPrimeras filas:")
    print(df.head(3).to_string())
    print(f"\nEstadísticas de la variable objetivo (tiempo_viaje_min):")
    print(df["tiempo_viaje_min"].describe().round(2))
