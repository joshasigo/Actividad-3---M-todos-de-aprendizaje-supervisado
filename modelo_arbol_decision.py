"""
=============================================================
  MODELO DE APRENDIZAJE SUPERVISADO - TRANSMILENIO BOGOTÁ
  Árbol de Decisión para Regresión del Tiempo de Viaje
=============================================================

Curso: Inteligencia Artificial
Actividad 4 - Métodos Supervisados

Referencia:
  Palma Méndez, J. T. (2008). Inteligencia artificial: métodos,
  técnicas y aplicaciones. Madrid: McGraw-Hill España.
  Capítulo 17: Aprendizaje de árboles y reglas de decisión.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
import os
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN ───────────────────────────────────────────
RUTA_DATASET   = "datos/dataset_transmilenio.csv"
RUTA_FIGURAS   = "resultados"
SEMILLA        = 42
os.makedirs(RUTA_FIGURAS, exist_ok=True)

# ─── 1. CARGA Y EXPLORACIÓN ──────────────────────────────────

def cargar_datos():
    print("\n" + "="*60)
    print("  1. CARGA Y EXPLORACIÓN DEL DATASET")
    print("="*60)

    df = pd.read_csv(RUTA_DATASET)
    print(f"\n  Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"\n  Tipos de variables:")
    for col in df.columns:
        print(f"    {col:<25} {df[col].dtype}")

    print(f"\n  Valores nulos por columna:")
    nulos = df.isnull().sum()
    print(f"    {nulos[nulos > 0].to_string() if nulos.sum() > 0 else '    Ninguno ✅'}")

    print(f"\n  Variable objetivo (tiempo_viaje_min):")
    desc = df["tiempo_viaje_min"].describe()
    print(f"    Media:   {desc['mean']:.2f} min")
    print(f"    Mediana: {df['tiempo_viaje_min'].median():.2f} min")
    print(f"    Mín:     {desc['min']:.2f} min")
    print(f"    Máx:     {desc['max']:.2f} min")
    print(f"    Desv. estándar: {desc['std']:.2f} min")

    return df


# ─── 2. PREPROCESAMIENTO ─────────────────────────────────────

def preprocesar(df):
    print("\n" + "="*60)
    print("  2. PREPROCESAMIENTO")
    print("="*60)

    df = df.copy()

    # Codificar variables categóricas con LabelEncoder
    le_origen  = LabelEncoder()
    le_destino = LabelEncoder()
    le_linea   = LabelEncoder()

    df["origen_enc"]  = le_origen.fit_transform(df["estacion_origen"])
    df["destino_enc"] = le_destino.fit_transform(df["estacion_destino"])
    df["linea_enc"]   = le_linea.fit_transform(df["linea"])

    # Variables de entrada (features)
    FEATURES = [
        "distancia_km",      # distancia geográfica entre estaciones
        "num_paradas",       # número de paradas en la ruta
        "trasbordos",        # cantidad de transbordos necesarios
        "hora_salida",       # hora del día (0-23)
        "dia_semana",        # día (0=lunes, 6=domingo)
        "hora_pico",         # binaria: 1=hora pico, 0=no
        "fin_de_semana",     # binaria: 1=fin de semana, 0=no
        "factor_congestion", # multiplicador de congestión por hora
        "origen_enc",        # estación origen codificada
        "destino_enc",       # estación destino codificada
        "linea_enc",         # línea de transporte codificada
    ]

    TARGET = "tiempo_viaje_min"

    X = df[FEATURES]
    y = df[TARGET]

    print(f"\n  Features seleccionadas ({len(FEATURES)}):")
    for f in FEATURES:
        print(f"    • {f}")
    print(f"\n  Variable objetivo: {TARGET}")

    return X, y, FEATURES, (le_origen, le_destino, le_linea)


# ─── 3. DIVISIÓN TRAIN / TEST ────────────────────────────────

def dividir_datos(X, y):
    print("\n" + "="*60)
    print("  3. DIVISIÓN TRAIN / TEST (80% / 20%)")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA
    )
    print(f"\n  Entrenamiento: {len(X_train)} muestras")
    print(f"  Prueba:        {len(X_test)} muestras")
    return X_train, X_test, y_train, y_test


# ─── 4. ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN ──────────────────

def entrenar_modelo(X_train, y_train, features):
    print("\n" + "="*60)
    print("  4. ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN")
    print("="*60)

    # Buscar la profundidad óptima con validación cruzada
    print("\n  Buscando profundidad óptima (3 a 12)...")
    profundidades = range(3, 13)
    scores_cv = []

    for depth in profundidades:
        modelo_temp = DecisionTreeRegressor(
            max_depth=depth, random_state=SEMILLA
        )
        cv_scores = cross_val_score(
            modelo_temp, X_train, y_train,
            cv=5, scoring="r2"
        )
        scores_cv.append(cv_scores.mean())
        print(f"    Profundidad {depth:2d} → R² promedio CV: {cv_scores.mean():.4f}")

    mejor_depth = list(profundidades)[np.argmax(scores_cv)]
    print(f"\n  ✅ Mejor profundidad: {mejor_depth} (R²={max(scores_cv):.4f})")

    # Entrenar modelo final
    modelo = DecisionTreeRegressor(
        max_depth=mejor_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=SEMILLA
    )
    modelo.fit(X_train, y_train)

    print(f"\n  Parámetros del modelo:")
    print(f"    max_depth:         {mejor_depth}")
    print(f"    min_samples_split: 10")
    print(f"    min_samples_leaf:  5")
    print(f"    Nodos del árbol:   {modelo.tree_.node_count}")
    print(f"    Hojas del árbol:   {modelo.tree_.n_leaves}")

    # Guardar gráfica de búsqueda de profundidad
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(profundidades), scores_cv, "o-", color="#1B4FBE", linewidth=2)
    ax.axvline(mejor_depth, color="#F5A623", linestyle="--", label=f"Óptimo: {mejor_depth}")
    ax.set_xlabel("Profundidad del árbol", fontsize=12)
    ax.set_ylabel("R² (validación cruzada)", fontsize=12)
    ax.set_title("Selección de profundidad óptima", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RUTA_FIGURAS}/01_profundidad_optima.png", dpi=150)
    plt.close()
    print(f"\n  📊 Gráfica guardada: {RUTA_FIGURAS}/01_profundidad_optima.png")

    return modelo, mejor_depth


# ─── 5. EVALUACIÓN ───────────────────────────────────────────

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, features):
    print("\n" + "="*60)
    print("  5. EVALUACIÓN DEL MODELO")
    print("="*60)

    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)

    metricas = {
        "Conjunto":  ["Entrenamiento", "Prueba"],
        "MAE (min)": [mean_absolute_error(y_train, y_pred_train),
                      mean_absolute_error(y_test, y_pred_test)],
        "RMSE (min)":[np.sqrt(mean_squared_error(y_train, y_pred_train)),
                      np.sqrt(mean_squared_error(y_test, y_pred_test))],
        "R²":        [r2_score(y_train, y_pred_train),
                      r2_score(y_test, y_pred_test)],
    }

    print("\n  Métricas de desempeño:")
    print(f"  {'Conjunto':<16} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print("  " + "-"*48)
    for i in range(2):
        print(f"  {metricas['Conjunto'][i]:<16} "
              f"{metricas['MAE (min)'][i]:>10.3f} "
              f"{metricas['RMSE (min)'][i]:>10.3f} "
              f"{metricas['R²'][i]:>8.4f}")

    # ── Gráfica 1: Predicho vs Real ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (y_true, y_pred, titulo, color) in zip(axes, [
        (y_train, y_pred_train, "Entrenamiento", "#1B4FBE"),
        (y_test,  y_pred_test,  "Prueba",        "#E53E3E"),
    ]):
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=color)
        lim = [min(y_true.min(), y_pred.min()) - 2,
               max(y_true.max(), y_pred.max()) + 2]
        ax.plot(lim, lim, "k--", linewidth=1, label="Predicción perfecta")
        ax.set_xlabel("Tiempo real (min)", fontsize=11)
        ax.set_ylabel("Tiempo predicho (min)", fontsize=11)
        ax.set_title(f"Predicho vs Real — {titulo}\nR²={r2_score(y_true, y_pred):.4f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RUTA_FIGURAS}/02_predicho_vs_real.png", dpi=150)
    plt.close()
    print(f"\n  📊 Gráfica guardada: {RUTA_FIGURAS}/02_predicho_vs_real.png")

    # ── Gráfica 2: Distribución de errores ──
    errores = y_test - y_pred_test
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errores, bins=30, color="#1B4FBE", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="#F5A623", linewidth=2, linestyle="--", label="Error = 0")
    ax.axvline(errores.mean(), color="#E53E3E", linewidth=2, linestyle="-",
               label=f"Media error = {errores.mean():.2f} min")
    ax.set_xlabel("Error de predicción (min)", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.set_title("Distribución del error de predicción (conjunto de prueba)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RUTA_FIGURAS}/03_distribucion_errores.png", dpi=150)
    plt.close()
    print(f"  📊 Gráfica guardada: {RUTA_FIGURAS}/03_distribucion_errores.png")

    # ── Gráfica 3: Importancia de features ──
    importancias = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1B4FBE" if imp > 0.1 else "#5B9BF5" for imp in importancias]
    importancias.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Importancia relativa", fontsize=11)
    ax.set_title("Importancia de variables — Árbol de Decisión", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(f"{RUTA_FIGURAS}/04_importancia_variables.png", dpi=150)
    plt.close()
    print(f"  📊 Gráfica guardada: {RUTA_FIGURAS}/04_importancia_variables.png")

    return y_pred_test, metricas


# ─── 6. REGLAS EXTRAÍDAS DEL ÁRBOL ───────────────────────────

def mostrar_reglas(modelo, features):
    print("\n" + "="*60)
    print("  6. REGLAS EXTRAÍDAS DEL ÁRBOL (primeros 3 niveles)")
    print("="*60)
    reglas = export_text(modelo, feature_names=features, max_depth=3)
    print(reglas)


# ─── 7. PREDICCIÓN DE NUEVOS CASOS ───────────────────────────

def predecir_casos(modelo, encoders):
    print("\n" + "="*60)
    print("  7. PREDICCIÓN DE NUEVOS CASOS")
    print("="*60)

    le_origen, le_destino, le_linea = encoders

    casos = [
        {
            "descripcion":     "Portal Norte → Portal Sur, hora pico mañana",
            "distancia_km":    24.5,
            "num_paradas":     17,
            "trasbordos":      1,
            "hora_salida":     7,
            "dia_semana":      1,   # martes
            "hora_pico":       1,
            "fin_de_semana":   0,
            "factor_congestion": 2.0,
            "origen_enc":      le_origen.transform(["Portal Norte"])[0],
            "destino_enc":     le_destino.transform(["Portal Sur"])[0],
            "linea_enc":       le_linea.transform(["B11"])[0],
        },
        {
            "descripcion":     "Portal Américas → Calle 26, domingo medio día",
            "distancia_km":    12.3,
            "num_paradas":     8,
            "trasbordos":      2,
            "hora_salida":     12,
            "dia_semana":      6,   # domingo
            "hora_pico":       0,
            "fin_de_semana":   1,
            "factor_congestion": 1.4,
            "origen_enc":      le_origen.transform(["Portal Américas"])[0],
            "destino_enc":     le_destino.transform(["Calle 26"])[0],
            "linea_enc":       le_linea.transform(["G14"])[0],
        },
        {
            "descripcion":     "Museo del Oro → Calle 100, noche entre semana",
            "distancia_km":    9.8,
            "num_paradas":     6,
            "trasbordos":      0,
            "hora_salida":     21,
            "dia_semana":      3,   # jueves
            "hora_pico":       0,
            "fin_de_semana":   0,
            "factor_congestion": 1.2,
            "origen_enc":      le_origen.transform(["Museo del Oro"])[0],
            "destino_enc":     le_destino.transform(["Calle 100"])[0],
            "linea_enc":       le_linea.transform(["TRANSV"])[0],
        },
    ]

    features = [
        "distancia_km", "num_paradas", "trasbordos", "hora_salida",
        "dia_semana", "hora_pico", "fin_de_semana", "factor_congestion",
        "origen_enc", "destino_enc", "linea_enc",
    ]

    print()
    for caso in casos:
        desc = caso.pop("descripcion")
        X_nuevo = pd.DataFrame([caso])[features]
        tiempo_pred = modelo.predict(X_nuevo)[0]
        print(f"  📍 {desc}")
        print(f"     Tiempo predicho: {tiempo_pred:.1f} minutos")
        print()
        caso["descripcion"] = desc  # restaurar


# ─── MAIN ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🚌 APRENDIZAJE SUPERVISADO — TRANSMILENIO BOGOTÁ")
    print("     Árbol de Decisión · Regresión de Tiempo de Viaje")
    print("="*60)

    # Ejecutar pipeline completo
    df              = cargar_datos()
    X, y, features, encoders = preprocesar(df)
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    modelo, depth   = entrenar_modelo(X_train, y_train, features)
    y_pred, metricas = evaluar_modelo(modelo, X_train, X_test,
                                      y_train, y_test, features)
    mostrar_reglas(modelo, features)
    predecir_casos(modelo, encoders)

    print("\n" + "="*60)
    print("  ✅ Pipeline completado exitosamente.")
    print(f"  📁 Figuras generadas en: /{RUTA_FIGURAS}/")
    print("="*60 + "\n")
