# TP2 - Predicción de fumadores

Trabajo práctico 2 del Módulo 2 de la diplomatura en Inteligencia Artificial.

## Descripción y objetivo

El objetivo del proyecto es entrenar un modelo de clasificación para predecir si una persona es fumadora (`smoking = 1`) o no fumadora (`smoking = 0`) a partir de variables de características personales y mediciones de salud.

Se trabaja con un conjunto de datos etiquetados para entrenar y evaluar distintos modelos y con un segundo conjunto de datos sin la columna `smoking`, sobre el que se generan las predicciones finales.

La **métrica principal** es el **F1-Score de la clase 1 (fumador)**. La evaluación definitiva de la consigna se realiza posteriormente sobre las predicciones del segundo conjunto de datos.

## Estructura del proyecto

```text
.
├── data/
│   ├── raw/
│   │   ├── TP2_Modulo_2.ipynb
│   │   ├── smoking_prediction.csv
|   |   ├── smoking_prediction.xlsx
|   |   ├── smoking_prediction_entrega.csv
│   │   └── smoking_prediction_entrega.xlsx
│   ├── processed/
│   │   ├── X_train.joblib
│   │   ├── X_test.joblib
│   │   ├── y_train.joblib
│   │   ├── y_test.joblib
│   │   └── predicciones.csv
│   └── external/
├── models/
│   ├── encoder.joblib
│   ├── scaler.joblib
│   ├── columnas_categoricas.joblib
│   ├── columnas_numericas.joblib
│   └── random_forest_final.joblib
├── notebooks/
│   ├── 01_lectura_y_discovery.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_entrenamiento_y_optimizacion.ipynb
│   ├── 05_validacion.ipynb
│   └── 06_prediccion.ipynb
├── requirements.txt
└── README.md
```

Los archivos `.joblib` y `predicciones.csv` son generados durante la ejecución de los notebooks. La carpeta `data/external/` se reserva para datos externos, si se utilizaran.

## Preparación del entorno

Se recomienda usar Python 3 y un entorno virtual. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

En Windows, la activación del entorno virtual se realiza con `.venv\\Scripts\\activate` (PowerShell: `.venv\\Scripts\\Activate.ps1`).

> **Importante:** los notebooks utilizan rutas relativas como `../data/raw/` y `../models/`. Deben ejecutarse con `notebooks/` como directorio de trabajo, por ejemplo abriéndolos desde JupyterLab en esa carpeta.

## Datos

Colocar los archivos de entrada en `data/raw/`:

- `smoking_prediction.csv`: conjunto etiquetado para entrenamiento y evaluación.
- `smoking_prediction_entrega.csv`: conjunto sin etiquetas para generar las predicciones finales.

La columna `ID` identifica cada registro. La columna `smoking` es la variable objetivo del conjunto etiquetado.

## Ejecución de los notebooks

Ejecutar en este orden, usando **Run All** en cada notebook:

1. **`01_lectura_y_discovery.ipynb`**: lectura, inspección inicial y comprobaciones básicas de los datos.
2. **`02_eda.ipynb`**: análisis exploratorio y visualizaciones de las variables.
3. **`03_preprocesamiento.ipynb`**: separación entre entrenamiento (70 %) y prueba (30 %), preparación de variables y guardado de datos y transformadores.
4. **`04_entrenamiento_y_optimizacion.ipynb`**: entrenamiento de Decision Tree, Random Forest y XGBoost, comparación de resultados y búsqueda de hiperparámetros.
5. **`05_validacion.ipynb`**: selección del Random Forest controlado, evaluación en Train y Test, guardado del modelo en `models/random_forest_final.joblib` y comprobación de su carga.
6. **`06_prediccion.ipynb`**: carga de los datos sin etiquetas, aplicación de los transformadores ya entrenados, generación de predicciones y exportación del CSV.

El preprocesamiento codifica las variables categóricas `gender` y `tartar` con `OneHotEncoder` y aplica `MinMaxScaler` a las variables numéricas. Se excluyen `ID` y `oral` de las variables predictoras; `smoking` se usa únicamente como objetivo durante el entrenamiento. Se mantienen **26 variables predictoras** después de transformar los datos.

Para predecir sobre datos nuevos se reutilizan `models/encoder.joblib` y `models/scaler.joblib` mediante `transform()`, sin volver a ajustar estos objetos.

## Experimentos y resultados

Se probaron diferentes modelos y configuraciones. El F1-Score se refiere siempre a la clase fumador (`smoking = 1`). Los valores de la tabla están redondeados a dos decimales.

| Modelo | F1 Train | F1 Test |
|---|---:|---:|
| Decision Tree inicial | 0,66 | 0,65 |
| Decision Tree optimizado | 0,69 | 0,68 |
| Random Forest inicial | 1,00 | 0,72 |
| Random Forest controlado | 0,75 | 0,70 |
| Random Forest optimizado | 1,00 | 0,73 |
| XGBoost inicial | 0,74 | 0,70 |
| XGBoost optimizado | 0,80 | 0,70 |

### Cambios realizados durante los experimentos

- **Decision Tree:** primero se entrenó un árbol con profundidad máxima 3. Luego se utilizaron combinaciones de `max_depth`, `min_samples_split` y `min_samples_leaf` mediante `GridSearchCV`. La versión optimizada mejoró el F1 en Test hasta 0,68.
- **Random Forest:** el modelo inicial obtuvo F1 de 1,00 en Train y 0,72 en Test, mostrando sobreajuste. Se probó una versión controlada, con restricciones en la profundidad y el tamaño de las divisiones y hojas, que obtuvo F1 de 0,75 en Train y 0,70 en Test. También se aplicó `RandomizedSearchCV`; la mejor configuración de esa búsqueda obtuvo F1 de 0,73 en Test, pero mantuvo una diferencia importante respecto a Train.
- **XGBoost:** se probó inicialmente la configuración usada en clase. Luego se aplicó `RandomizedSearchCV` para ajustar sus hiperparámetros. El F1 en Test se mantuvo en 0,70, mientras que la diferencia con Train aumentó.

En las búsquedas de hiperparámetros se utilizó `scoring="f1"` para mantener el criterio de evaluación indicado en la consigna.

## Modelo seleccionado

Se seleccionó **Random Forest controlado**, con la siguiente configuración:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
```

El modelo obtuvo **F1 de 0,75 en Train y 0,70 en Test**, con **accuracy de 0,76 en Test**, **precision de 0,66** y **recall de 0,74** para la clase fumador.

Aunque el Random Forest optimizado logró un F1 superior en Test (0,73), se eligió el modelo controlado porque presentó menos sobreajuste y un rendimiento aceptable. Esta elección no significa que el modelo controlado tenga necesariamente mejor rendimiento sobre cualquier conjunto de datos nuevo.

## Predicciones finales

El notebook `06_prediccion.ipynb` procesa **5.692 registros sin etiquetar** y genera:

```text
data/processed/predicciones.csv
```

El archivo contiene dos columnas:

- **`ID`**: identificador del registro.
- **`smoking_prediction`**: predicción, con `0` para no fumador y `1` para fumador.

En la ejecución realizada, el modelo clasificó **3.328 registros como no fumadores** y **2.364 como fumadores**. Estas cifras son predicciones del modelo, no etiquetas reales.

## Conclusiones

Se realizó el proceso completo de lectura, exploración, preparación de datos, entrenamiento, comparación de modelos y predicción sobre un conjunto nuevo.

Se observó que los modelos con mejor F1 de entrenamiento no necesariamente obtuvieron el mejor resultado en Test. El Random Forest controlado se seleccionó porque presentó resultados más parecidos entre Train y Test que las variantes con mayor sobreajuste.

El F1-Score obtenido sobre Test sirve como referencia. El resultado definitivo se conocerá cuando se evalúen las predicciones sobre el segundo conjunto de datos, según la consigna del trabajo práctico.
