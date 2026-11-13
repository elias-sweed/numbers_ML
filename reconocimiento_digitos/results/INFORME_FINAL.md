# Informe final del proyecto

**Sistema Inteligente de Reconocimiento y Clasificación de Dígitos Manuscritos**

Generado: 2026-11-11

> Este informe resume el estado real del proyecto después de las mejoras
> finales. Todas las cifras provienen de ejecuciones reales registradas en
> `results/`. No se inventó ningún resultado ni se modificaron las métricas
> anteriores de test.

---

## 1. Resumen

El proyecto implementa un sistema completo de reconocimiento de dígitos
manuscritos (0–9) con el dataset público MNIST, seis modelos, una aplicación
Streamlit y una batería de pruebas automatizadas.

Mejoras incorporadas en esta revisión:

1. **Nueve experimentos reales** de hiperparámetros de la CNN (learning rate,
   filtros, dropout y batch size), seleccionados exclusivamente con validación.
2. **Infraestructura de prueba con fotografías reales** de papel, con manifiesto
   y rechazo explícito de imágenes sintéticas.
3. **Aclaración de fuentes y tipos de datos**: MNIST original, muestras
   sintéticas y fotografías reales.
4. **Unificación documental** en `DOCUMENTACION.md` (sin `.env`).
5. **Interfaz simplificada de una sola pantalla**, con guía del tipo de imagen
   admitida, explicación detallada del resultado en lenguaje sencillo e
   historial persistente de las últimas 100 pruebas (sin guardar imágenes).
6. **Verificación** con 49 pruebas, AppTest y revisión de artefactos.
7. **Informe final** (este archivo) con resultados, limitaciones y cumplimiento.

Los modelos y las métricas originales de test **no se sobrescribieron**.

---

## 2. Datos: origen y tipos

MNIST es un **dataset público descargado de internet**, no una colección de
fotografías tomadas para este proyecto.

- **Autores:** Yann LeCun, Corinna Cortes y Christopher Burges.
- **Descarga usada:** `keras.datasets.mnist.load_data()`.
- **URL del archivo:** `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`
- **Referencia oficial:** <http://yann.lecun.com/exdb/mnist/>
- **División:** 54 000 entrenamiento / 6 000 validación / 10 000 test.

| Tipo | Ubicación | Origen | Uso |
|---|---|---|---|
| MNIST original | `data/mnist_split.npz` | Dataset público | Entrenamiento, validación y test |
| Muestras sintéticas | `data/samples/` | MNIST + rotación/ruido/desenfoque | Demo del pipeline |
| Fotografías reales | `data/real_photos/` | Fotos del usuario | Prueba con papel real |

Las muestras sintéticas **no son fotografías reales** y no se usan para
calcular métricas de test. Detalle completo en
`data/FUENTES_Y_TIPOS_DE_DATOS.md`.

---

## 3. Resultados de test (sin cambios)

Evaluación original sobre las 10 000 imágenes del test oficial de MNIST.
No se volvió a ejecutar `evaluate` porque las mejoras no reentrenaron los
modelos de producción.

| Modelo | Accuracy | Precision | Recall | F1 macro |
|---|---:|---:|---:|---:|
| CNN | **99.61 %** | 99.61 % | 99.60 % | 99.61 % |
| MLP | 98.44 % | 98.44 % | 98.43 % | 98.43 % |
| KNN | 97.58 % | 97.58 % | 97.56 % | 97.57 % |
| SVM | 97.07 % | 97.06 % | 97.04 % | 97.05 % |
| Regresión Logística | 92.15 % | 92.04 % | 92.03 % | 92.02 % |

K-Means (no supervisado): ARI 0.3657, NMI 0.4979, silueta 0.0873,
pureza 59.31 % tras mapeo a posteriori.

Fuente: `results/metrics.csv`, `results/RESUMEN.md`.

---

## 4. Validación

- **Modelos tradicionales:** validación cruzada estratificada de 3 folds sobre
  6 000 imágenes de entrenamiento.

  | Modelo | Accuracy media | Desviación |
  |---|---:|---:|
  | SVM | 95.20 % | 0.0029 |
  | KNN | 94.37 % | 0.0056 |
  | Regresión Logística | 89.50 % | 0.0008 |

- **MLP y CNN:** conjunto de validación de 6 000 imágenes para
  `EarlyStopping`, `ReduceLROnPlateau` y selección de la mejor época.
- **Ajuste de hiperparámetros de la CNN:** experimentos comparativos sobre
  entrenamiento y evaluación con validación. **El test no se usó para elegir
  hiperparámetros.**
- **Test:** reservado para la evaluación final de los modelos ya entrenados.

Fuente: `results/cross_validation.csv` y `results/tuning/`.

---

## 5. Ajuste de hiperparámetros de la CNN

### 5.1. Protocolo real

- Subconjunto estratificado de **20 000** imágenes de entrenamiento.
- Validación completa de **6 000** imágenes.
- **5 épocas** por experimento (viabilidad en CPU).
- Semilla 42 y `set_random_seed` antes de cada experimento.
- Un factor distinto respecto a la base por experimento.
- Criterio: mayor accuracy de validación; desempate por menor `val_loss`.
- El conjunto de **test no se utilizó**.
- Ejecución: 11/11/2026, 23:38–23:49 (≈10 min 39 s en total).

Comando:

```bash
python main.py tune-cnn
```

### 5.2. Resultados de validación

| Experimento | Factor | LR | Filtros | Dropout | Batch | Mejor val accuracy | val_loss | Época |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `dropout_bajo` | dropout | 1e-3 | 32/64 | 0.15/0.35 | 128 | **98.55 %** | **0.0558** | 5 |
| `batch_64` | batch size | 1e-3 | 32/64 | 0.25/0.50 | 64 | 98.40 % | 0.0740 | 5 |
| `lr_5e-4` | learning rate | 5e-4 | 32/64 | 0.25/0.50 | 128 | 98.30 % | 0.0601 | 5 |
| `filtros_16_32` | filtros | 1e-3 | 16/32 | 0.25/0.50 | 128 | 98.18 % | 0.0639 | 5 |
| `lr_2e-3` | learning rate | 2e-3 | 32/64 | 0.25/0.50 | 128 | 98.07 % | 0.0660 | 5 |
| `dropout_alto` | dropout | 1e-3 | 32/64 | 0.35/0.60 | 128 | 97.83 % | 0.0837 | 4 |
| `filtros_48_96` | filtros | 1e-3 | 48/96 | 0.25/0.50 | 128 | 97.63 % | 0.0892 | 5 |
| `base` | base | 1e-3 | 32/64 | 0.25/0.50 | 128 | 97.60 % | 0.0869 | 5 |
| `batch_256` | batch size | 1e-3 | 32/64 | 0.25/0.50 | 256 | 95.68 % | 0.2250 | 5 |

Archivos: `results/cnn_tuning.csv`, `results/cnn_tuning.json`,
`results/TUNING_CNN.md`, `plots/cnn_tuning.png` y
`results/tuning/history_<experimento>.csv`.

### 5.3. Configuración seleccionada y justificación

**Seleccionada:** `dropout_bajo` — filtros 32/64, dropout 0.15/0.35, LR 1e-3,
batch 128.

Justificación basada en los datos:

- Obtuvo la **mayor accuracy de validación (98.55 %)** y el **menor
  `val_loss` (0.0558)** de la mejor época.
- Mejora **+0.95 puntos porcentuales** respecto a la base, con el mismo número
  de parámetros (468 010) y un tiempo similar (73.5 s frente a 59.4 s).
- Una regularización más suave converge más rápido en un presupuesto corto de
  5 épocas.
- `batch_64` queda cerca (98.40 %), pero tiene más coste por época.
- `filtros_16_32` mantiene un rendimiento alto con la mitad de parámetros
  (218 586), útil si se prioriza ligereza.
- `filtros_48_96` no converge en 5 épocas (77.9 % de accuracy de train).
- `batch_256` empeora claramente la validación y `dropout_alto` queda por
  debajo de la configuración elegida.

**Decisión:** se registra como configuración **candidata**. No se reentrenó el
modelo de producción para no sobrescribir `models/cnn.keras` ni las métricas de
test ya evaluadas. Adoptarla exigiría reentrenar con las 54 000 imágenes,
reevaluar en test y actualizar los artefactos.

### 5.4. Limitaciones del ajuste

- Es un **cribado comparativo** con 20 000 imágenes y 5 épocas, no un
  entrenamiento final.
- Las diferencias entre las configuraciones más cercanas son pequeñas y pueden
  variar con la inicialización.
- Solo se varió un factor por experimento; no se exploraron interacciones.
- TensorFlow en CPU puede variar en los últimos decimales por oneDNN.

---

## 6. Prueba con fotografías reales

**Estado: pendiente de fotografías del usuario.** No se encontraron
fotografías reales en el proyecto y no se presentan imágenes sintéticas como
tales.

Infraestructura implementada:

- Carpeta `data/real_photos/` con instrucciones en `LEEME.md`.
- Manifiesto `manifest.csv` con columnas `archivo`, `esperado`, `modelo`,
  `min_area`, `tipo`, `descripcion`.
- Comando `python main.py real-test`.
- Solo se procesan filas con `tipo=foto_real`.
- Resultados separados en `results/real_photos_results.json` y
  `results/REAL_PHOTOS.md`.
- Si no hay fotos, el informe indica honestamente el estado pendiente.
- También se puede subir una foto en la interfaz web de Streamlit.

Cómo completar la prueba:

1. Hacer una foto real de un número escrito en papel.
2. Guardarla en `data/real_photos/`.
3. Añadir una fila a `manifest.csv` con `tipo=foto_real`.
4. Ejecutar `python main.py real-test`.

---

## 7. Verificación y reproducibilidad

Ejecutado realmente:

- **49 pruebas unitarias aprobadas:**
  `..\.venv\Scripts\python.exe -m unittest discover -s tests -v`
- **AppTest:** 0 excepciones en la interfaz de una sola pantalla; los cinco
  modelos se seleccionan sin errores.
- **Navegador:** `data/samples/numero_2026.png` reconocida como "2026" con la
  CNN (4 dígitos, 100 % de confianza), con explicación detallada e historial
  guardado, sin errores de consola.
- **`python main.py real-test`:** estado `sin_fotografias`, sin resultados
  inventados.
- **Tuning:** 9/9 experimentos completados con sus 9 historiales.
- **Reproducibilidad:** semilla global 42, división estratificada fija,
  `set_random_seed` antes de cada experimento y criterios documentados.
  TensorFlow puede variar en los últimos decimales por oneDNN y paralelismo.

Los archivos `models/cnn.keras`, `results/metrics.csv` y el resto de artefactos
originales permanecen intactos.

---

## 8. Cumplimiento de los requisitos académicos

| Requisito | Estado | Evidencia |
|---|---|---|
| 1. Ajuste de hiperparámetros real | Cumplido | 9 experimentos, tabla y justificación en §5 |
| 2. Validación y selección sin test | Cumplido | Validación completa; test no usado; CV tradicional intacta |
| 3. Prueba con imágenes reales | Parcial: infraestructura lista, faltan fotos | `src/real_test.py`, `manifest.csv`, informe pendiente |
| 4. Origen y tipos de datos | Cumplido | §2 y `data/FUENTES_Y_TIPOS_DE_DATOS.md` |
| 5. Documentación unificada | Cumplido | Solo `DOCUMENTACION.md`; referencias actualizadas |
| 6. Verificación y reproducibilidad | Cumplido | 49 pruebas, AppTest, artefactos y semillas |
| 7. Informe final | Cumplido | Este archivo |

---

## 9. Limitaciones generales

- No hay fotografías reales de papel en el repositorio; esa validación queda
  pendiente.
- El ajuste de hiperparámetros es un cribado con 5 épocas y 20 000 imágenes.
- K-Means es agrupamiento no supervisado, no un clasificador.
- La confianza del SVM es una transformación softmax de sus márgenes, no una
  probabilidad calibrada.
- El rendimiento sobre fotos reales dependerá de luz, encuadre, contraste y
  separación entre dígitos.

---

## 10. Archivos principales de esta revisión

- `src/tuning.py`, `src/real_test.py`, `src/models.py`
- `main.py`, `app.py`
- `tests/test_tuning.py`, `tests/test_real_test.py`, `tests/test_models.py`
- `README.md`, `DOCUMENTACION.md`
- `data/FUENTES_Y_TIPOS_DE_DATOS.md`, `data/real_photos/`
- `results/cnn_tuning.csv`, `results/cnn_tuning.json`, `results/TUNING_CNN.md`
- `results/tuning/` (9 historiales)
- `results/REAL_PHOTOS.md`, `results/real_photos_results.json`
- `plots/cnn_tuning.png`
