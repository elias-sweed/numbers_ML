# Fuentes y tipos de datos

Este proyecto distingue tres tipos de datos y no los mezcla en las métricas ni
en la documentación.

## 1. Datos originales: MNIST

- **Qué es:** dataset público de dígitos manuscritos en escala de grises, 28×28.
- **Origen:** Yann LeCun, Corinna Cortes y Christopher Burges.
- **Descarga utilizada:** `keras.datasets.mnist.load_data()`, que obtiene el
  archivo desde:
  `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`
- **Caché local de Keras:** `~/.keras/datasets/mnist.npz`
- **Referencia oficial:** <http://yann.lecun.com/exdb/mnist/>
- **Uso en el proyecto:** entrenamiento, validación y evaluación final.
- **División:** 54 000 entrenamiento, 6 000 validación y 10 000 test.
- **Naturaleza:** son imágenes escaneadas y normalizadas por los creadores del
  dataset. No son fotografías tomadas para este proyecto.

## 2. Muestras sintéticas

- **Qué son:** composiciones generadas por `src/demo.py` a partir de imágenes
  del test de MNIST.
- **Transformaciones:** rotación, ruido gaussiano, desenfoque y composición
  sobre fondo blanco.
- **Ubicación:** `data/samples/`.
- **Uso:** probar el pipeline completo de forma reproducible.
- **Importante:** **no son fotografías reales** y no se presentan como tales.
  No se usan para calcular las métricas de test ni para afirmar una validación
  con fotos de papel.

## 3. Fotografías reales

- **Qué son:** fotos tomadas por el usuario a números escritos a mano en papel.
- **Ubicación:** `data/real_photos/`.
- **Manifiesto:** `data/real_photos/manifest.csv`.
- **Requisito:** cada fila debe tener `tipo=foto_real`.
- **Comando:** `python main.py real-test`.
- **Estado actual:** no se dispone de fotografías reales en el repositorio. La
  infraestructura de prueba está creada y documentada, pero la validación con
  fotos reales queda pendiente hasta que el usuario añada sus propias fotos.
- **Alternativa:** subir la foto directamente en la pestaña **Reconocer** de la
  aplicación Streamlit.

## Regla de honestidad

Las métricas de `results/metrics.csv` corresponden exclusivamente al test
oficial de MNIST. Las imágenes sintéticas se reportan aparte en
`results/demo_predictions.json`. Las fotografías reales se reportan aparte en
`results/real_photos_results.json` y solo si existen.
