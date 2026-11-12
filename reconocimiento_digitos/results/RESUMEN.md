# Resumen de resultados

Generado: 2026-11-11T15:26:18

> Todas las cifras provienen de ejecuciones reales sobre el conjunto de test oficial de MNIST (10 000 imágenes). No hay valores simulados.

## Dataset

- Nombre: MNIST (Modified National Institute of Standards and Technology)
- Entrenamiento: 54,000 imágenes
- Validación: 6,000 imágenes
- Prueba: 10,000 imágenes
- Dimensiones: 28x28, escala de grises

## Métricas de test

| Modelo | Accuracy (%) | Precision (%) | Recall (%) | F1 macro (%) | Inferencia (s) |
| --- | --- | --- | --- | --- | --- |
| Red Neuronal Convolucional (CNN) | 99.6100 | 99.6100 | 99.6000 | 99.6100 | 1.8720 |
| Red Neuronal Artificial (MLP) | 98.4400 | 98.4400 | 98.4300 | 98.4300 | 0.2970 |
| K-Nearest Neighbors (KNN) | 97.5800 | 97.5800 | 97.5600 | 97.5700 | 1.0310 |
| Máquina de Vectores de Soporte (SVM) | 97.0700 | 97.0600 | 97.0400 | 97.0500 | 27.3060 |
| Regresión Logística | 92.1500 | 92.0400 | 92.0300 | 92.0200 | 0.0290 |

## K-Means (agrupamiento no supervisado)

- Clústeres: 10
- Adjusted Rand Index (ARI): 0.3657
- Información mutua normalizada (NMI): 0.4979
- Coeficiente de silueta: 0.0873
- Pureza: 59.31 %
- Exactitud tras mapeo clúster→dígito: 59.31 %

K-Means no utiliza etiquetas durante el entrenamiento. La última cifra es solo una interpretación a posteriori.

## Validación cruzada (3 folds)

| nombre | n_muestras | accuracy_mean | accuracy_std | f1_macro_mean | f1_macro_std |
| --- | --- | --- | --- | --- | --- |
| Regresión Logística | 6000 | 0.8950 | 0.0008 | 0.8933 | 0.0009 |
| K-Nearest Neighbors (KNN) | 6000 | 0.9437 | 0.0056 | 0.9430 | 0.0056 |
| Máquina de Vectores de Soporte (SVM) | 6000 | 0.9520 | 0.0029 | 0.9512 | 0.0030 |

## Prueba del pipeline completo

- Modelo: Red Neuronal Convolucional (CNN)
- Secuencias correctas: 100.0 %
- Muestras: 13

Imágenes sintéticas compuestas a partir del test de MNIST con rotación, ruido y desenfoque. No son fotografías reales.
