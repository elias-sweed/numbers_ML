# ============================================================================
# DOCUMENTACION.env
# Documentación completa del proyecto
# Sistema Inteligente de Reconocimiento y Clasificación de Dígitos Manuscritos
# ============================================================================
#
# IMPORTANTE:
# Este archivo NO contiene secretos ni variables de entorno reales. Aunque usa
# la extensión .env, su contenido es documentación técnica en texto plano.
# No debe cargarse con python-dotenv ni usarse como configuración ejecutable.
#
# Proyecto...........: reconocimiento_digitos
# Ruta...............: C:\Users\elias\Documents\SARA\reconocimiento_digitos
# Entorno virtual....: C:\Users\elias\Documents\SARA\.venv
# Fecha de entrega...: 2026-11-11
# Estado.............: completado, entrenado, evaluado y verificado
# Autor..............: proyecto académico de Machine Learning y Deep Learning
# Idioma.............: español
# ============================================================================


# ============================================================================
# 00. ÍNDICE
# ============================================================================
#
#  01. Resumen ejecutivo
#  02. Objetivo y alcance
#  03. Cumplimiento de las fases solicitadas
#  04. Decisiones metodológicas y aclaraciones
#  05. Entorno de ejecución y dependencias
#  06. Estructura completa del proyecto
#  07. Flujo general del sistema
#  08. Dataset MNIST
#  09. Preprocesamiento de imágenes
#  10. Segmentación con OpenCV
#  11. Modelos implementados
#  12. Entrenamiento y artefactos
#  13. Evaluación y métricas
#  14. Resultados reales
#  15. Validación cruzada
#  16. K-Means: evaluación no supervisada
#  17. Resultados por clase de la CNN
#  18. Demostración del pipeline completo
#  19. Aplicación Streamlit
#  20. Referencia completa de la CLI
#  21. Pruebas automatizadas
#  22. Errores encontrados y correcciones aplicadas
#  23. Verificación final realizada
#  24. Reproducción desde cero
#  25. Archivos generados e inventario
#  26. Limitaciones y honestidad académica
#  27. Solución de problemas
#  28. Seguridad y uso responsable
#  29. Glosario
#  30. Checklist final de entrega
#  31. Historial de esta documentación
#
# ============================================================================


# ============================================================================
# 01. RESUMEN EJECUTIVO
# ============================================================================
#
# Se construyó un proyecto académico completo, ejecutable y verificable para el
# reconocimiento de dígitos manuscritos del 0 al 9 con el dataset MNIST.
#
# El sistema:
#
#   1. Descarga MNIST automáticamente con TensorFlow/Keras.
#   2. Divide los datos en 54 000 entrenamiento, 6 000 validación y 10 000 test.
#   3. Preprocesa imágenes nuevas con OpenCV y las adapta al formato MNIST.
#   4. Segmenta uno o varios dígitos de una imagen.
#   5. Entrena y compara seis algoritmos:
#        - Regresión Logística
#        - K-Means (agrupamiento no supervisado)
#        - K-Nearest Neighbors (KNN)
#        - Máquina de Vectores de Soporte (SVM)
#        - Perceptrón Multicapa (MLP)
#        - Red Neuronal Convolucional (CNN)
#   6. Evalúa con accuracy, precision, recall, F1 y matriz de confusión.
#   7. Aplica validación cruzada a los modelos tradicionales.
#   8. Genera gráficos, informes y un resumen en Markdown.
#   9. Ofrece una interfaz Streamlit moderna con cinco pestañas.
#  10. Incluye una demostración end-to-end con imágenes sintéticas.
#  11. Incluye 31 pruebas unitarias, todas aprobadas.
#
# Resultado principal:
#
#   La CNN obtuvo 99.61 % de accuracy y 99.61 % de F1 macro sobre las 10 000
#   imágenes del test oficial de MNIST.
#
# Todos los valores de este documento provienen de ejecuciones reales. No se
# inventaron métricas ni se simularon resultados. Los archivos de resultados
# originales se conservan en results/ y los modelos entrenados en models/.
#
# Estado de la aplicación:
#
#   - La app Streamlit arranca correctamente.
#   - Responde HTTP 200.
#   - AppTest no reporta excepciones.
#   - Reconoce correctamente la secuencia 2026 en la prueba automatizada.
#   - Las pestañas de evaluación, comparación, K-Means y guía funcionan.
#
# ============================================================================


# ============================================================================
# 02. OBJETIVO Y ALCANCE
# ============================================================================
#
# OBJETIVO GENERAL
#
# Diseñar, implementar, entrenar, evaluar y documentar un sistema inteligente
# capaz de reconocer dígitos manuscritos del 0 al 9, comparando técnicas
# clásicas de Machine Learning con redes neuronales de Deep Learning, y
# exponiendo el resultado mediante una aplicación web profesional.
#
# OBJETIVOS ESPECÍFICOS
#
#   - Preparar el dataset MNIST sin fuga de información.
#   - Implementar un preprocesamiento robusto para imágenes nuevas.
#   - Implementar segmentación de varios dígitos en una línea.
#   - Entrenar los seis algoritmos exigidos.
#   - Evaluar con métricas estándar y matrices de confusión.
#   - Aplicar validación cruzada a los modelos tradicionales.
#   - Evaluar K-Means con métricas propias de agrupamiento.
#   - Generar gráficos interpretables.
#   - Construir una interfaz Streamlit funcional y moderna.
#   - Documentar decisiones, limitaciones y resultados reales.
#   - Incluir pruebas automatizadas.
#
# ALCANCE FUNCIONAL
#
#   - Reconocimiento de un dígito individual.
#   - Reconocimiento de números de varios dígitos en una línea.
#   - Carga de PNG, JPG, JPEG, BMP y WEBP.
#   - Soporte para fondo claro con trazo oscuro y fondo oscuro con trazo claro.
#   - Soporte para PNG con transparencia y orientación EXIF.
#   - Selección del modelo desde la interfaz.
#   - Ajuste del área mínima y del máximo de dígitos.
#   - Visualización de la máscara, los recortes 28x28 y la imagen anotada.
#   - Descarga de la imagen anotada.
#   - Consulta de métricas, matrices, curvas, comparación y K-Means.
#
# FUERA DEL ALCANCE
#
#   - Reconocimiento de letras, símbolos o dígitos fuera del rango 0-9.
#   - Reconocimiento de varias líneas de texto.
#   - Detección de dígitos en video en tiempo real.
#   - Entrenamiento distribuido o con GPU.
#   - Autenticación multiusuario.
#   - Despliegue en nube o contenedores.
#   - Corrección manual interactiva de segmentaciones fallidas.
#
# ============================================================================


# ============================================================================
# 03. CUMPLIMIENTO DE LAS FASES SOLICITADAS
# ============================================================================
#
# FASE 01 - Entorno y dependencias
#   Implementación: requirements.txt, main.py info, README.md.
#   Evidencia: Python 3.12.10 y todas las bibliotecas instaladas y verificadas.
#
# FASE 02 - Dataset MNIST
#   Implementación: src/dataset.py y comando `python main.py prepare`.
#   Evidencia: data/mnist_split.npz, data/dataset_info.json y data/labels/.
#
# FASE 03 - Preprocesamiento
#   Implementación: src/preprocessing.py y src/segmentation.py.
#   Evidencia: 31 pruebas unitarias y pipeline end-to-end.
#
# FASE 04 - Algoritmos obligatorios
#   Implementación: src/models.py y src/training.py.
#   Evidencia: los seis modelos entrenados y guardados en models/.
#
# FASE 05 - CNN con capas exigidas
#   Implementación: src/models.py::build_cnn.
#   Capas: Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten,
#          Dense, ReLU y Softmax; optimizador Adam.
#
# FASE 06 - Evaluación y validación cruzada
#   Implementación: src/evaluation.py y `python main.py evaluate`.
#   Evidencia: results/metrics.csv y results/cross_validation.csv.
#
# FASE 07 - Gráficos
#   Implementación: src/evaluation.py.
#   Evidencia: 28 archivos PNG en plots/.
#
# FASE 08 - Software Streamlit
#   Implementación: app.py.
#   Evidencia: app probada con AppTest, HTTP 200 y predicción correcta de 2026.
#
# FASE 09 - Estructura del proyecto
#   Implementación: carpetas app.py, src/, models/, results/, plots/, data/,
#   tests/, README.md, requirements.txt y config.py.
#
# FASE 10 - Comentarios y documentación
#   Implementación: código comentado en español, README.md y este archivo.
#
# FASE 11 - Pruebas y verificación
#   Implementación: tests/.
#   Evidencia: 31 pruebas aprobadas en 5.49 segundos.
#
# ============================================================================


# ============================================================================
# 04. DECISIONES METODOLÓGICAS Y ACLARACIONES
# ============================================================================
#
# 04.01. El test oficial nunca se usa para entrenar
#
#   MNIST trae 60 000 imágenes de entrenamiento y 10 000 de test. Del bloque de
#   60 000 se separaron 6 000 para validación. El test de 10 000 permaneció
#   intacto hasta la evaluación final.
#
#   54 000 train + 6 000 validación + 10 000 test = 70 000 imágenes.
#
# 04.02. La validación se usa para early stopping
#
#   MLP y CNN usan las 6 000 imágenes de validación para:
#     - EarlyStopping.
#     - ReduceLROnPlateau.
#     - ModelCheckpoint.
#     - Selección de la mejor época.
#
# 04.03. K-Means no es un clasificador
#
#   K-Means se entrenó sin etiquetas. Se evalúa con ARI, NMI, silueta y pureza.
#   La "exactitud tras mapeo clúster -> dígito" se calcula a posteriori solo
#   para interpretar los grupos. No se incluye en la tabla de clasificadores
#   supervisados porque sería una comparación metodológicamente incorrecta.
#
# 04.04. SVM entrenado con subconjunto
#
#   El SVM RBF se entrenó con 12 000 imágenes estratificadas para que fuera
#   viable en CPU. Se evaluó sobre las 10 000 imágenes de test. Esta decisión
#   está declarada en el README, en la app y en results/RESUMEN.md.
#
# 04.05. KNN con PCA
#
#   KNN aplica PCA(50) antes de calcular distancias. Con 784 dimensiones el
#   coste y la maldición de la dimensionalidad serían prohibitivos.
#
# 04.06. K-Means con subconjunto aleatorio
#
#   K-Means se ajustó con 20 000 imágenes elegidas aleatoriamente del train.
#   El muestreo no fue estratificado a propósito: K-Means no debe ver etiquetas
#   ni siquiera para construir la muestra.
#
# 04.07. Validación cruzada con subconjunto
#
#   Los modelos tradicionales se validaron con 3 folds sobre 6 000 imágenes
#   estratificadas. Es una estimación de estabilidad, no sustituye al test.
#
# 04.08. Imágenes de demostración sintéticas
#
#   Las 13 imágenes de data/samples/ se construyeron a partir de imágenes del
#   test de MNIST con rotación, ruido y desenfoque. No son fotografías reales.
#   Se usan para probar de forma reproducible la cadena completa con imágenes
#   que el modelo no vio durante el entrenamiento.
#
# 04.09. Confianza del SVM
#
#   El SVM se entrenó sin probability=True para evitar un coste elevado. La
#   confianza mostrada se obtiene aplicando softmax a los márgenes de decisión.
#   Es una medida relativa, no una probabilidad calibrada.
#
# 04.10. Reproducibilidad
#
#   Semilla global: 42.
#   Divisiones estratificadas con random_state fijo.
#   random_state en todos los modelos que lo permiten.
#   TensorFlow puede variar en el último decimal por oneDNN y paralelismo.
#
# ============================================================================


# ============================================================================
# 05. ENTORNO DE EJECUCIÓN Y DEPENDENCIAS
# ============================================================================
#
# 05.01. Sistema
#
#   Sistema operativo....: Windows 11 (10.0.26300)
#   Python...............: 3.12.10
#   Ejecutable...........: C:\Users\elias\Documents\SARA\.venv\Scripts\python.exe
#   Entorno virtual......: C:\Users\elias\Documents\SARA\.venv
#   Directorio proyecto..: C:\Users\elias\Documents\SARA\reconocimiento_digitos
#
# 05.02. Por qué Python 3.12
#
#   En el equipo había Python 3.14, pero TensorFlow 2.21 no publica wheels para
#   cp314. Se instaló Python 3.12 con:
#
#     winget install --id Python.Python.3.12 --scope user
#
#   Después se creó el entorno virtual compartido en la carpeta padre SARA.
#
# 05.03. Versiones exactas instaladas
#
#   tensorflow...........: 2.21.0
#   keras................: 3.15.1
#   opencv-python........: 5.0.0.93
#   numpy................: 2.5.4
#   pandas...............: 3.1.1
#   matplotlib...........: 3.11.2
#   seaborn..............: 0.13.2
#   scikit-learn.........: 1.9.1
#   pillow...............: 13.0.0
#   streamlit............: 1.67.0
#   joblib...............: 1.6.0
#
# 05.04. Instalación desde cero
#
#   cd C:\Users\elias\Documents\SARA\reconocimiento_digitos
#   py -3.12 -m venv ..\.venv
#   ..\.venv\Scripts\python.exe -m pip install --upgrade pip
#   ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
#
# 05.05. Verificación del entorno
#
#   ..\.venv\Scripts\python.exe main.py info
#
#   Este comando muestra versiones, dataset, modelos entrenados y últimas
#   métricas de test.
#
# 05.06. Nota sobre GPU
#
#   TensorFlow 2.21 en Windows nativo no usa GPU. Todo el entrenamiento se hizo
#   en CPU. El aviso de TensorFlow es informativo y no afecta al proyecto.
#
# 05.07. Dependencias declaradas en requirements.txt
#
#   tensorflow==2.21.0
#   keras==3.15.1
#   opencv-python==5.0.0.93
#   numpy==2.5.4
#   pandas==3.1.1
#   matplotlib==3.11.2
#   seaborn==0.13.2
#   scikit-learn==1.9.1
#   pillow==13.0.0
#   streamlit==1.67.0
#   joblib==1.6.0
#
# ============================================================================


# ============================================================================
# 06. ESTRUCTURA COMPLETA DEL PROYECTO
# ============================================================================
#
# reconocimiento_digitos/
# |
# +-- app.py
# |     Interfaz Streamlit con cinco pestañas: Reconocer, Evaluación,
# |     Comparación, K-Means y Guía.
# |
# +-- main.py
# |     Punto de entrada CLI. Comandos: prepare, train, evaluate, demo,
# |     predict, info y all.
# |
# +-- config.py
# |     Rutas, semillas, hiperparámetros y nombres de artefactos.
# |
# +-- requirements.txt
# |     Dependencias con versiones fijadas.
# |
# +-- README.md
# |     Documentación académica y guía de uso.
# |
# +-- DOCUMENTACION.env
# |     Este documento técnico completo.
# |
# +-- .gitignore
# |     Excluye caché, entorno virtual, logs temporales y dataset pesado.
# |
# +-- src/
# |   +-- __init__.py
# |   +-- dataset.py
# |   |     Descarga, validación, división estratificada, caché y metadatos.
# |   +-- preprocessing.py
# |   |     Grises, CLAHE, ruido, Otsu, recorte, escala 20x20, centrado por masa
# |   |     y normalización.
# |   +-- segmentation.py
# |   |     Contornos, filtrado, separación de dígitos pegados, orden de lectura
# |   |     y recortes preparados.
# |   +-- models.py
# |   |     Definición de los seis modelos y utilidades de arquitectura.
# |   +-- training.py
# |   |     Entrenamiento, callbacks, persistencia y metadatos.
# |   +-- evaluation.py
# |   |     Métricas, matrices, curvas, comparación, validación cruzada,
# |   |     evaluación de K-Means y gráficos.
# |   +-- prediction.py
# |   |     Carga de imágenes, EXIF, transparencia, inferencia y confianza.
# |   +-- demo.py
# |   |     Generación de muestras sintéticas y prueba end-to-end.
# |   +-- utils.py
# |   |     Temporización, JSON, submuestreo estratificado y formatos.
# |
# +-- models/
# |   +-- logistic_regression.joblib
# |   +-- knn.joblib
# |   +-- svm.joblib
# |   +-- kmeans.joblib
# |   +-- mlp.keras
# |   +-- cnn.keras
# |
# +-- results/
# |   +-- RESUMEN.md
# |   +-- metrics.csv
# |   +-- cross_validation.csv
# |   +-- kmeans_metrics.json
# |   +-- demo_predictions.json
# |   +-- classification_<modelo>.csv
# |   +-- classification_report_<modelo>.txt
# |   +-- history_mlp.csv
# |   +-- history_cnn.csv
# |   +-- training_<modelo>.json
# |
# +-- plots/
# |   +-- dataset_overview.png
# |   +-- model_comparison.png
# |   +-- confusion_matrix_<modelo>.png
# |   +-- training_mlp.png
# |   +-- training_cnn.png
# |   +-- sample_predictions_cnn.png
# |   +-- kmeans_centers.png
# |   +-- kmeans_cluster_sizes.png
# |   +-- kmeans_clusters_pca.png
# |   +-- kmeans_confusion.png
# |   +-- demo_predictions.png
# |   +-- demo_annotated_*.png
# |
# +-- data/
# |   +-- mnist_split.npz
# |   +-- dataset_info.json
# |   +-- labels/
# |   |   +-- train_labels.csv
# |   |   +-- validation_labels.csv
# |   |   +-- test_labels.csv
# |   +-- samples/
# |       +-- demo_manifest.json
# |       +-- digito_0.png ... digito_9.png
# |       +-- numero_2026.png
# |       +-- numero_31415.png
# |       +-- numero_789.png
# |
# +-- tests/
#     +-- __init__.py
#     +-- test_models.py
#     +-- test_prediction.py
#     +-- test_preprocessing.py
#     +-- test_segmentation.py
#
# ============================================================================


# ============================================================================
# 07. FLUJO GENERAL DEL SISTEMA
# ============================================================================
#
# 07.01. Flujo de preparación y entrenamiento
#
#   MNIST (Keras)
#        |
#        v
#   División estratificada 54k / 6k / 10k
#        |
#        v
#   Caché data/mnist_split.npz
#        |
#        +--> Modelos clásicos: aplanado 784 y normalización
#        |
#        +--> MLP: vector 784
#        |
#        +--> CNN: tensor 28x28x1
#        |
#        v
#   Entrenamiento y guardado en models/
#        |
#        v
#   Evaluación sobre test y generación de results/ y plots/
#
# 07.02. Flujo de reconocimiento de una imagen nueva
#
#   Imagen (ruta, bytes o array)
#        |
#        v
#   Corrección EXIF + composición de transparencia sobre blanco
#        |
#        v
#   Escala de grises
#        |
#        v
#   CLAHE / mejora de contraste
#        |
#        v
#   Filtro de mediana / gaussiano
#        |
#        v
#   Umbralización de Otsu con corrección automática de polaridad
#        |
#        v
#   Morfología: cierre y apertura
#        |
#        v
#   Búsqueda de contornos externos
#        |
#        v
#   Filtrado por área y altura
#        |
#        v
#   Separación heurística de bloques anchos
#        |
#        v
#   Orden de izquierda a derecha
#        |
#        v
#   Recorte con padding
#        |
#        v
#   Preparación 20x20 + centrado por masa en 28x28
#        |
#        v
#   Predicción con el modelo seleccionado
#        |
#        v
#   Número completo + confianza + imágenes de apoyo
#
# 07.03. Flujo de evaluación
#
#   Modelo entrenado
#        |
#        v
#   Preparación del test según el modelo
#        |
#        v
#   Predicción de las 10 000 imágenes
#        |
#        v
#   Accuracy, precision, recall y F1
#        |
#        v
#   Matriz de confusión + informe por clase
#        |
#        v
#   Comparación + curvas + ejemplos
#
# 07.04. Flujo de la aplicación Streamlit
#
#   Barra lateral: modelo, área mínima, máximo de dígitos
#        |
#        v
#   Pestaña Reconocer
#        |
#        +--> Subir imagen o usar ejemplo
#        |
#        v
#   Segmentación + predicción + visualización
#        |
#        +--> Pestañas de evaluación, comparación, K-Means y guía
#
# ============================================================================


# ============================================================================
# 08. DATASET MNIST
# ============================================================================
#
# 08.01. Identidad
#
#   Nombre....: MNIST (Modified National Institute of Standards and Technology)
#   Tipo......: dígitos manuscritos en escala de grises
#   Fuente....: keras.datasets.mnist.load_data()
#   Descarga..: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
#   Caché.....: ~/.keras/datasets/mnist.npz
#   Formato...: 28x28 píxeles, 1 canal, uint8, valores 0-255
#   Clases....: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
#
# 08.02. División utilizada
#
#   Entrenamiento....: 54 000 imágenes
#   Validación.......: 6 000 imágenes
#   Prueba...........: 10 000 imágenes
#   Total............: 70 000 imágenes
#
#   El conjunto de validación se extrajo del bloque oficial de 60 000 con una
#   división estratificada del 10 %. El test oficial nunca se modificó.
#
# 08.03. Distribución de clases en entrenamiento
#
#   Dígito 0....: 5 331
#   Dígito 1....: 6 068
#   Dígito 2....: 5 362
#   Dígito 3....: 5 518
#   Dígito 4....: 5 258
#   Dígito 5....: 4 879
#   Dígito 6....: 5 326
#   Dígito 7....: 5 638
#   Dígito 8....: 5 266
#   Dígito 9....: 5 354
#
# 08.04. Distribución de clases en validación
#
#   Dígito 0....: 592
#   Dígito 1....: 674
#   Dígito 2....: 596
#   Dígito 3....: 613
#   Dígito 4....: 584
#   Dígito 5....: 542
#   Dígito 6....: 592
#   Dígito 7....: 627
#   Dígito 8....: 585
#   Dígito 9....: 595
#
# 08.05. Distribución de clases en test
#
#   Dígito 0....: 980
#   Dígito 1....: 1 135
#   Dígito 2....: 1 032
#   Dígito 3....: 1 010
#   Dígito 4....: 982
#   Dígito 5....: 892
#   Dígito 6....: 958
#   Dígito 7....: 1 028
#   Dígito 8....: 974
#   Dígito 9....: 1 009
#
# 08.06. Archivos generados
#
#   data/mnist_split.npz
#     Arrays comprimidos: X_train, y_train, X_val, y_val, X_test, y_test.
#
#   data/dataset_info.json
#     Metadatos: nombre, fuente, fecha, formas, conteos, distribuciones,
#     semilla, estrategia de división y nota metodológica.
#
#   data/labels/train_labels.csv
#   data/labels/validation_labels.csv
#   data/labels/test_labels.csv
#     Dos columnas: indice y etiqueta.
#
# 08.07. Validaciones aplicadas al dataset
#
#   - Forma esperada de train: (60 000, 28, 28).
#   - Forma esperada de test: (10 000, 28, 28).
#   - Etiquetas entre 0 y 9.
#   - Número de etiquetas igual al número de imágenes.
#   - dtype uint8 para las imágenes originales.
#   - El archivo cacheado debe contener las seis claves requeridas.
#
# 08.08. Prevención de fuga de información
#
#   - El test no participa en el entrenamiento.
#   - El test no participa en early stopping.
#   - El test no se usa para elegir hiperparámetros.
#   - La validación se usa solo para seleccionar la mejor época de MLP y CNN.
#   - La validación cruzada se hace sobre train, no sobre test.
#
# ============================================================================


# ============================================================================
# 09. PREPROCESAMIENTO DE IMÁGENES
# ============================================================================
#
# Archivo: src/preprocessing.py
#
# 09.01. ensure_grayscale
#
#   Convierte BGR, BGRA, RGB, escala de grises o un canal a un array 2D uint8.
#   Si hay canal alfa, compone la transparencia sobre fondo blanco para evitar
#   que las zonas transparentes se vuelvan negras.
#   Rechaza imágenes None, vacías o con dimensiones no soportadas.
#
# 09.02. enhance_contrast
#
#   Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) con:
#     clipLimit = 2.0
#     tileGridSize = (8, 8)
#   Mejora fotos con iluminación desigual sin saturar en exceso.
#
# 09.03. denoise_image
#
#   Métodos disponibles:
#     - median: filtro de mediana, ideal para ruido sal y pimienta.
#     - gaussian: suavizado gaussiano.
#     - bilateral: preserva bordes.
#   El kernel se fuerza a impar y a un mínimo de 3.
#
# 09.04. normalize_pixels
#
#   Convierte a float32 y normaliza a [0, 1].
#   Solo divide entre 255 si el máximo es mayor que 1.5. Este umbral evita la
#   doble normalización cuando un array ya normalizado contiene valores como
#   1.0000001 después de una interpolación.
#   Finalmente aplica clip a [0, 1].
#
# 09.05. resize_digit
#
#   Redimensiona a 28x28 usando INTER_AREA al reducir y INTER_LINEAR al
#   ampliar. Se usa para utilidades generales, no para la normalización MNIST
#   final.
#
# 09.06. binarize_digit
#
#   - Convierte a grises.
#   - Aplica GaussianBlur(3, 3) si blur=True.
#   - Aplica Otsu con THRESH_BINARY + THRESH_OTSU.
#   - Calcula la proporción de foreground.
#   - Si el foreground supera 0.5, invierte la imagen.
#   Resultado: dígito blanco (255) sobre fondo negro (0), independientemente de
#   si la tinta original era oscura sobre claro o clara sobre oscuro.
#
# 09.07. remove_small_objects
#
#   Usa connectedComponentsWithStats con conectividad 8 y elimina componentes
#   cuya área sea menor que min_area.
#
# 09.08. crop_to_content
#
#   Encuentra el bounding box de los píxeles no nulos y recorta al contenido,
#   con margen opcional.
#
# 09.09. center_by_mass
#
#   Calcula el centro de masa de la imagen:
#     cy = suma(y * pixel) / suma(pixel)
#     cx = suma(x * pixel) / suma(pixel)
#   Calcula el desplazamiento hasta el centro geométrico y aplica warpAffine
#   con interpolación lineal y borde negro. Después recorta a [0, 1].
#
# 09.10. resize_and_center
#
#   Réplica de la normalización estándar de MNIST:
#     - Recorta al contenido.
#     - Escala el lado mayor a 20 px conservando la relación de aspecto.
#     - Coloca el resultado en un lienzo de 28x28.
#     - Centra por masa.
#
#   Corrección importante:
#     Antes se redimensionaba la imagen completa a 28x28 antes de recortar. Eso
#     deformaba los dígitos estrechos, especialmente el "1", y podía convertirlos
#     en manchas. Ahora se recorta primero y se escala conservando el aspecto.
#
# 09.11. prepare_digit_for_model
#
#   Cadena para una imagen nueva:
#     - Grises.
#     - Mediana 3x3 si denoise=True.
#     - CLAHE con protección ante imágenes demasiado pequeñas.
#     - Otsu con polaridad automática si binarize=True.
#     - Normalización.
#     - resize_and_center a 28x28.
#   Devuelve float32 (28, 28) en [0, 1].
#
#   Si binarize=False se usa para recortes que ya son máscaras binarias.
#
# 09.12. prepare_dataset_for_ml
#
#   (N, 28, 28) uint8 -> (N, 784) float32 en [0, 1].
#   También acepta (N, 28, 28, 1). Evita dividir dos veces si ya está en [0, 1].
#
# 09.13. prepare_dataset_for_cnn
#
#   (N, 28, 28) -> (N, 28, 28, 1) float32 en [0, 1].
#   También acepta (N, 784) y lo reconstruye.
#
# 09.14. to_uint8 y make_white_background
#
#   - to_uint8 convierte [0, 1] a [0, 255] para visualización.
#   - make_white_background invierte para mostrar dígitos oscuros sobre blanco.
#
# ============================================================================


# ============================================================================
# 10. SEGMENTACIÓN CON OPENCV
# ============================================================================
#
# Archivo: src/segmentation.py
#
# 10.01. Estructura de datos
#
#   SegmentedDigit:
#     index.........: posición de lectura, empezando en 0.
#     bbox..........: (x, y, ancho, alto) en la imagen original.
#     binary_crop...: máscara blanca sobre negro.
#     prepared......: imagen float32 28x28 lista para el modelo.
#     area..........: área del contorno.
#
# 10.02. Binarización para segmentación
#
#   - GaussianBlur(5, 5).
#   - Otsu con polaridad automática.
#   - Cierre morfológico 3x3 para unir trazos cortados.
#   - Apertura morfológica 3x3 para eliminar motas.
#
# 10.03. Búsqueda de contornos
#
#   cv2.findContours con RETR_EXTERNAL y CHAIN_APPROX_SIMPLE.
#   Se descartan contornos con:
#     area < min_area
#     alto < 8
#     ancho < 2
#
# 10.04. Filtrado por tamaño relativo
#
#   Se calcula la altura máxima y el área máxima de los candidatos.
#   Se conservan candidatos con:
#     altura >= 20 % de la altura máxima
#     área >= 1 % del área máxima
#
# 10.05. Separación de dígitos pegados
#
#   Si un bloque es ancho (w >= 1.25 * h y w >= 20), se analiza el perfil
#   vertical de píxeles. Se suaviza el perfil y se busca el mínimo en la zona
#   central (30 % a 70 %). Si el valle es menor que 35 % del pico y la división
#   no produce mitades demasiado estrechas, se parte en dos cajas.
#   Es una heurística; no resuelve todos los casos de caligrafía unida.
#
# 10.06. Orden de lectura
#
#   Los segmentos se ordenan por coordenada X ascendente. La aplicación está
#   diseñada para una línea de dígitos. Varias líneas se ordenarían globalmente
#   por X, no por filas.
#
# 10.07. Recorte y preparación
#
#   Cada caja se recorta con padding de 2 px. Se llama a
#   prepare_digit_for_model(crop, binarize=False, denoise=False), porque el
#   recorte ya es una máscara binaria.
#
# 10.08. Anotación
#
#   annotate_segments dibuja:
#     - rectángulo verde alrededor de cada dígito;
#     - número de orden en una etiqueta verde con texto blanco.
#
# 10.09. Fallback
#
#   Si no se detecta ningún componente, prediction.py procesa la imagen
#   completa como un único dígito y marca fallback=True.
#
# 10.10. Límites
#
#   - Máximo de dígitos configurable, por defecto 15.
#   - Área mínima configurable, por defecto 40 px².
#   - Los dígitos que se tocan pueden seguir fallando.
#
# ============================================================================


# ============================================================================
# 11. MODELOS IMPLEMENTADOS
# ============================================================================
#
# Archivo: src/models.py
#
# 11.01. Regresión Logística
#
#   Clave............: logreg
#   Tipo.............: clasificación supervisada
#   Implementación...: sklearn.linear_model.LogisticRegression
#   Entrada..........: 784 píxeles normalizados
#   Clases...........: 10
#   Hiperparámetros..:
#     C = 1.0
#     solver = lbfgs
#     max_iter = 400
#     tol = 1e-3
#     random_state = 42
#   Entrenamiento....: 54 000 imágenes
#   Artefacto........: models/logistic_regression.joblib
#   Ventaja..........: rápida e interpretable.
#   Limitación.......: solo aprende fronteras lineales.
#
# 11.02. K-Nearest Neighbors
#
#   Clave............: knn
#   Tipo.............: clasificación supervisada
#   Implementación...: Pipeline(PCA + KNeighborsClassifier)
#   Entrada..........: 784 píxeles normalizados
#   PCA..............: 50 componentes
#   Vecinos..........: k = 5
#   Pesos............: distance
#   Métrica..........: euclidiana
#   n_jobs...........: -1
#   Entrenamiento....: 54 000 imágenes
#   Artefacto........: models/knn.joblib
#   Ventaja..........: sencillo y no paramétrico.
#   Limitación.......: inferencia más lenta y sensible a la dimensionalidad.
#
# 11.03. Máquina de Vectores de Soporte
#
#   Clave............: svm
#   Tipo.............: clasificación supervisada
#   Implementación...: sklearn.svm.SVC
#   Entrada..........: 784 píxeles normalizados
#   Kernel...........: RBF
#   C................: 10.0
#   gamma............: scale
#   cache_size.......: 1000 MB
#   probability......: False (no se calibró)
#   random_state.....: 42
#   Entrenamiento....: 12 000 imágenes estratificadas
#   Artefacto........: models/svm.joblib
#   Confianza........: softmax de los márgenes de decisión.
#   Ventaja..........: buena precisión con fronteras no lineales.
#   Limitación.......: coste elevado; se usó un subconjunto.
#
# 11.04. K-Means
#
#   Clave............: kmeans
#   Tipo.............: agrupamiento NO supervisado
#   Implementación...: Pipeline(PCA + KMeans)
#   Entrada..........: 784 píxeles normalizados
#   PCA..............: 50 componentes
#   Clústeres........: 10
#   init.............: k-means++
#   n_init...........: 10
#   max_iter.........: 300
#   tol..............: 1e-4
#   algorithm........: lloyd
#   random_state.....: 42
#   Entrenamiento....: 20 000 imágenes aleatorias sin etiquetas
#   Artefacto........: models/kmeans.joblib
#   Evaluación.......: ARI, NMI, silueta, pureza e inercia.
#   Limitación.......: no clasifica; asume grupos aproximadamente esféricos.
#
# 11.05. Perceptrón Multicapa (MLP)
#
#   Clave............: mlp
#   Tipo.............: red neuronal supervisada
#   Implementación...: TensorFlow/Keras Sequential
#   Entrada..........: 784 píxeles normalizados
#   Arquitectura.....:
#     Input(784)
#     Dense(256, ReLU)
#     Dropout(0.20)
#     Dense(128, ReLU)
#     Dropout(0.20)
#     Dense(10, Softmax)
#   Optimizador......: Adam, learning_rate = 1e-3
#   Pérdida..........: sparse_categorical_crossentropy
#   Métrica..........: accuracy
#   Épocas máximas...: 25
#   Batch size.......: 128
#   Early stopping...: paciencia 4, monitor val_accuracy, restore_best_weights.
#   ReduceLROnPlateau: factor 0.5, paciencia 2, min_lr 1e-5.
#   Parámetros.......: 235 146 entrenables.
#   Artefacto........: models/mlp.keras
#
# 11.06. Red Neuronal Convolucional (CNN)
#
#   Clave............: cnn
#   Tipo.............: red neuronal supervisada
#   Implementación...: TensorFlow/Keras Sequential
#   Entrada..........: 28x28x1
#   Arquitectura.....:
#     Input(28, 28, 1)
#     Conv2D(32, 3x3, padding=same, ReLU)
#     BatchNormalization
#     Conv2D(32, 3x3, padding=same, ReLU)
#     MaxPooling2D(2x2)
#     Dropout(0.25)
#     Conv2D(64, 3x3, padding=same, ReLU)
#     BatchNormalization
#     Conv2D(64, 3x3, padding=same, ReLU)
#     MaxPooling2D(2x2)
#     Dropout(0.25)
#     Flatten
#     Dense(128, ReLU)
#     Dropout(0.50)
#     Dense(10, Softmax)
#   Optimizador......: Adam, learning_rate = 1e-3
#   Pérdida..........: sparse_categorical_crossentropy
#   Métrica..........: accuracy
#   Épocas máximas...: 12
#   Batch size.......: 128
#   Callbacks........: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
#   Parámetros.......: 468 010 entrenables.
#   Artefacto........: models/cnn.keras
#
# 11.07. Detección de modelos Keras
#
#   is_keras_model comprueba la presencia de predict, fit y layers.
#   count_parameters suma el tamaño de los pesos entrenables.
#   model_summary devuelve una descripción textual para metadatos.
#
# ============================================================================


# ============================================================================
# 12. ENTRENAMIENTO Y ARTEFACTOS
# ============================================================================
#
# Archivo: src/training.py
#
# 12.01. Orden de entrenamiento
#
#   TRAIN_ORDER = ["logreg", "knn", "svm", "kmeans", "mlp", "cnn"]
#
# 12.02. Persistencia
#
#   Modelos clásicos y K-Means:
#     joblib.dump(model, path, compress=3)
#
#   Redes neuronales:
#     keras ModelCheckpoint + model.save(...)
#
#   Metadatos:
#     results/training_<modelo>.json
#
#   Historiales de redes:
#     results/history_mlp.csv
#     results/history_cnn.csv
#
# 12.03. Callbacks de MLP y CNN
#
#   ModelCheckpoint:
#     monitor = val_accuracy
#     save_best_only = True
#
#   EarlyStopping:
#     monitor = val_accuracy
#     mode = max
#     patience = 4
#     restore_best_weights = True
#
#   ReduceLROnPlateau:
#     monitor = val_loss
#     factor = 0.5
#     patience = 2
#     min_lr = 1e-5
#
# 12.04. Metadatos reales de entrenamiento
#
#   Modelo      Datos    Tiempo      Val accuracy   Detalle
#   ------------------------------------------------------------------------
#   LogReg      54 000   7.923 s     91.95 %       54k, sin subconjunto
#   KNN         54 000   0.629 s     97.67 %       54k, PCA(50)
#   SVM         12 000   17.006 s    97.02 %       12k estratificado
#   K-Means     20 000   1.529 s     N/A           inercia 599 443.6875
#   MLP         54 000   42.9 s      98.28 %       21 épocas, mejor 17
#   CNN         54 000   471.864 s   99.23 %       12 épocas, mejor 12
#
#   Nota: los tiempos corresponden a la ejecución real en CPU de este equipo.
#   Pueden variar en otro hardware.
#
# 12.05. Detalle de las redes
#
#   MLP:
#     épocas ejecutadas.......: 21
#     mejor época.............: 17
#     mejor val_accuracy......: 0.9828333258628845
#     val_loss................: 0.0723288357257843
#     parámetros entrenables..: 235 146
#     tiempo..................: 42.9 s
#
#   CNN:
#     épocas ejecutadas.......: 12
#     mejor época.............: 12
#     mejor val_accuracy......: 0.9923333525657654
#     val_loss................: 0.02886231057345867
#     parámetros entrenables..: 468 010
#     tiempo..................: 471.864 s (7 min 51.864 s)
#
# 12.06. Carga de modelos
#
#   load_trained_model(name):
#     - Para mlp y cnn usa keras.models.load_model(path, compile=False).
#     - Para el resto usa joblib.load(path).
#     - Lanza FileNotFoundError con instrucciones si el modelo no existe.
#
# 12.07. Tamaños aproximados de los artefactos
#
#   logistic_regression.joblib....: 27.23 KB
#   knn.joblib....................: 9.80 MB
#   svm.joblib....................: 1.94 MB
#   kmeans.joblib.................: 154.55 KB
#   mlp.keras.....................: 2.72 MB
#   cnn.keras.....................: 5.42 MB
#
# ============================================================================


# ============================================================================
# 13. EVALUACIÓN Y MÉTRICAS
# ============================================================================
#
# Archivo: src/evaluation.py
#
# 13.01. Conjunto de evaluación
#
#   Todas las métricas supervisadas se calculan sobre las 10 000 imágenes del
#   test oficial de MNIST, que no participaron en entrenamiento ni validación.
#
# 13.02. Métricas calculadas
#
#   accuracy:
#     proporción de predicciones correctas.
#
#   precision_macro:
#     precisión media no ponderada de las 10 clases.
#
#   recall_macro:
#     recall medio no ponderado de las 10 clases.
#
#   f1_macro:
#     media armónica de precision y recall, con el mismo peso por clase.
#
#   precision_weighted, recall_weighted, f1_weighted:
#     variantes ponderadas por el soporte de cada clase.
#
#   tiempo_inferencia_s:
#     tiempo medido para predecir las 10 000 imágenes.
#
# 13.03. Matriz de confusión
#
#   Se genera una figura con dos paneles:
#     - recuentos absolutos;
#     - proporción por clase real.
#   Archivo: plots/confusion_matrix_<modelo>.png
#
# 13.04. Informes por clase
#
#   Para cada modelo se guarda:
#     results/classification_<modelo>.csv
#       columnas: digito, precision, recall, f1_score, soporte.
#     results/classification_report_<modelo>.txt
#       informe completo con precision, recall, f1 y support.
#
# 13.05. Comparación de modelos
#
#   plot_model_comparison genera barras agrupadas de accuracy, precision macro,
#   recall macro y F1 macro para los cinco clasificadores supervisados.
#   Archivo: plots/model_comparison.png
#
# 13.06. Curvas de entrenamiento
#
#   plot_training_curves genera dos paneles:
#     - accuracy de entrenamiento y validación por época;
#     - pérdida de entrenamiento y validación por época.
#   Archivos: plots/training_mlp.png y plots/training_cnn.png
#
# 13.07. Ejemplos de predicción
#
#   plot_sample_predictions selecciona 16 imágenes reales del test con semilla
#   fija, muestra la etiqueta real, la predicha y la confianza.
#   Archivo: plots/sample_predictions_cnn.png
#
# 13.08. Vista general del dataset
#
#   plot_dataset_overview muestra:
#     - distribución de clases en entrenamiento;
#     - rejilla de 25 ejemplos.
#   Archivo: plots/dataset_overview.png
#
# 13.09. Evaluación de K-Means
#
#   Se calculan:
#     - Adjusted Rand Index (ARI);
#     - Normalized Mutual Information (NMI);
#     - coeficiente de silueta sobre una muestra de 5 000 imágenes;
#     - pureza;
#     - exactitud tras mapeo clúster -> dígito mayoritario;
#     - inercia.
#
#   La silueta se calcula en el espacio PCA(50), no en 784 dimensiones.
#
# 13.10. Validación cruzada
#
#   StratifiedKFold con:
#     n_splits = 3
#     shuffle = True
#     random_state = 42
#   Subconjunto estratificado de 6 000 imágenes.
#   Métricas: accuracy, precision_macro, recall_macro, f1_macro.
#   Se calculan media y desviación estándar de los 3 folds.
#
# 13.11. Orden de la evaluación
#
#   1. Cargar cada clasificador entrenado.
#   2. Preparar el test según el modelo.
#   3. Predecir.
#   4. Calcular métricas.
#   5. Guardar informe CSV y TXT.
#   6. Dibujar matriz de confusión.
#   7. Guardar metrics.csv ordenado por accuracy.
#   8. Dibujar comparación.
#   9. Dibujar curvas de MLP y CNN.
#  10. Dibujar ejemplos del mejor modelo.
#  11. Evaluar K-Means.
#  12. Dibujar vista del dataset.
#  13. Ejecutar validación cruzada si no se usa --no-cv.
#
# ============================================================================


# ============================================================================
# 14. RESULTADOS REALES
# ============================================================================
#
# 14.01. Métricas finales sobre test (10 000 imágenes)
#
#   Modelo                         Accuracy  Precision  Recall  F1 macro
#   -----------------------------------------------------------------------
#   CNN                            99.61 %   99.61 %    99.60 % 99.61 %
#   MLP                            98.44 %   98.44 %    98.43 % 98.43 %
#   KNN                            97.58 %   97.58 %    97.56 % 97.57 %
#   SVM                            97.07 %   97.06 %    97.04 % 97.05 %
#   Regresión Logística            92.15 %   92.04 %    92.03 % 92.02 %
#
# 14.02. Métricas ponderadas y tiempos de inferencia
#
#   Modelo             Precision_w  Recall_w  F1_weighted  Inferencia
#   -----------------------------------------------------------------------
#   CNN                99.6102 %    99.6100 % 99.6099 %    1.872 s
#   MLP                98.4415 %    98.4400 % 98.4399 %    0.297 s
#   KNN                97.5804 %    97.5800 % 97.5773 %    1.031 s
#   SVM                97.0715 %    97.0700 % 97.0681 %    27.306 s
#   Regresión Logística 92.1354 %   92.1500 % 92.1296 %    0.029 s
#
# 14.03. Lectura de los resultados
#
#   - La CNN es el mejor modelo con diferencia.
#   - La MLP es la segunda mejor y aprende representaciones no lineales.
#   - KNN sorprende por su buen rendimiento con PCA(50).
#   - SVM queda ligeramente por debajo de KNN pese a usar kernel RBF.
#   - La Regresión Logística es la más rápida en inferencia, pero la menos
#     precisa por su frontera lineal.
#   - El SVM es el más lento en inferencia (27.306 s para 10 000 imágenes).
#
# 14.04. Interpretación académica
#
#   La jerarquía CNN > MLP > KNN > SVM > LogReg es coherente con la literatura:
#   las convoluciones explotan la estructura espacial de la imagen; una red
#   densa ya mejora a los modelos lineales; KNN y SVM son competitivos con
#   representaciones adecuadas; la regresión logística queda limitada por su
#   linealidad.
#
# 14.05. Fuentes de los datos
#
#   results/metrics.csv
#   results/classification_report_<modelo>.txt
#   results/classification_<modelo>.csv
#   results/RESUMEN.md
#
# ============================================================================


# ============================================================================
# 15. VALIDACIÓN CRUZADA
# ============================================================================
#
# 15.01. Configuración
#
#   Subconjunto....: 6 000 imágenes estratificadas del train.
#   Folds..........: 3.
#   Métrica........: accuracy, precision macro, recall macro y F1 macro.
#   Semilla........: 42.
#
# 15.02. Resultados reales
#
#   Modelo       Accuracy media  Desv. estándar  F1 macro medio  Desv. F1
#   -----------------------------------------------------------------------
#   SVM          95.20 %         ± 0.29 %        95.12 %         ± 0.30 %
#   KNN          94.37 %         ± 0.56 %        94.30 %         ± 0.56 %
#   LogReg       89.50 %         ± 0.08 %        89.33 %         ± 0.09 %
#
# 15.03. Tiempos totales de validación cruzada
#
#   LogReg........: 9.59 s
#   KNN...........: 0.696 s
#   SVM...........: 9.757 s
#
# 15.04. Detalle de precision y recall
#
#   LogReg:
#     precision macro = 0.8939576022400941 ± 0.0007760349142964701
#     recall macro    = 0.8933923903966333 ± 0.0009678495642831463
#
#   KNN:
#     precision macro = 0.9446460361321231 ± 0.005489444454593745
#     recall macro    = 0.9425393172078499 ± 0.005437007655078071
#
#   SVM:
#     precision macro = 0.9515175441287735 ± 0.0032170563091077097
#     recall macro    = 0.9510953771791933 ± 0.002895894280180272
#
# 15.05. Archivo
#
#   results/cross_validation.csv
#
# 15.06. Advertencia metodológica
#
#   La validación cruzada se ejecutó sobre un subconjunto de 6 000 imágenes por
#   viabilidad computacional. No sustituye la evaluación final sobre las 10 000
#   imágenes de test.
#
# ============================================================================


# ============================================================================
# 16. K-MEANS: EVALUACIÓN NO SUPERVISADA
# ============================================================================
#
# 16.01. Configuración
#
#   Clústeres........: 10
#   PCA..............: 50 componentes
#   Muestra..........: 20 000 imágenes aleatorias del train
#   Etiquetas........: no usadas durante el ajuste
#   Inicialización...: k-means++
#   n_init...........: 10
#   max_iter.........: 300
#   Semilla..........: 42
#
# 16.02. Métricas reales
#
#   Adjusted Rand Index (ARI)................: 0.36573505373338905
#   Normalized Mutual Information (NMI)......: 0.4978674302946518
#   Coeficiente de silueta...................: 0.08729181438684464
#   Pureza...................................: 59.31 %
#   Exactitud tras mapeo clúster -> dígito...: 59.31 %
#   Inercia..................................: 599 443.6875
#   Tiempo de inferencia.....................: 0.095 s
#
# 16.03. Mapeo aprendido a posteriori
#
#   Clúster 0 -> dígito 8
#   Clúster 1 -> dígito 1
#   Clúster 2 -> dígito 6
#   Clúster 3 -> dígito 1
#   Clúster 4 -> dígito 0
#   Clúster 5 -> dígito 4
#   Clúster 6 -> dígito 3
#   Clúster 7 -> dígito 7
#   Clúster 8 -> dígito 2
#   Clúster 9 -> dígito 0
#
# 16.04. Interpretación
#
#   K-Means separa grupos por similitud de píxeles, no por clase. Puede dividir
#   un mismo dígito en varios clústeres por estilo, grosor o inclinación. Los
#   clústeres 1 y 3 comparten el dígito 1; los clústeres 0 y 9 comparten el
#   dígito 0; varios dígitos no tienen un clúster propio. Esto explica la baja
#   pureza y demuestra por qué K-Means no debe presentarse como clasificador.
#
# 16.05. Archivos
#
#   results/kmeans_metrics.json
#   plots/kmeans_centers.png
#   plots/kmeans_cluster_sizes.png
#   plots/kmeans_clusters_pca.png
#   plots/kmeans_confusion.png
#
# ============================================================================


# ============================================================================
# 17. RESULTADOS POR CLASE DE LA CNN
# ============================================================================
#
# 17.01. Informe completo de la CNN sobre test
#
#   Dígito   Precision   Recall   F1-score   Soporte
#   -------------------------------------------------
#   0        0.9959      0.9969   0.9964     980
#   1        0.9956      0.9982   0.9969     1 135
#   2        0.9952      0.9971   0.9961     1 032
#   3        0.9941      0.9970   0.9956     1 010
#   4        0.9980      0.9980   0.9980     982
#   5        0.9955      0.9933   0.9944     892
#   6        0.9969      0.9916   0.9942     958
#   7        0.9980      0.9942   0.9961     1 028
#   8        0.9969      0.9969   0.9969     974
#   9        0.9951      0.9970   0.9960     1 009
#
#   Accuracy global...: 0.9961
#   Macro avg.........: precision 0.9961, recall 0.9960, F1 0.9961
#   Weighted avg......: precision 0.9961, recall 0.9961, F1 0.9961
#   Tiempo inferencia.: 1.872 s
#
# 17.02. Fortalezas y debilidades observadas
#
#   - La mejor clase es el 4, con F1 0.9980.
#   - La clase con menor F1 es el 5, con 0.9944.
#   - La clase 6 tiene el recall más bajo, 0.9916.
#   - Incluso la peor clase supera el 99.1 % de F1.
#
# 17.03. Informes de los demás modelos
#
#   Los informes completos están en:
#     results/classification_report_mlp.txt
#     results/classification_report_knn.txt
#     results/classification_report_svm.txt
#     results/classification_report_logreg.txt
#
#   Las métricas por dígito están en:
#     results/classification_mlp.csv
#     results/classification_knn.csv
#     results/classification_svm.csv
#     results/classification_logreg.csv
#     results/classification_cnn.csv
#
# ============================================================================


# ============================================================================
# 18. DEMOSTRACIÓN DEL PIPELINE COMPLETO
# ============================================================================
#
# Archivo: src/demo.py
#
# 18.01. Objetivo
#
#   Probar de forma reproducible toda la cadena:
#     generación de imagen -> carga -> segmentación -> preprocesamiento ->
#     predicción -> anotación -> métricas.
#
# 18.02. Generación
#
#   Las imágenes se construyen a partir de imágenes del test de MNIST:
#     - inversión para obtener trazo oscuro sobre fondo blanco;
#     - escala a una altura objetivo;
#     - rotación aleatoria entre -7 y +7 grados;
#     - ruido gaussiano con desviación 4.5;
#     - desenfoque gaussiano con radio 0.45;
#     - composición sobre lienzo blanco;
#     - separación de 26 px entre dígitos;
#     - margen de 38 px.
#
#   No son fotografías reales. Se documenta así en la app, el README y el JSON.
#
# 18.03. Imágenes generadas
#
#   Dígitos individuales:
#     digito_0.png ... digito_9.png
#
#   Secuencias:
#     numero_2026.png   -> 2026
#     numero_31415.png  -> 31415
#     numero_789.png    -> 789
#
# 18.04. Resultados reales
#
#   Archivo           Esperado   Predicho   Confianza media   Correcto
#   -------------------------------------------------------------------
#   digito_0.png      0          0          100.00 %          sí
#   digito_1.png      1          1          100.00 %          sí
#   digito_2.png      2          2          100.00 %          sí
#   digito_3.png      3          3          100.00 %          sí
#   digito_4.png      4          4          100.00 %          sí
#   digito_5.png      5          5          99.9961 %         sí
#   digito_6.png      6          6          100.00 %          sí
#   digito_7.png      7          7          100.00 %          sí
#   digito_8.png      8          8          100.00 %          sí
#   digito_9.png      9          9          100.00 %          sí
#   numero_2026.png   2026       2026       99.9943 %         sí
#   numero_31415.png  31415      31415      99.9990 %         sí
#   numero_789.png    789        789        99.9990 %         sí
#
#   Secuencias correctas: 13 de 13 = 100.0 %.
#
# 18.05. Confianza por dígito en las secuencias
#
#   numero_2026.png:
#     2 -> 0.999999
#     0 -> 0.999775
#     2 -> 1.0
#     6 -> 1.0
#
#   numero_31415.png:
#     3 -> 1.0
#     1 -> 0.999971
#     4 -> 1.0
#     1 -> 1.0
#     5 -> 0.999981
#
#   numero_789.png:
#     7 -> 1.0
#     8 -> 1.0
#     9 -> 0.999971
#
# 18.06. Archivos
#
#   data/samples/demo_manifest.json
#   results/demo_predictions.json
#   plots/demo_predictions.png
#   plots/demo_annotated_digito_0.png ... demo_annotated_digito_9.png
#   plots/demo_annotated_numero_2026.png
#   plots/demo_annotated_numero_31415.png
#   plots/demo_annotated_numero_789.png
#
# ============================================================================


# ============================================================================
# 19. APLICACIÓN STREAMLIT
# ============================================================================
#
# Archivo: app.py
#
# 19.01. Configuración de página
#
#   Título......: Reconocimiento de Dígitos · ML & DL
#   Icono.......: 🔢
#   Layout......: wide
#   Sidebar.....: expandida
#
# 19.02. Diseño visual
#
#   - Fondo con degradado claro.
#   - Tipografía Segoe UI / Inter / system-ui.
#   - Hero con degradado azul y verde.
#   - Tarjetas de métricas con bordes redondeados y sombra suave.
#   - Tarjeta de resultado con número grande y confianza.
#   - Filas de dígitos con chip, barra de confianza y porcentaje.
#   - Pestañas con estilo propio.
#   - Colores de confianza:
#       >= 90 %: verde.
#       70-90 %: ámbar.
#       < 70 %: rojo.
#
# 19.03. Barra lateral
#
#   - Selector de modelo entrenado.
#   - CNN como opción por defecto si está disponible.
#   - Slider de área mínima: 10 a 400 px², paso 10, por defecto 40.
#   - Slider de máximo de dígitos: 1 a 15, por defecto 15.
#   - Métricas del modelo seleccionado: accuracy y F1 macro.
#   - Información del dataset.
#   - Nota de honestidad académica.
#
# 19.04. Pestaña Reconocer
#
#   Fuentes:
#     - Subir imagen PNG, JPG, JPEG, BMP o WEBP.
#     - Usar ejemplo del manifiesto de data/samples/.
#
#   Al procesar:
#     - Carga el modelo con caché de Streamlit.
#     - Segmenta y predice.
#     - Muestra el número reconocido y la confianza media.
#     - Indica si coincide con el valor esperado del ejemplo.
#     - Avisa si se usó fallback.
#     - Muestra confianza por dígito con barras.
#     - Permite ver probabilidades por clase del primer dígito.
#     - Permite descargar la imagen anotada en PNG.
#
#   Visualizaciones:
#     - Imagen original.
#     - Umbralización de Otsu.
#     - Segmentación y predicción.
#     - Recortes 28x28 preparados para el modelo.
#     - Ejemplos de test si aún no hay imagen.
#
#   Interpretación de confianza:
#     - >= 90 %: predicción muy segura.
#     - 70-90 %: predicción probable.
#     - < 70 %: el modelo duda; revisar ruido, trazo o segmentación.
#     - SVM: confianza relativa, no probabilidad calibrada.
#
# 19.05. Pestaña Evaluación
#
#   - Tarjetas: mejor modelo, modelos evaluados, mejor F1, imágenes de test.
#   - Tabla de métricas ordenada por accuracy.
#   - Selector de modelo para inspeccionar.
#   - Matriz de confusión.
#   - Métricas por clase.
#   - Informe completo en texto.
#   - Curvas de entrenamiento de CNN y MLP.
#   - Tabla de validación cruzada.
#   - Ejemplos reales de test con predicción.
#
# 19.06. Pestaña Comparación
#
#   - Gráfico comparativo de los cinco clasificadores.
#   - Tabla de accuracy, precision, recall y F1.
#   - Explicaciones desplegables de cada algoritmo:
#       Regresión Logística, KNN, SVM, MLP, CNN y K-Means.
#   - Nota metodológica: K-Means no se mezcla con clasificadores.
#
# 19.07. Pestaña K-Means
#
#   - Tarjetas de ARI, NMI, silueta y pureza.
#   - Exactitud tras mapeo e inercia.
#   - Centroides reconstruidos.
#   - Tamaño de clústeres.
#   - Proyección PCA 2D: clústeres frente a etiquetas reales.
#   - Matriz de confusión del mapeo.
#   - Tabla del mapeo clúster -> dígito mayoritario.
#   - Explicación de clústeres duplicados por estilo.
#
# 19.08. Pestaña Guía
#
#   - Flujo de la aplicación.
#   - Comandos útiles.
#   - Información del dataset.
#   - Recomendaciones para fotos reales.
#   - Limitaciones declaradas.
#   - Estructura del proyecto.
#
# 19.09. Caché
#
#   @st.cache_resource para modelos.
#   @st.cache_data para métricas, JSON, manifiesto y textos.
#   Esto evita recargar modelos y archivos en cada interacción.
#
# 19.10. Ejecución
#
#   Opción principal:
#
#     cd C:\Users\elias\Documents\SARA\reconocimiento_digitos
#     ..\.venv\Scripts\streamlit.exe run app.py
#
#   Opción alternativa:
#
#     ..\.venv\Scripts\python.exe -m streamlit run app.py
#
#   URL por defecto: http://localhost:8501
#
# 19.11. Verificación de la app
#
#   - AppTest ejecutado desde archivo: 0 excepciones.
#   - Selección de numero_2026.png: predicción correcta "2026".
#   - Servidor HTTP: respuesta 200.
#   - Navegador: página cargada sin errores de consola.
#   - Captura visual no disponible por visibilidad del escritorio; se verificó
#     mediante snapshot de accesibilidad y consola.
#
# ============================================================================


# ============================================================================
# 20. REFERENCIA COMPLETA DE LA CLI
# ============================================================================
#
# Archivo: main.py
#
# 20.01. Ayuda general
#
#   ..\.venv\Scripts\python.exe main.py --help
#
# 20.02. prepare
#
#   ..\.venv\Scripts\python.exe main.py prepare
#   ..\.venv\Scripts\python.exe main.py prepare --force
#
#   Descarga MNIST si no existe, divide train/validación, guarda la caché,
#   las etiquetas y los metadatos.
#
#   --force regenera la división aunque exista la caché.
#
# 20.03. train
#
#   ..\.venv\Scripts\python.exe main.py train --model all
#   ..\.venv\Scripts\python.exe main.py train --model cnn
#   ..\.venv\Scripts\python.exe main.py train --model logreg knn svm
#   ..\.venv\Scripts\python.exe main.py train --model kmeans mlp
#   ..\.venv\Scripts\python.exe main.py train --model all --skip-existing
#
#   Claves válidas:
#     logreg, knn, svm, kmeans, mlp, cnn, all.
#
#   --skip-existing omite modelos cuyo artefacto ya existe.
#
# 20.04. evaluate
#
#   ..\.venv\Scripts\python.exe main.py evaluate
#   ..\.venv\Scripts\python.exe main.py evaluate --no-cv
#   ..\.venv\Scripts\python.exe main.py evaluate --models cnn mlp
#
#   Evalúa sobre test, genera métricas, informes, matrices, comparación,
#   curvas, ejemplos, K-Means y validación cruzada.
#
#   --no-cv omite la validación cruzada.
#   --models limita la evaluación a los modelos indicados.
#
# 20.05. demo
#
#   ..\.venv\Scripts\python.exe main.py demo
#   ..\.venv\Scripts\python.exe main.py demo --model cnn
#
#   Genera las imágenes sintéticas, ejecuta el pipeline completo, guarda
#   results/demo_predictions.json y plots/demo_predictions.png.
#
# 20.06. predict
#
#   ..\.venv\Scripts\python.exe main.py predict --image ruta\foto.png
#   ..\.venv\Scripts\python.exe main.py predict --image ruta\foto.png --model cnn
#   ..\.venv\Scripts\python.exe main.py predict --image ruta\foto.png --model cnn --min-area 60
#
#   Imprime:
#     - modelo utilizado;
#     - ruta de la imagen;
#     - número de dígitos detectados;
#     - número reconocido;
#     - confianza media;
#     - detalle por dígito;
#     - ruta de la imagen anotada.
#
#   Guarda results/prediccion_<nombre>.png.
#
# 20.07. info
#
#   ..\.venv\Scripts\python.exe main.py info
#
#   Muestra:
#     - versión de Python y ejecutable;
#     - si el entorno virtual está activo;
#     - sistema operativo;
#     - versiones de las bibliotecas;
#     - conteos del dataset;
#     - modelos entrenados y tamaños;
#     - últimas métricas de test.
#
# 20.08. all
#
#   ..\.venv\Scripts\python.exe main.py all
#   ..\.venv\Scripts\python.exe main.py all --force-data
#   ..\.venv\Scripts\python.exe main.py all --skip-existing
#
#   Ejecuta prepare -> train all -> evaluate -> demo -> write_summary.
#   Puede tardar varios minutos en CPU.
#
# 20.09. Códigos de salida
#
#   0....: éxito.
#   1....: error controlado o error de entrenamiento.
#   130..: cancelación con Ctrl+C.
#
# ============================================================================


# ============================================================================
# 21. PRUEBAS AUTOMATIZADAS
# ============================================================================
#
# Comando:
#
#   ..\.venv\Scripts\python.exe -m unittest discover -s tests -v
#
# Resultado real:
#
#   Ran 31 tests in 5.490s
#   OK
#
# 21.01. tests/test_models.py (5 pruebas)
#
#   - test_classic_models:
#       construye Regresión Logística, KNN y SVM.
#   - test_cnn_architecture:
#       verifica entrada (None, 28, 28, 1), salida (None, 10), más de 100 000
#       parámetros y capas conv_1, pool_1, flatten, densa y salida.
#   - test_mlp_architecture:
#       verifica entrada (None, 784) y salida (None, 10).
#   - test_logistic_regression_learns_toy_data:
#       ajusta con datos aleatorios y comprueba forma de predicción.
#   - test_kmeans_fits:
#       ajusta un K-Means reducido y comprueba etiquetas.
#
# 21.02. tests/test_prediction.py (7 pruebas)
#
#   - test_softmax:
#       suma 1 y clase correcta.
#   - test_load_image_from_bytes:
#       carga una imagen PNG desde bytes.
#   - test_transparent_png_is_composed_on_white:
#       fondo transparente se compone sobre blanco.
#   - test_predict_multi_digit:
#       segmenta "1 2 3" y devuelve tres predicciones.
#   - test_predict_blank_falls_back:
#       imagen vacía activa fallback y una predicción.
#   - test_predict_single_digit:
#       predicción individual y confianza.
#   - test_probabilities_shape:
#       probabilidades con forma (1, 10) y suma 1.
#
# 21.03. tests/test_preprocessing.py (13 pruebas)
#
#   Preprocesamiento:
#     - conversión BGR a grises;
#     - normalización uint8;
#     - ausencia de doble normalización;
#     - redimensionado;
#     - binarización de trazo oscuro sobre fondo claro;
#     - binarización de trazo claro sobre fondo oscuro;
#     - centrado por masa;
#     - preparación de un dígito;
#     - resize_and_center;
#     - preparación por lotes para ML y CNN.
#
#   Dataset:
#     - formas y conteos 54k/6k/10k;
#     - etiquetas entre 0 y 9 y longitudes coherentes;
#     - suma total de 70 000 imágenes.
#
# 21.04. tests/test_segmentation.py (6 pruebas)
#
#   - test_segment_five_digits:
#       segmenta "1 2 3 4 5", ordena por X y prepara 28x28.
#   - test_single_digit:
#       segmenta un único "8".
#   - test_blank_image_returns_empty:
#       imagen vacía devuelve lista vacía.
#   - test_annotate:
#       la anotación conserva forma y canales.
#   - test_preview_is_binary:
#       la vista previa solo contiene 0 y 255.
#   - test_full_image_fallback:
#       prepara la imagen completa como un dígito.
#
# 21.05. Advertencias esperadas
#
#   TensorFlow muestra mensajes de oneDNN y de ausencia de GPU en Windows.
#   Son informativos, no fallos. No afectan al resultado de las pruebas.
#
# ============================================================================


# ============================================================================
# 22. ERRORES ENCONTRADOS Y CORRECCIONES APLICADAS
# ============================================================================
#
# 22.01. Doble normalización
#
#   Síntoma:
#     Las imágenes nuevas podían quedar casi negras tras varias etapas de
#     interpolación.
#
#   Causa:
#     Un array ya normalizado podía contener valores ligeramente superiores a
#     1.0, y la comprobación `max > 1.0` lo volvía a dividir entre 255.
#
#   Corrección:
#     normalize_pixels solo divide si max > 1.5. Se añadió la prueba
#     test_no_double_normalization.
#
# 22.02. Distorsión de relación de aspecto
#
#   Síntoma:
#     El dígito "1" y otros trazos estrechos se convertían en manchas.
#
#   Causa:
#     resize_and_center redimensionaba la imagen completa a 28x28 antes de
#     recortar, deformando el contenido.
#
#   Corrección:
#     Primero se recorta al contenido y luego se escala el lado mayor a 20 px
#     conservando la proporción. Después se centra en 28x28.
#
# 22.03. Parámetro obsoleto en LogisticRegression
#
#   Síntoma:
#     Advertencia por `n_jobs=-1`.
#
#   Corrección:
#     Se eliminó `n_jobs` de LogisticRegression.
#
# 22.04. Parámetro innecesario en SVC
#
#   Síntoma:
#     `probability=False` explícito y coste asociado.
#
#   Corrección:
#     Se eliminó el parámetro. El SVM usa softmax de márgenes para la confianza.
#
# 22.05. Falso fallo de entrenamiento en PowerShell
#
#   Síntoma:
#     El comando de entrenamiento devolvía código 1 por redirección de stderr.
#
#   Diagnóstico:
#     Los resúmenes y artefactos demostraron que ambos modelos se entrenaron
#     correctamente. Fue un artefacto de shell, no un fallo del código.
#
# 22.06. Captura de pantalla no disponible
#
#   Síntoma:
#     La captura del navegador falló porque la ventana de escritorio no estaba
#     visible.
#
#   Solución:
#     Se verificó la app con snapshot de accesibilidad, consola, HTTP 200 y
#     AppTest. No fue un fallo funcional.
#
# 22.07. Limpieza final
#
#   Se eliminaron:
#     - plots/_debug_*.png;
#     - plots/_sanity_segmentation.png;
#     - results/streamlit_out.txt;
#     - results/streamlit_err.txt;
#     - results/train_log.txt;
#     - results/train_redes_log.txt;
#     - directorios __pycache__.
#   Se detuvo el servidor Streamlit de prueba en el puerto 8503.
#
# ============================================================================


# ============================================================================
# 23. VERIFICACIÓN FINAL REALIZADA
# ============================================================================
#
# 23.01. Entorno
#
#   Comando:
#     ..\.venv\Scripts\python.exe main.py info
#
#   Resultado:
#     Python 3.12.10.
#     Entorno virtual activo.
#     TensorFlow 2.21.0, Keras 3.15.1, OpenCV 5.0.0.93, NumPy 2.5.4,
#     Pandas 3.1.1, scikit-learn 1.9.1, Streamlit 1.67.0, Pillow 13.0.0.
#     Dataset 54 000 / 6 000 / 10 000.
#     Los seis modelos entrenados.
#
# 23.02. Pruebas
#
#   31 pruebas aprobadas.
#   Tiempo: 5.490 s.
#
# 23.03. Aplicación
#
#   AppTest:
#     - 0 excepciones.
#     - predicción correcta de "2026".
#
#   Servidor:
#     - HTTP 200 en el puerto de prueba 8503.
#     - página cargada.
#     - sin errores de consola.
#
# 23.04. Gráficos verificados visualmente
#
#   - model_comparison.png.
#   - confusion_matrix_cnn.png.
#   - training_cnn.png.
#   - kmeans_clusters_pca.png.
#   - demo_predictions.png.
#
# 23.05. Artefactos verificados
#
#   - results/RESUMEN.md.
#   - results/metrics.csv.
#   - results/cross_validation.csv.
#   - results/kmeans_metrics.json.
#   - results/demo_predictions.json.
#   - Los seis modelos en models/.
#   - Los 28 gráficos en plots/.
#
# 23.06. Estado de limpieza
#
#   - Servidor de prueba detenido.
#   - Archivos temporales eliminados.
#   - Cachés __pycache__ eliminadas.
#   - No hay procesos del proyecto escuchando en el puerto 8503.
#
# ============================================================================


# ============================================================================
# 24. REPRODUCCIÓN DESDE CERO
# ============================================================================
#
# 24.01. Requisitos previos
#
#   - Windows 11 o compatible.
#   - Python 3.12 instalado.
#   - Conexión a Internet para descargar MNIST y dependencias.
#   - Espacio libre suficiente para el dataset, modelos y resultados.
#
# 24.02. Crear el entorno virtual
#
#   cd C:\Users\elias\Documents\SARA\reconocimiento_digitos
#   py -3.12 -m venv ..\.venv
#   ..\.venv\Scripts\python.exe -m pip install --upgrade pip
#   ..\.venv\Scripts\python.exe -m pip install -r requirements.txt
#
# 24.03. Verificar
#
#   ..\.venv\Scripts\python.exe main.py info
#
# 24.04. Preparar el dataset
#
#   ..\.venv\Scripts\python.exe main.py prepare
#
# 24.05. Entrenar
#
#   Todos los modelos:
#     ..\.venv\Scripts\python.exe main.py train --model all
#
#   Solo algunos:
#     ..\.venv\Scripts\python.exe main.py train --model logreg knn svm
#     ..\.venv\Scripts\python.exe main.py train --model kmeans mlp cnn
#
# 24.06. Evaluar
#
#   ..\.venv\Scripts\python.exe main.py evaluate
#
# 24.07. Generar y probar la demo
#
#   ..\.venv\Scripts\python.exe main.py demo
#
# 24.08. Ejecutar pruebas
#
#   ..\.venv\Scripts\python.exe -m unittest discover -s tests -v
#
# 24.09. Abrir la aplicación
#
#   ..\.venv\Scripts\streamlit.exe run app.py
#
# 24.10. Reproducir todo de una vez
#
#   ..\.venv\Scripts\python.exe main.py all
#
# 24.11. Tiempos aproximados observados
#
#   Preparación del dataset....: menos de 1 minuto si ya está en caché.
#   LogReg.....................: 8 s.
#   KNN.......................: menos de 1 s de ajuste.
#   SVM.......................: 17 s.
#   K-Means...................: 1.5 s.
#   MLP.......................: 43 s.
#   CNN.......................: 7 min 52 s.
#   Evaluación completa.......: depende de la validación cruzada y SVM.
#   Pruebas...................: 5.5 s.
#
#   Los tiempos pueden variar según CPU, carga del sistema y versión de las
#   bibliotecas.
#
# 24.12. Reproducibilidad esperada
#
#   Las métricas pueden variar en el último decimal por:
#     - optimizaciones oneDNN de TensorFlow;
#     - paralelismo de CPU;
#     - diferencias de versión;
#     - inicialización aleatoria de las redes.
#   La semilla 42 reduce la variabilidad, pero no garantiza bit a bit el mismo
#   resultado en hardware distinto.
#
# ============================================================================


# ============================================================================
# 25. ARCHIVOS GENERADOS E INVENTARIO
# ============================================================================
#
# 25.01. Código y documentación
#
#   app.py
#   main.py
#   config.py
#   requirements.txt
#   README.md
#   DOCUMENTACION.env
#   .gitignore
#   src/__init__.py
#   src/dataset.py
#   src/preprocessing.py
#   src/segmentation.py
#   src/models.py
#   src/training.py
#   src/evaluation.py
#   src/prediction.py
#   src/demo.py
#   src/utils.py
#
# 25.02. Modelos
#
#   models/logistic_regression.joblib
#   models/knn.joblib
#   models/svm.joblib
#   models/kmeans.joblib
#   models/mlp.keras
#   models/cnn.keras
#
# 25.03. Resultados
#
#   results/RESUMEN.md
#   results/metrics.csv
#   results/cross_validation.csv
#   results/kmeans_metrics.json
#   results/demo_predictions.json
#
#   results/classification_cnn.csv
#   results/classification_knn.csv
#   results/classification_logreg.csv
#   results/classification_mlp.csv
#   results/classification_svm.csv
#
#   results/classification_report_cnn.txt
#   results/classification_report_knn.txt
#   results/classification_report_logreg.txt
#   results/classification_report_mlp.txt
#   results/classification_report_svm.txt
#
#   results/history_cnn.csv
#   results/history_mlp.csv
#
#   results/training_cnn.json
#   results/training_kmeans.json
#   results/training_knn.json
#   results/training_logreg.json
#   results/training_mlp.json
#   results/training_svm.json
#
# 25.04. Gráficos (28 archivos PNG)
#
#   plots/dataset_overview.png
#   plots/model_comparison.png
#
#   plots/confusion_matrix_cnn.png
#   plots/confusion_matrix_knn.png
#   plots/confusion_matrix_logreg.png
#   plots/confusion_matrix_mlp.png
#   plots/confusion_matrix_svm.png
#
#   plots/training_cnn.png
#   plots/training_mlp.png
#   plots/sample_predictions_cnn.png
#
#   plots/kmeans_centers.png
#   plots/kmeans_cluster_sizes.png
#   plots/kmeans_clusters_pca.png
#   plots/kmeans_confusion.png
#
#   plots/demo_predictions.png
#   plots/demo_annotated_digito_0.png ... demo_annotated_digito_9.png
#   plots/demo_annotated_numero_2026.png
#   plots/demo_annotated_numero_31415.png
#   plots/demo_annotated_numero_789.png
#
# 25.05. Datos
#
#   data/mnist_split.npz
#   data/dataset_info.json
#   data/labels/train_labels.csv
#   data/labels/validation_labels.csv
#   data/labels/test_labels.csv
#   data/samples/demo_manifest.json
#   data/samples/digito_0.png ... digito_9.png
#   data/samples/numero_2026.png
#   data/samples/numero_31415.png
#   data/samples/numero_789.png
#
# 25.06. Pruebas
#
#   tests/__init__.py
#   tests/test_models.py
#   tests/test_prediction.py
#   tests/test_preprocessing.py
#   tests/test_segmentation.py
#
# 25.07. Archivos temporales eliminados
#
#   plots/_debug_*.png
#   plots/_sanity_segmentation.png
#   results/streamlit_out.txt
#   results/streamlit_err.txt
#   results/train_log.txt
#   results/train_redes_log.txt
#   __pycache__/
#
# ============================================================================


# ============================================================================
# 26. LIMITACIONES Y HONESTIDAD ACADÉMICA
# ============================================================================
#
# 26.01. SVM con subconjunto
#
#   El SVM se entrenó con 12 000 imágenes, no con 54 000. Se hizo por coste
#   computacional y se declara en todos los resultados.
#
# 26.02. KNN con PCA
#
#   El KNN no calcula distancias en 784 dimensiones; usa PCA(50). Es una decisión
#   de viabilidad y reducción de ruido.
#
# 26.03. Validación cruzada parcial
#
#   La validación cruzada usa 6 000 imágenes y 3 folds. Es una estimación de
#   estabilidad, no una evaluación sobre las 54 000.
#
# 26.04. K-Means no supervisado
#
#   K-Means no clasifica. Su accuracy tras mapeo es una interpretación a
#   posteriori y no debe compararse con la accuracy de un clasificador.
#
# 26.05. Imágenes de demo sintéticas
#
#   Las 13 imágenes de demostración son composiciones de MNIST con rotación,
#   ruido y desenfoque. No son fotos reales. Las fotos reales pueden requerir
#   ajuste del área mínima, mejor iluminación o recorte.
#
# 26.06. Confianza del SVM
#
#   La confianza del SVM es softmax de márgenes, no una probabilidad calibrada.
#
# 26.07. Segmentación
#
#   Los dígitos que se tocan pueden segmentarse mal. La heurística de perfil de
#   proyección ayuda, pero no resuelve todos los casos.
#
# 26.08. Una sola línea
#
#   El sistema está pensado para una línea de dígitos. No procesa varias líneas
#   como filas independientes.
#
# 26.09. Dominio MNIST
#
#   MNIST contiene dígitos centrados, en escala de grises y con un estilo
#   concreto. El rendimiento puede bajar con letra muy atípica, fondos complejos,
#   sombras fuertes o cámaras de baja calidad.
#
# 26.10. Sin GPU
#
#   Todo se entrenó en CPU. Los tiempos son los de este equipo.
#
# 26.11. Sin datos personales
#
#   No se usan datos personales ni se envían imágenes a servicios externos.
#   Todo el procesamiento es local.
#
# ============================================================================


# ============================================================================
# 27. SOLUCIÓN DE PROBLEMAS
# ============================================================================
#
# 27.01. Error: No matching distribution found for tensorflow
#
#   Causa: Python 3.14 no tiene wheels de TensorFlow 2.21.
#   Solución: instalar Python 3.12 y recrear el entorno virtual.
#
# 27.02. Error: ModuleNotFoundError: config
#
#   Causa: ejecutar desde otra carpeta.
#   Solución: cd a reconocimiento_digitos antes de ejecutar main.py o app.py.
#
# 27.03. Error: modelo no encontrado
#
#   Causa: el artefacto no existe.
#   Solución:
#     ..\.venv\Scripts\python.exe main.py train --model cnn
#
# 27.04. MNIST no descarga
#
#   Causa: sin Internet o bloqueo de storage.googleapis.com.
#   Solución: revisar conexión, proxy o firewall y repetir prepare.
#
# 27.05. Streamlit no abre
#
#   Solución:
#     ..\.venv\Scripts\python.exe -m streamlit run app.py
#   Revisar que el puerto 8501 esté libre.
#
# 27.06. No detecta dígitos
#
#   Solución:
#     - bajar el área mínima;
#     - usar fondo claro y trazo oscuro;
#     - recortar la zona del número;
#     - mejorar la iluminación.
#
# 27.07. Detecta ruido como dígitos
#
#   Solución:
#     - subir el área mínima;
#     - recortar la imagen;
#     - limpiar el fondo.
#
# 27.08. Dígitos pegados
#
#   Solución:
#     - dejar más espacio al escribir;
#     - subir una imagen más nítida;
#     - separar manualmente los dígitos.
#
# 27.09. Entrenamiento lento
#
#   Solución:
#     - reducir épocas en config.py;
#     - entrenar solo un modelo;
#     - usar --skip-existing;
#     - tener en cuenta que la CNN tarda varios minutos en CPU.
#
# 27.10. Advertencia de GPU en TensorFlow
#
#   Es informativa en Windows nativo. El proyecto funciona en CPU.
#
# 27.11. La app no muestra métricas
#
#   Causa: no se ha ejecutado evaluate.
#   Solución:
#     ..\.venv\Scripts\python.exe main.py evaluate
#
# 27.12. La app no muestra K-Means
#
#   Causa: no está entrenado o evaluado.
#   Solución:
#     ..\.venv\Scripts\python.exe main.py train --model kmeans
#     ..\.venv\Scripts\python.exe main.py evaluate
#
# 27.13. Caracteres extraños en la consola
#
#   Causa: página de códigos de PowerShell.
#   Solución: los archivos están en UTF-8; abrirlos con un editor UTF-8 o
#   ejecutar `chcp 65001` si se desea mejorar la visualización.
#
# ============================================================================


# ============================================================================
# 28. SEGURIDAD Y USO RESPONSABLE
# ============================================================================
#
# 28.01. Sin secretos
#
#   Este archivo no contiene contraseñas, tokens, claves API ni credenciales.
#   No debe tratarse como un .env real.
#
# 28.02. Procesamiento local
#
#   Las imágenes se procesan en el equipo donde se ejecuta Streamlit. No se
#   envían a servidores externos.
#
# 28.03. Exposición de red
#
#   Streamlit puede mostrar una URL externa si el equipo está en red. Para uso
#   local, no compartir esa URL. Si se despliega, añadir autenticación y HTTPS.
#
# 28.04. Archivos subidos
#
#   La app no guarda las imágenes subidas por defecto. Solo guarda la imagen
#   anotada si el usuario pulsa descargar y la guarda manualmente.
#
# 28.05. Modelos
#
#   Los archivos .joblib y .keras deben cargarse solo de fuentes confiables.
#   La deserialización de joblib puede ejecutar código si el archivo está
#   manipulado.
#
# 28.06. Uso académico
#
#   El proyecto es académico. No debe usarse para decisiones críticas sin
#   validación adicional, calibración y control de errores.
#
# ============================================================================


# ============================================================================
# 29. GLOSARIO
# ============================================================================
#
# Accuracy:
#   proporción de predicciones correctas sobre el total.
#
# ARI (Adjusted Rand Index):
#   medida de concordancia entre dos agrupamientos, corregida por azar.
#
# Batch size:
#   número de muestras procesadas antes de actualizar los pesos.
#
# CLAHE:
#   ecualización adaptativa del histograma con límite de contraste.
#
# CNN:
#   red neuronal convolucional; usa filtros que comparten pesos para detectar
#   patrones locales.
#
# Dropout:
#   regularización que apaga neuronas al azar durante el entrenamiento.
#
# Early stopping:
#   detiene el entrenamiento cuando la validación deja de mejorar.
#
# Época:
#   una pasada completa por el conjunto de entrenamiento.
#
# F1-score:
#   media armónica de precision y recall.
#
# K-Means:
#   algoritmo de agrupamiento que asigna puntos al centroide más cercano.
#
# KNN:
#   clasificador basado en los k vecinos más cercanos.
#
# Matriz de confusión:
#   tabla que cruza etiqueta real con etiqueta predicha.
#
# MLP:
#   perceptrón multicapa; red densa con capas ocultas.
#
# NMI:
#   información mutua normalizada; mide cuánta información comparten dos
#   agrupamientos.
#
# Otsu:
#   método automático para elegir un umbral de binarización.
#
# PCA:
#   análisis de componentes principales; reduce dimensionalidad.
#
# Precision:
#   de lo predicho como positivo, qué proporción era correcta.
#
# Pureza:
#   proporción de muestras que pertenecen a la clase mayoritaria de su clúster.
#
# Recall:
#   de lo que era realmente positivo, qué proporción se detectó.
#
# Silueta:
#   medida de cohesión interna y separación entre clústeres.
#
# Softmax:
#   función que convierte puntuaciones en probabilidades que suman 1.
#
# SVM:
#   máquina de vectores de soporte; busca el margen máximo entre clases.
#
# Validación cruzada:
#   evaluación repetida con distintas particiones de entrenamiento y prueba.
#
# ============================================================================


# ============================================================================
# 30. CHECKLIST FINAL DE ENTREGA
# ============================================================================
#
# [x] Python 3.12 instalado y entorno virtual funcional.
# [x] requirements.txt con versiones fijadas.
# [x] MNIST descargado y dividido 54k/6k/10k.
# [x] Test oficial intacto hasta la evaluación final.
# [x] Preprocesamiento implementado y probado.
# [x] Segmentación con OpenCV implementada y probada.
# [x] Regresión Logística entrenada.
# [x] K-Means entrenado como agrupamiento.
# [x] KNN entrenado con PCA(50).
# [x] SVM entrenado con RBF y subconjunto de 12k.
# [x] MLP entrenada con callbacks.
# [x] CNN entrenada con callbacks.
# [x] Métricas de test calculadas.
# [x] Matrices de confusión generadas.
# [x] Curvas de entrenamiento generadas.
# [x] Comparación de modelos generada.
# [x] Predicciones de ejemplo generadas.
# [x] Gráficos de K-Means generados.
# [x] Validación cruzada ejecutada.
# [x] Resultados guardados en CSV, JSON y Markdown.
# [x] Demo end-to-end con 13/13 secuencias correctas.
# [x] Aplicación Streamlit con cinco pestañas.
# [x] Carga de imágenes y selección de ejemplos.
# [x] Visualización de máscara, recortes y anotación.
# [x] Descarga de imagen anotada.
# [x] README completo.
# [x] DOCUMENTACION.env completo.
# [x] 31 pruebas unitarias aprobadas.
# [x] AppTest sin excepciones.
# [x] HTTP 200 verificado.
# [x] Predicción correcta de 2026 verificada.
# [x] Gráficos principales revisados visualmente.
# [x] Archivos temporales eliminados.
# [x] Servidor de prueba detenido.
#
# ============================================================================


# ============================================================================
# 31. HISTORIAL DE ESTA DOCUMENTACIÓN
# ============================================================================
#
# 2026-11-11
#   - Creación de DOCUMENTACION.env.
#   - Recopilación del entorno exacto, estructura, decisiones, modelos,
#     hiperparámetros, resultados, pruebas, verificación y limitaciones.
#   - Inclusión de los resultados reales de test, validación cruzada, K-Means,
#     CNN por clase y demostración end-to-end.
#   - Inclusión de comandos de reproducción y solución de problemas.
#
# ============================================================================
#
# FIN DE DOCUMENTACION.env
#
# Este documento resume todo el trabajo realizado en el proyecto. Para detalles
# adicionales, consultar README.md, results/RESUMEN.md y los archivos de
# results/ y plots/.
#
# ============================================================================
