# Fotografías reales

Esta carpeta es para **fotografías reales** de números escritos a mano en papel.
No debe contener imágenes sintéticas generadas por `src/demo.py` ni capturas de
MNIST.

## Cómo añadir una fotografía

1. Escribe un número en papel blanco con tinta oscura.
2. Haz una foto con el móvil, con buena luz y sin sombras fuertes.
3. Recorta la zona del número si la foto es muy grande.
4. Guarda la imagen en esta carpeta, por ejemplo `foto_2026.jpg`.
5. Añade una fila a `manifest.csv`:

```csv
archivo,esperado,modelo,min_area,tipo,descripcion
foto_2026.jpg,2026,cnn,40,foto_real,Número 2026 en papel blanco
```

Si no conoces el valor esperado, deja `esperado` vacío. Si no indicas modelo o
área mínima, se usan los valores de la CLI.

## Ejecutar la prueba

```powershell
cd C:\Users\elias\Documents\SARA\reconocimiento_digitos
..\.venv\Scripts\python.exe main.py real-test
```

También puedes probar una foto directamente en la app Streamlit, pestaña
**Reconocer**, con el botón **Subir imagen**.

## Reglas de honestidad

- Solo se aceptan filas con `tipo=foto_real`.
- Las imágenes de `data/samples/` son sintéticas y no cuentan como fotos reales.
- Si no hay fotos, el informe debe indicar que la prueba real está pendiente.
