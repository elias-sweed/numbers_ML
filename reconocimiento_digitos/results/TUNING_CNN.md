# Ajuste de hiperparámetros de la CNN

Generado: 2026-11-12T00:20:40

> Experimentos reales ejecutados sobre un subconjunto estratificado del
> conjunto de entrenamiento. La selección se hizo exclusivamente con el
> conjunto de validación. El conjunto de test no se utilizó.

## Protocolo

- Subconjunto de entrenamiento: 20,000 imágenes.
- Validación: 6,000 imágenes.
- Épocas por experimento: 5.
- Semilla: 42.
- Criterio de selección: mayor `best_val_accuracy`; desempate por menor `val_loss_best_epoch`.

## Resultados

| Experimento | Factor | LR | Filtros | Dropout | Batch | Mejor val accuracy | val_loss | Época |
|---|---|---:|---|---:|---:|---:|---:|---:|
| dropout_bajo | dropout | 0.001 | 32/64 | 0.15/0.35 | 128 | 0.9855 | 0.0558 | 5 |
| batch_64 | batch size | 0.001 | 32/64 | 0.25/0.5 | 64 | 0.9840 | 0.0740 | 5 |
| lr_5e-4 | learning rate | 0.0005 | 32/64 | 0.25/0.5 | 128 | 0.9830 | 0.0601 | 5 |
| filtros_16_32 | cantidad de filtros | 0.001 | 16/32 | 0.25/0.5 | 128 | 0.9818 | 0.0639 | 5 |
| lr_2e-3 | learning rate | 0.002 | 32/64 | 0.25/0.5 | 128 | 0.9807 | 0.0660 | 5 |
| dropout_alto | dropout | 0.001 | 32/64 | 0.35/0.6 | 128 | 0.9783 | 0.0837 | 4 |
| filtros_48_96 | cantidad de filtros | 0.001 | 48/96 | 0.25/0.5 | 128 | 0.9763 | 0.0892 | 5 |
| base | configuración base | 0.001 | 32/64 | 0.25/0.5 | 128 | 0.9760 | 0.0869 | 5 |
| batch_256 | batch size | 0.001 | 32/64 | 0.25/0.5 | 256 | 0.9568 | 0.2250 | 5 |

## Configuración seleccionada

- Experimento: `dropout_bajo`.
- Descripción: Regularización más suave: dropout 0.15/0.35.
- Mejor accuracy de validación: 98.55 %.
- Pérdida de validación en la mejor época: 0.0558.

## Justificación

La configuración 'dropout_bajo' obtuvo la mejor accuracy de validación (98.55 %), con una mejora de 0.95 puntos porcentuales frente a la base. Se registra como configuración candidata. El modelo de producción no se sobrescribió: para adoptarla habría que reentrenar con las 54 000 imágenes, reevaluar en test y actualizar los artefactos.

## Limitaciones del experimento

- El ajuste se hizo sobre un subconjunto de entrenamiento y un número
  reducido de épocas para que fuera viable en CPU. Es un cribado
  comparativo, no un entrenamiento final con las 54 000 imágenes.
- El modelo CNN de producción conserva su entrenamiento original con
  54 000 imágenes y 12 épocas. No se sobrescribió ningún artefacto.
- Las diferencias pequeñas entre configuraciones pueden deberse a la
  variabilidad de inicialización y al número limitado de épocas.

## Archivos

- `results/cnn_tuning.csv`
- `results/cnn_tuning.json`
- `results/tuning/history_<experimento>.csv`
- `plots/cnn_tuning.png`
