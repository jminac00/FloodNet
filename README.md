# FloodNet Track 1 — U-Net híbrida CNN + ViT

Segmentación semántica de imágenes aéreas tomadas por dron después del huracán
Harvey: asignar a **cada píxel** una de diez clases (agua, edificio inundado,
carretera anegada, vehículo, vegetación…). El objetivo práctico es delimitar la
extensión de la inundación y qué infraestructura ha quedado afectada.

Todo el trabajo vive en un único cuaderno, `floodnet-segmentation.ipynb`, pensado
para leerse de arriba abajo: monta el entorno, audita los datos, construye el
modelo, lo entrena y lo evalúa, explicando en cada paso por qué se hace así y no
de otra forma. El cuaderno del repositorio está ejecutado de principio a fin, así
que todas las cifras de aquí abajo se pueden contrastar con sus salidas.

---

## Resultados

Conjunto de test: ~41 imágenes que no se han visto en ningún momento del
entrenamiento. Se evalúa el checkpoint que mejor mIoU dio en validación, con
*test-time augmentation* por volteos.

| Métrica | 10 clases | Sin *Background* |
|---|---:|---:|
| mIoU | 0.714 | 0.777 |
| Dice | 0.809 | 0.871 |
| Pixel accuracy | 0.927 | — |

Desglose por clase:

| Clase | Dice | IoU |
|---|---:|---:|
| Water | 0.955 | 0.915 |
| Grass | 0.950 | 0.905 |
| Building-flooded | 0.908 | 0.831 |
| Building-non-flooded | 0.891 | 0.803 |
| Tree | 0.883 | 0.790 |
| Road-non-flooded | 0.875 | 0.777 |
| Pool | 0.837 | 0.719 |
| Road-flooded | 0.809 | 0.679 |
| Vehicle | 0.729 | 0.573 |
| Background | 0.251 | 0.144 |

Las dos tablas son la misma medición: el mIoU de 0.714 es exactamente la media
de la columna IoU, y 0.777 esa misma media dejando *Background* fuera. Ambas
salen de una única matriz de confusión acumulada sobre todo el test, no de
promediar resultados por lote.

Tres lecturas:

- **Las clases inundadas funcionan.** *Building-flooded* llega a 0.831 de IoU y
  *Road-flooded* a 0.679, que son las dos clases que de verdad importan para la
  tarea. Salen bien pese a ser minoritarias, lo que sugiere que el oversampling
  con augmentación diferenciada y los pesos por clase están haciendo su trabajo.
- **Los puntos débiles son *Vehicle* y *Background*.** Los vehículos (0.573)
  ocupan pocos píxeles a 512×512 y se pierden en el reescalado. *Background*
  (0.144) no es un objeto: es la etiqueta residual de lo que no encaja en las
  otras nueve, ocupa una fracción mínima de la imagen y no tiene una apariencia
  consistente que aprender. Por eso se reporta también el mIoU sin ella.
- **El test son ~41 imágenes.** Conviene leer estas cifras como un orden de
  magnitud. Una diferencia de dos o tres puntos entre variantes del modelo no
  sería distinguible del ruido con una muestra así.

### Nota sobre cómo se miden

Dice e IoU se calculan acumulando la matriz de confusión del conjunto entero y
dividiendo una sola vez al final (`SegMetrics`, sección 5.5 del cuaderno).
Suena a detalle, pero el cuaderno de partida lo hacía promediando el IoU de cada
lote y eso producía dos errores que se compensaban a medias:

- La fórmula `(intersección + ε) / (unión + ε)` devuelve `ε/ε = 1.0` cuando una
  clase no aparece **ni en la predicción ni en la máscara**. La clase puntuaba
  perfecto por estar ausente. Con diez clases y lotes de cuatro imágenes casi
  siempre hay varias ausentes, así que el mIoU global salía inflado.
- Promediar cocientes por lote no da el cociente del conjunto, y encima el
  resultado depende del tamaño de lote: un lote con tres píxeles de *Vehicle*
  pesaba lo mismo que otro con treinta mil.

Ahora una clase sin píxeles en todo el conjunto sale `NaN` y queda fuera de la
media, que es la convención habitual en segmentación. El desglose por clase y el
número global son consistentes por construcción.

---

## Contenido del repositorio

| Archivo | Qué es |
|---|---|
| [floodnet-segmentation.ipynb](floodnet-segmentation.ipynb) | El cuaderno principal, ejecutado entero. Entorno, datos, modelo, entrenamiento y evaluación. |
| [hybrid-cnn-vit-attention-u-net.ipynb](hybrid-cnn-vit-attention-u-net.ipynb) | El cuaderno de Kaggle que se tomó como punto de partida. Se conserva para poder comparar. |
| [pyproject.toml](pyproject.toml) | Dependencias del proyecto, gestionadas con `uv`. |
| [uv.lock](uv.lock) | Versiones exactas resueltas, para reproducir el entorno bit a bit. |

El dataset, la caché de imágenes y los checkpoints entrenados (~350 MB cada uno)
quedan fuera del repositorio por tamaño; los excluye [.gitignore](.gitignore).

---

## Los datos

Se usan dos datasets de Kaggle, y hace falta bajar los dos:

| Dataset | Para qué |
|---|---|
| [FloodNet Challenge — aerial imagery](https://www.kaggle.com/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge) | El dataset oficial del reto. De aquí sale `class_mapping.csv`, el mapa de índice a nombre de clase. |
| [FloodNet segmentation — fixed masks](https://www.kaggle.com/datasets/catashiro31/floodnet-dataset-for-segmentation-fixed-mask) | Las mismas 398 imágenes supervisadas, pero con las máscaras reparadas. Es la fuente real de entrenamiento. |

Descomprímelos bajo `data/` con esta estructura:

```
data/
├── FloodNet Challenge - Track 1/
│   ├── class_mapping.csv
│   ├── Train/ · Validation/ · Test/
├── data_finally/
│   ├── train/  (image · mask · mask_colored — 317 imágenes)
│   └── val/    (image · mask · mask_colored —  81 imágenes)
└── cache512/   ← lo genera el cuaderno, no lo descargues
```

### Por qué hacen falta las máscaras reparadas

Parte de las máscaras del dataset oficial están corruptas: etiquetan los
4000×3000 píxeles enteros con una sola clase. La máscara de la imagen `10165`,
por ejemplo, marca la foto completa como *Water*, y su PNG pesa 15 KB para 12
megapíxeles — el tamaño al que comprime una imagen de color uniforme. Entrenar
con ellas enseña al modelo justo lo contrario de lo que se busca. La sección 1.2
del cuaderno incluye la auditoría que las localiza.

### Por qué el split está rehecho

FloodNet Track 1 tiene 398 imágenes con máscara (`Train/Labeled`); sus
particiones `Validation` y `Test` oficiales son ciegas, sin etiquetas. El
cuaderno de partida entrenaba con las 398 y validaba sobre `data_finally/val`,
que son **81 de esas mismas 398**: el 100 % de su validación se había visto
durante el entrenamiento, así que sus métricas estaban infladas.

Aquí `data_finally` es la única fuente y se respeta su separación:

| Partición | Origen | Imágenes |
|---|---|---:|
| train | `data_finally/train` | 317 |
| val | mitad de `data_finally/val` | 40 |
| test | otra mitad de `data_finally/val` | 41 |

El cuaderno comprueba con un `assert` que ninguna imagen aparece en dos
particiones.

---

## Puesta en marcha

El entorno se gestiona con [uv](https://docs.astral.sh/uv/), que sustituye a
`pip` + `venv` + `pyenv`. Se encarga también de descargar Python 3.12, que es lo
que PyTorch soporta con ruedas CUDA.

```bash
# Instalar uv (Windows/PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Crear el entorno e instalar todo
uv sync

# Registrar el kernel para Jupyter / VS Code
uv run python -m ipykernel install --user --name floodnet --display-name "FloodNet (Python 3.12)"
```

La primera sincronización descarga del orden de 3 GB, porque PyTorch con CUDA
pesa lo suyo. Las siguientes son casi instantáneas gracias a la caché de `uv`.

Después, abre el cuaderno y **selecciona el kernel `FloodNet (Python 3.12)`**
antes de ejecutar nada más. Las secciones 0.x del propio cuaderno repiten estos
pasos y verifican que la GPU se ve.

Las ruedas apuntan a **CUDA 12.6**. Con otra versión de CUDA, o sin GPU, cambia
el índice de PyTorch en `pyproject.toml` por el que corresponda.

---

## La arquitectura

Un encoder híbrido CNN + Transformer alimentando un decoder U-Net con
*attention gates*:

```
imagen 512×512
      │
      ▼
ConvNeXt-Tiny (preentrenado)  →  x1 x2 x3 x4   skips multiescala
      │
      ▼
Patch embedding 4×4  +  posición  →  tokens (dim 512)
      │
      ▼
12 bloques Transformer (MHSA 8 cabezas + MLP)   contexto global
      │
      ▼
ASPP (rates 1·6·12·18)   cuello de botella multiescala
      │
      ▼
Decoder ×4 con attention gates sobre cada skip
      │
      ▼
Cabeza 1×1 → 10 clases     (+ 2 cabezas auxiliares, deep supervision)
```

La idea detrás de la mezcla: la CNN aporta el sesgo inductivo de localidad y
texturas que hace falta con solo 317 imágenes de entrenamiento, mientras los
bloques Transformer dan el campo receptivo global que distingue una carretera
anegada de un río — algo que depende del contexto de toda la escena, no del
parche local. Las *attention gates* filtran cada skip antes de concatenarlo, para
que el decoder no reintroduzca ruido de las capas superficiales.

---

## Entrenamiento

| | |
|---|---|
| Pérdida | Focal + Dice + Lovász-Softmax, con pesos por clase `1 / log(1.02 + n)` |
| Optimizador | AdamW, lr 3e-5, weight decay 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts (T₀ = 10, T_mult = 2) |
| Precisión | Mixta (AMP) con GradScaler |
| Batch | 4 imágenes de 512×512 |
| Épocas | 30, con 554 iteraciones cada una tras el oversampling |
| Regularización | Early stopping sobre la pérdida de validación |
| Inferencia | TTA: promedio sobre la imagen y sus tres volteos |

Mejores cifras de validación alcanzadas: mIoU 0.716, Dice 0.795, accuracy 0.932.

Un par de decisiones que merecen explicación:

- **Oversampling con dos pipelines de augmentación.** Las imágenes que contienen
  clases raras se repiten dentro de cada época. Las copias reciben augmentación
  agresiva (deformación elástica, distorsión de rejilla, rotaciones amplias) y
  los originales una suave, de modo que las repeticiones aporten variedad real en
  vez de ser duplicados que invitan al sobreajuste.
- **Caché a 512×512.** Los originales son JPEG de 4000×3000. Decodificar 12
  megapíxeles en cada época convertiría la CPU en el cuello de botella. Se
  reescalan una sola vez: la imagen con `INTER_AREA` y la máscara con
  `INTER_NEAREST`, porque interpolar suavemente una máscara inventaría clases
  que no existen (el punto medio entre la clase 2 y la 4 sería un 3, que es otra
  clase distinta, no una mezcla).

En una RTX 4060 Laptop, una época ronda los 2 min 40 s. El cuaderno completo,
entrenamiento incluido, tarda unos 75 minutos.

---

## Correcciones respecto al cuaderno de partida

Además de la fuga de datos y de las máscaras rotas, el cuaderno arregla cinco
errores del original que afectaban al entrenamiento o a su medición:

- **Las métricas.** Promediaban IoU por lote y regalaban un 1.0 a cada clase
  ausente. Explicado arriba, en la nota sobre cómo se miden.
- Los **pesos por clase** estaban escritos a mano y correspondían a otro conjunto
  de entrenamiento. Ahora se cuentan sobre el split real.
- Faltaba `scaler.unscale_(optimizer)` antes de `clip_grad_norm_`. Con AMP los
  gradientes vienen multiplicados por el factor de escala (del orden de 2¹⁶), así
  que recortar a `max_norm=0.5` sobre gradientes escalados no recortaba nada: la
  protección contra gradientes explosivos era decorativa.
- `scheduler.step(val_loss)` pasaba una métrica a `CosineAnnealingWarmRestarts`,
  que interpreta su argumento como el **número de época** (lo de la métrica es
  `ReduceLROnPlateau`). El learning rate saltaba de forma errática durante todo
  el entrenamiento.
- `torch.cuda.amp.GradScaler()` está deprecado en favor de
  `torch.amp.GradScaler('cuda')`.

Y uno que solo afectaba a la visualización: las imágenes salen del DataLoader
normalizadas, con valores negativos, y pasárselas directamente a `imshow`
producía colores falseados.

---

## Limitaciones conocidas

- **La pérdida de validación no es comparable con la de entrenamiento.**
  `val_loss_fn` recibe las probabilidades promediadas por el TTA y se las pasa a
  `nn.CrossEntropyLoss`, que espera logits. La curva sirve para ver la tendencia
  y para el early stopping (que en esta ejecución nunca llegó a dispararse), pero
  su valor absoluto está en otra escala que el de entrenamiento. Las métricas
  reportadas arriba no dependen de esto.
- **No se fija semilla en PyTorch.** El split es reproducible (`SEED = 42` en
  `train_test_split`), pero la inicialización de pesos y el orden de los lotes no,
  así que dos entrenamientos no dan pesos idénticos.

---

## Por dónde seguir

- Aprovechar las 1.047 imágenes sin etiquetar de `Train/Unlabeled` con
  entrenamiento semisupervisado (pseudo-etiquetado o consistencia).
- Validación cruzada en lugar de un único split: con 40 imágenes de validación,
  la varianza entre particiones pesa más que muchas de las diferencias que se
  miden.
- Probar resolución 768 o inferencia por tiles. *Vehicle* es la clase que más lo
  pediría: a 512 píxeles un coche ocupa muy pocos píxeles y su IoU lo refleja.
- Tratar *Background* como `ignore_index` también en la pérdida, no solo al
  reportar, y ver si el resto de clases gana algo.

---

## Créditos

El dataset procede del [FloodNet Challenge](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
(Rahnemoonfar et al.), imágenes de dron posteriores al huracán Harvey. El punto
de partida del modelo es el cuaderno de Kaggle conservado en este repositorio.
