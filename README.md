# FloodNet — U-Net híbrida CNN + ViT

Segmentación semántica de imágenes aéreas tomadas por dron después del huracán
Harvey: asignar a **cada píxel** una de diez clases (agua, edificio inundado,
carretera anegada, vehículo, vegetación…). El objetivo práctico es delimitar la
extensión de la inundación y qué infraestructura ha quedado afectada.

Todo el trabajo vive en un único cuaderno, `floodnet-segmentation.ipynb`, pensado
para leerse de arriba abajo: monta el entorno, audita los datos, construye el
modelo, lo entrena y lo evalúa, explicando en cada paso por qué se hace así y no
de otra forma. Dos cuadernos más entrenan DeepLabV3+ y YOLO11n-seg sobre la misma
partición para comparar.

Los tres cuadernos entrenan sobre el dataset **FloodNet-Supervised v1.0 completo**
(2.343 imágenes con máscara, 2.247 tras la limpieza) y están ejecutados de principio
a fin, así que todas las cifras de aquí abajo se pueden contrastar con sus salidas. La
versión anterior, entrenada con las 398 imágenes etiquetadas de Track 1, se conserva en
la etiqueta `dataset-reducido`.

---

## Resultados

Conjunto de test: 426 imágenes de la partición `test` oficial que no se han visto en
ningún momento del entrenamiento. Se evalúa el checkpoint que mejor mIoU dio en
validación, con *test-time augmentation* por volteos.

| Métrica | 10 clases | Sin *Background* |
|---|---:|---:|
| mIoU | 0.732 | 0.754 |
| Dice | 0.841 | 0.857 |
| Pixel accuracy | 0.910 | — |

Desglose por clase:

| Clase | Dice | IoU |
|---|---:|---:|
| Grass | 0.939 | 0.884 |
| Road-non-flooded | 0.909 | 0.833 |
| Tree | 0.898 | 0.815 |
| Building-non-flooded | 0.892 | 0.806 |
| Water | 0.871 | 0.772 |
| Building-flooded | 0.829 | 0.707 |
| Pool | 0.810 | 0.680 |
| Road-flooded | 0.800 | 0.666 |
| Vehicle | 0.770 | 0.626 |
| Background | 0.689 | 0.526 |

Las dos tablas son la misma medición: el mIoU de 0.732 es exactamente la media
de la columna IoU, y 0.754 esa misma media dejando *Background* fuera. Ambas
salen de una única matriz de confusión acumulada sobre todo el test, no de
promediar resultados por lote.

Tres lecturas:

- **Las clases inundadas funcionan.** *Building-flooded* llega a 0.707 de IoU y
  *Road-flooded* a 0.666, que son las dos clases que de verdad importan para la
  tarea, pese a aparecer solo en ~150 de las 1.387 imágenes de entrenamiento.
- **El punto débil sigue siendo *Vehicle*** (0.626): los coches ocupan pocos píxeles a
  512×512 y se pierden en el reescalado. Aun así es la clase que más mejora frente a
  la versión anterior (0.573).
- **El test es ahora diez veces mayor** (426 imágenes frente a 41), así que las
  diferencias de un par de puntos entre modelos ya dicen algo.

### Frente a la versión con el dataset reducido

| Métrica | Dataset reducido (41 test) | Dataset completo (426 test) |
|---|---:|---:|
| mIoU (10 clases) | 0.714 | 0.732 |
| mIoU (sin *Background*) | 0.777 | 0.754 |
| Dice (10 clases) | 0.809 | 0.841 |
| Pixel accuracy | 0.927 | 0.910 |

Estas cifras **no se pueden comparar directamente**: aquel test eran 41 imágenes con
máscaras corregidas a mano y el nuevo son 426 con las máscaras oficiales, que tienen
errores de anotación (ver *Los datos*). Se nota en dos clases. *Background* sube de
0.144 a 0.526 de IoU, porque en las máscaras oficiales suele ser una región grande y
coherente y no un resto de pocos píxeles. *Water* baja de 0.915 a 0.772, en línea con
la confusión agua/hierba que encuentra la auditoría. Lo que sí es comparable es la
clasificación entre los tres modelos, porque los tres se miden igual.

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
| [floodnet-segmentation.ipynb](floodnet-segmentation.ipynb) | El cuaderno principal. Entorno, datos, modelo, entrenamiento y evaluación. |
| [floodnet-deeplabv3plus.ipynb](floodnet-deeplabv3plus.ipynb) | Comparación: DeepLabV3+ (ResNet-50) con el mismo split, exportado a ONNX. |
| [floodnet-yolo11n-seg.ipynb](floodnet-yolo11n-seg.ipynb) | Comparación: YOLO11n-seg con las máscaras convertidas en polígonos, mismo split, exportado a ONNX. |
| [original-hybrid-cnn-vit-attention-u-net.ipynb](original-hybrid-cnn-vit-attention-u-net.ipynb) | El cuaderno de Kaggle que se tomó como punto de partida. Se conserva para poder comparar. |
| [pyproject.toml](pyproject.toml) | Dependencias del proyecto, gestionadas con `uv`. |
| [uv.lock](uv.lock) | Versiones exactas resueltas, para reproducir el entorno bit a bit. |

El dataset, la caché de imágenes, los checkpoints entrenados (~350 MB cada uno) y los
modelos exportados (`models/`, `runs/`) quedan fuera del repositorio por tamaño; los excluye
[.gitignore](.gitignore).

---

## Los datos

Se usa **FloodNet-Supervised v1.0**, la versión del dataset que publica máscara para
todas las imágenes, más las máscaras corregidas a mano de un dataset de Kaggle:

| Dataset | Para qué |
|---|---|
| [FloodNet-Supervised v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | Las 2.343 imágenes con máscara y su reparto oficial train / val / test. El mapa índice → clase sale de su `DATASET-VERSION-NOTE-v1.0.txt`. |
| [FloodNet segmentation — fixed masks](https://www.kaggle.com/datasets/catashiro31/floodnet-dataset-for-segmentation-fixed-mask) | 398 máscaras corregidas a mano (las de Track 1, todas de la partición `train`). Sustituyen a la oficial cuando existen. Opcional: sin esta carpeta se usan solo las oficiales. |

Descomprímelos bajo `data/` con esta estructura:

```
data/
├── FloodNet/
│   └── FloodNet-Supervised_v1.0/
│       ├── DATASET-VERSION-NOTE-v1.0.txt
│       ├── train/  (train-org-img · train-label-img — 1.445 imágenes)
│       ├── val/    (val-org-img   · val-label-img   —   450 imágenes)
│       └── test/   (test-org-img  · test-label-img  —   448 imágenes)
├── data_finally/
│   ├── train/mask   ┐ 398 máscaras corregidas
│   └── val/mask     ┘
├── cache512_full/   ← lo genera el cuaderno, no lo descargues
└── yolo512_full/    ← lo genera el cuaderno de YOLO
```

La versión anterior usaba `data/FloodNet Challenge - Track 1/`, `data/cache512` y
`data/yolo512`. Los cuadernos nuevos ya no los leen ni los tocan.

### Las máscaras oficiales tienen errores

Hay dos tipos de fallo en las máscaras oficiales:

- **Máscaras de una sola clase.** Etiquetan los 12 megapíxeles enteros con una
  clase. La de `10165`, por ejemplo, marca toda la foto como *Water* y su PNG pesa
  15 KB, el tamaño al que comprime una imagen uniforme. Hay 84 en train, 16 en val y
  22 en test, casi todas *Grass*.
- **Regiones con la clase equivocada.** Hierba o árboles etiquetados como agua o
  como *Background*. En las 398 imágenes que tienen corrección, 138 máscaras
  oficiales difieren en más de un 5 % de sus píxeles de la versión corregida y 88 en
  más de un 10 %. Los cambios más frecuentes son *Water → Grass* y *Tree → Grass*.

Qué se hace con ello (sección 1 del cuaderno, con la auditoría en la 1.3):

1. Si una imagen tiene máscara corregida, se usa esa.
2. Las máscaras oficiales de una sola clase se excluyen de las tres particiones.
3. El resto de máscaras oficiales se usa tal cual: el segundo tipo de error no se
   puede detectar sin una referencia. Eso incluye val y test, así que **las métricas
   se miden contra referencias con algo de ruido**.

### La partición

Se respeta el reparto oficial y ninguna imagen cambia de partición:

| Partición | Oficial | Excluidas | Usadas |
|---|---:|---:|---:|
| train | 1.445 | 58 | 1.387 |
| val | 450 | 16 | 434 |
| test | 448 | 22 | 426 |

Los tres cuadernos comprueban con un `assert` estos tamaños y que ninguna imagen
aparece en dos particiones, y los de comparación guardan la lista exacta de imágenes
de cada partición en su JSON de configuración.

Frente a la versión anterior, que partía 81 imágenes por la mitad para tener val y
test, el test es ahora diez veces mayor: diferencias de uno o dos puntos entre modelos
empiezan a ser distinguibles del ruido.

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

La primera ejecución construye `data/cache512_full` (2.343 imágenes reducidas a
512×512, ~1,2 GB), que tarda del orden de 15-20 minutos. Los otros dos cuadernos la
reutilizan.

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
texturas que hace falta con un dataset de un millar largo de imágenes, mientras los
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
| Épocas | 30, con ~875 iteraciones cada una tras el oversampling |
| Oversampling | `target_ratio = 0.5`: cada clase rara hasta ~650 apariciones por época |
| Regularización | Early stopping sobre la pérdida de validación |
| Inferencia | TTA: promedio sobre la imagen y sus tres volteos |

Un par de decisiones que merecen explicación:

- **Oversampling con dos pipelines de augmentación.** Las imágenes que contienen
  clases raras se repiten dentro de cada época. Las copias reciben augmentación
  agresiva (deformación elástica, distorsión de rejilla, rotaciones amplias) y
  los originales una suave, de modo que las repeticiones aporten variedad real en
  vez de ser duplicados que invitan al sobreajuste. Con 317 imágenes se igualaban
  todas las clases (`target_ratio = 1.0`); con 1.387 eso daría épocas de ~8.700
  imágenes, así que se baja a 0.5 (~3.500 imágenes por época).
- **Caché a 512×512.** Los originales son JPEG de 4000×3000. Decodificar 12
  megapíxeles en cada época convertiría la CPU en el cuello de botella. Se
  reescalan una sola vez: la imagen con `INTER_AREA` y la máscara con
  `INTER_NEAREST`, porque interpolar suavemente una máscara inventaría clases
  que no existen (el punto medio entre la clase 2 y la 4 sería un 3, que es otra
  clase distinta, no una mezcla).

Mejores cifras de validación (434 imágenes), alcanzadas en la última época: mIoU
0.751, Dice 0.854, accuracy 0.909.

En una RTX 4060 Laptop cada época son 874 iteraciones y unos 4 minutos de
entrenamiento, más la validación con TTA. Las 30 épocas llevaron unas 2 h 15 min.

Los checkpoints se guardan en `models/` con el sufijo `_full`
(`best_model_iou_full.pth`, `best_vit_unet_dice_full.pth`,
`best_model_loss_full.pth`), para no pisar los del entrenamiento anterior.

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

- **Las referencias de val y test tienen ruido.** Salvo las de una sola clase, que
  se excluyen, las máscaras oficiales de val y test se usan tal cual, y por lo visto
  en las 398 corregidas una parte relevante tiene regiones mal etiquetadas. Las
  métricas absolutas quedan por debajo de lo que mediría una referencia limpia.
- **La pérdida de validación no es comparable con la de entrenamiento.**
  `val_loss_fn` recibe las probabilidades promediadas por el TTA y se las pasa a
  `nn.CrossEntropyLoss`, que espera logits. La curva sirve para ver la tendencia
  y para el early stopping (que en esta ejecución nunca llegó a dispararse), pero su
  valor absoluto está en otra escala que el de entrenamiento. Las métricas reportadas
  no dependen de esto.
- **No se fija semilla en PyTorch** en el cuaderno principal: la inicialización de
  pesos y el orden de los lotes cambian entre ejecuciones, así que dos
  entrenamientos no dan pesos idénticos. El split es el oficial y no depende de
  ninguna semilla.

---

## Modelos de comparación

Dos cuadernos independientes entrenan modelos estándar con **exactamente la misma partición**
(1.387 / 434 / 426, comprobada con `assert`) y a la misma resolución, 512×512, reutilizando
`data/cache512_full`. Los dos se evalúan con la misma `SegMetrics` y dejan su modelo en ONNX para
medir la inferencia en el mismo equipo.

Ajustes al pasar al dataset completo:

- **DeepLabV3+**: 40 épocas en lugar de 80. Cada época tiene ahora 173 lotes en vez de 39, así que
  el entrenamiento hace más del doble de pasos que antes.
- **YOLO11n-seg**: se mantienen las 300 épocas como techo, con `patience = 50` en lugar de 100. La
  conversión a polígonos reconstruye las 2.247 máscaras usadas con un mIoU de 0.984.

| | U-Net híbrido | DeepLabV3+ R50 | YOLO11n-seg |
|---|---:|---:|---:|
| mIoU test (10 clases) | **0.732** | 0.709 | 0.705 |
| mIoU test (sin *Background*) | **0.754** | 0.738 | 0.739 |
| Dice test (10 clases) | **0.841** | 0.823 | 0.818 |
| Pixel accuracy | **0.910** | 0.908 | 0.890 |
| Píxeles sin predicción | 0 % | 0 % | 3.0 % |
| Parámetros | — | 26.7 M | 2.9 M |
| TTA | sí | no | no |
| Entrenamiento (RTX 4060 Laptop) | ~2 h 15 min, 30 épocas | 47 min, 40 épocas | 2.6 h, 235 épocas |
| Post-proceso fuera del ONNX | — | `argmax` | NMS + máscaras + repintado |

| Clase | IoU U-Net | IoU DeepLabV3+ | IoU YOLO11n-seg |
|---|---:|---:|---:|
| Grass | 0.884 | **0.885** | 0.871 |
| Road-non-flooded | **0.833** | 0.831 | 0.809 |
| Tree | 0.815 | **0.831** | **0.831** |
| Building-non-flooded | **0.806** | 0.800 | 0.794 |
| Water | **0.772** | 0.757 | 0.767 |
| Building-flooded | 0.707 | 0.711 | **0.727** |
| Pool | **0.680** | 0.647 | 0.651 |
| Road-flooded | **0.666** | 0.612 | 0.639 |
| Vehicle | **0.626** | 0.570 | 0.566 |
| Background | **0.526** | 0.450 | 0.394 |

- **DeepLabV3+** es un *baseline* semántico limpio (CE ponderada + Dice, sin oversampling ni TTA).
  Queda a ~2 puntos del U-Net, que le saca la mayor ventaja en *Background*, *Vehicle*,
  *Road-flooded* y *Pool*.
  Su mejor época fue la 32 de 40 y el mIoU de val estaba estancado en ~0.73.
- **YOLO11n-seg** es de instancias, y FloodNet no tiene instancias sino regiones. Para entrenarlo,
  cada componente conexa de cada clase se trata como un objeto y su contorno se convierte en
  polígono. Con el dataset completo **alcanza a DeepLabV3+ sin *Background*** (0.739 frente a 0.738)
  con 9 veces menos parámetros, y es el mejor de los tres en *Building-flooded*. El umbral de
  confianza elegido en val baja a 0.25 y los píxeles sin cubrir, del 4.7 % al 3.0 %. Paró en la
  época 235 por *early stopping*.
- Los dos ONNX reproducen a su modelo de PyTorch: la diferencia de mIoU en test es de 0.00002
  (DeepLab) y 0.0003 (YOLO).

Con el dataset reducido (317 / 40 / 41) las cifras eran 0.714 / 0.696 / 0.678 de mIoU para U-Net,
DeepLabV3+ y YOLO, en el mismo orden que ahora.

Cada cuaderno tiene una celda **P** con todos los parámetros. Esa celda se vuelca, junto con la lista
de imágenes de cada partición, el contrato de entrada/salida del ONNX y los resultados, a
`models/<modelo>_full_config.json`. Los dos ONNX esperan la misma entrada: `images`, float32 RGB en
[0, 1], 1×3×512×512.

| Fichero | Salidas |
|---|---|
| `models/deeplabv3plus_r50_floodnet_full.onnx` | `logits` 1×10×512×512 |
| `models/yolo11n_seg_floodnet_full.onnx` | `output0` 1×46×5376 (cajas, clases, coeficientes) · `output1` 1×32×128×128 (prototipos) |

Todos los artefactos del dataset completo llevan el sufijo `_full` (y `runs/yolo/yolo11n-seg-full`
para YOLO), de modo que los del entrenamiento anterior siguen en su sitio.

`onnxruntime-gpu` está fijado a la 1.24 en `pyproject.toml`: las ruedas recientes de PyPI se
compilan contra CUDA 13 y no cargan junto a PyTorch cu126.

---

## Por dónde seguir

- Entrenar el U-Net más épocas. La mejor época de validación fue la última (la 30) y
  el mIoU de val seguía subiendo despacio; con `CosineAnnealingWarmRestarts`
  (T₀ = 10, T_mult = 2) el siguiente final de ciclo es la época 70.
- Limpiar las máscaras oficiales de val y test, aunque sea en una muestra, para tener
  una referencia sin ruido con la que calibrar cuánto penaliza el ruido actual.
- Probar resolución 768 o inferencia por tiles. *Vehicle* es la clase que más lo
  pediría: a 512 píxeles un coche ocupa muy pocos píxeles y su IoU lo refleja.
- Tratar *Background* como `ignore_index` también en la pérdida, no solo al
  reportar, y ver si el resto de clases gana algo.

---

## Créditos

El dataset procede del [FloodNet Challenge](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
(Rahnemoonfar et al.), imágenes de dron posteriores al huracán Harvey. El punto
de partida del modelo es el cuaderno de Kaggle conservado en este repositorio.
