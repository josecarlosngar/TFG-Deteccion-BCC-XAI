# Detección explicable de carcinoma basocelular (BCC) con VGG16 y Grad-CAM

Trabajo Fin de Grado — Grado en Ingeniería de las Tecnologías de las Telecomunicaciones,
Escuela Técnica Superior de Ingeniería, Universidad de Sevilla.

**Título completo:** Algoritmo de inteligencia artificial explicada para la detección
del carcinoma basocelular.

**Autor:** José Carlos Navarro García
**Tutoras:** Begoña Acha Piñero, María del Carmen Serrano Gotarredona

---

## Descripción

El carcinoma basocelular (BCC) es el tipo de cáncer de piel más frecuente, y se
confunde con facilidad con lesiones benignas, lo que provoca biopsias innecesarias.
Este proyecto entrena una red neuronal convolucional (VGG16 con *fine-tuning*) para
detectar, en imágenes dermatoscópicas, los siete patrones dermatoscópicos descritos
por Menzies et al. (2000) que permiten diagnosticar un BCC, y utiliza **Grad-CAM**
para generar mapas de calor que muestran en qué zona de la imagen se ha basado la red
para cada patrón — de forma que la predicción no sea una caja negra, sino una
herramienta auditable visualmente.

## Arquitectura

- **Extractor de características:** VGG16 preentrenada en ImageNet, con las últimas 5
  capas convolucionales (`block4_pool` a `block5_pool`) descongeladas y reentrenadas
  (*fine-tuning*); el resto de la red se mantiene congelada.
- **Cabeza de clasificación:** `GlobalAveragePooling2D` → `Dense(512, activación
  ReLU)` → `Dropout(0.15)` → `Dense(7, activación sigmoide)`.
- **Salida multietiqueta:** 7 probabilidades independientes (activación sigmoide, no
  softmax), una por patrón, ya que un mismo caso puede presentar varios patrones a la
  vez.
- **Explicabilidad:** Grad-CAM aplicado sobre el modelo ya entrenado, con la capa
  configurable (`block3_pool`, `block4_pool` o `block5_pool`, por defecto esta
  última).

## Los 7 patrones dermatoscópicos

`PigmentNetwork`, `Ulceration`, `Large_B_G_OvoidNests`, `Multi_B_G_Globules`,
`MapleLeaflike`, `SpokeWheel`, `ArborizingTelangiectasia`.

El criterio clínico implementado (Menzies et al., 2000) clasifica una lesión como BCC
si **no** presenta el patrón `PigmentNetwork` (retículo pigmentado, típico de
lesiones melanocíticas) **y** presenta **al menos uno** de los otros seis patrones.

## Resultados

Evaluado sobre un conjunto externo e independiente (ISIC 2018 Challenge, 50 imágenes
BCC + 50 No-BCC, nunca visto durante el entrenamiento):

| Métrica | Valor |
|---|---|
| Exactitud (accuracy) | 94 % |
| Sensibilidad | 98 % |
| Especificidad | 90 % |
| Precisión (VPP) | 91 % |

Resultados por patrón sobre el conjunto de test interno:

| Patrón | Sensibilidad | Especificidad |
|---|---|---|
| Retículo pigmentado | 65 % | ~100 % |
| Ulceración | 95 % | 87 % |
| Nidos ovoides azul-gris | 68 % | 97 % |
| Múltiples glóbulos azul-gris | 86 % | 90 % |
| Hojas de arce | 71 % | 98 % |
| Rueda de carro | 78 % | 99 % |
| Telangiectasias ramificadas | 84 % | 93 % |

El análisis con Grad-CAM confirma que, en la mayoría de los casos, la activación
coincide espacialmente con la estructura dermatoscópica real, y permite además
detectar limitaciones que las métricas por sí solas no revelan (por ejemplo, casos en
los que la red confunde vello corporal con el patrón reticular).

## Datos

Este repositorio **no incluye las imágenes** utilizadas (proceden en parte del ISIC
2018 Challenge y de un conjunto de imágenes propio recopilado para el TFG, sujetas a
sus propios términos de uso). Para ejecutar el notebook, se espera la siguiente
estructura de carpetas dentro de la ruta de datos (`RUTA_DATOS` en la celda de
configuración):

```
RUTA_DATOS/
├── datos_aumentados/        # imágenes de entrenamiento, con el/los patrón(es) en el nombre de fichero
├── datos_test_reticulo/     # imágenes reales con patrón reticular
├── datos_test_sinPatron/    # imágenes reales sin ningún patrón de BCC
├── BCC_ISIC/                # conjunto externo de evaluación, BCC confirmado
└── NoBCC_ISIC/               # conjunto externo de evaluación, No-BCC confirmado
```

## Cómo ejecutar

El notebook (`notebook/TFG_JCNG_ultimas5capas.ipynb`) detecta automáticamente si se
ejecuta en Google Colab, Kaggle Notebooks o en local, y ajusta las rutas en
consecuencia. Se recomienda ejecutarlo con GPU (imprescindible para un tiempo de
entrenamiento razonable, ya que las capas descongeladas de VGG16 no permiten
precalcular características).

```bash
pip install -r requirements.txt
jupyter notebook notebook/TFG_JCNG_ultimas5capas.ipynb
```

El notebook está dividido en tres partes independientes:

- **Parte A** — Entrenar el modelo desde cero.
- **Parte B** — Cargar un modelo ya entrenado y evaluar (sensibilidad, especificidad,
  matrices de confusión).
- **Parte C** — Generar mapas de calor Grad-CAM sobre cualquier imagen y patrón.

## Referencias principales

- Menzies, S. W. et al. (2000). *Surface microscopy of pigmented basal cell
  carcinoma*.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization*.
- Simonyan, K. y Zisserman, A. (2015). *Very Deep Convolutional Networks for
  Large-Scale Image Recognition* (VGG16).
- Serrano, C. et al. (2022). Artículo de referencia de las tutoras del TFG sobre
  detección de BCC combinando color, textura y aprendizaje profundo.

## Licencia

Código publicado bajo licencia MIT (ver [`LICENSE`](LICENSE)). No incluye ningún
dato ni imagen médica.
