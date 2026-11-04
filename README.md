# Sistema de Reconocimiento Facial — TUIA

**Trabajo Práctico — Computer Vision**

Tecnicatura Universitaria en Inteligencia Artificial  
Facultad de Ciencias Exactas, Ingeniería y Agrimensura — Universidad Nacional de Rosario

## Integrantes

- Uriel Duvia
- Laureano Diez

## Descripción del proyecto

El objetivo de este trabajo práctico es desarrollar un sistema de reconocimiento facial capaz de registrar identidades y reconocer personas en imágenes nuevas.

Para ello, se implementó un pipeline que comprende el procesamiento de imágenes faciales, el entrenamiento de una red neuronal convolucional propia, la comparación con un modelo preentrenado y la integración del modelo seleccionado en una aplicación web.

El sistema permite:

- Detectar rostros presentes en una imagen.
- Alinear los rostros detectados.
- Extraer embeddings faciales de 512 dimensiones.
- Registrar identidades en una base de datos.
- Comparar embeddings mediante similitud coseno.
- Reconocer personas registradas.
- Identificar personas desconocidas mediante un umbral de similitud.

## Tecnologías utilizadas

- **Python:** procesamiento e implementación de modelos.
- **PyTorch:** entrenamiento de la CNN propia.
- **ONNX Runtime:** inferencia de modelos exportados.
- **OpenCV:** procesamiento de imágenes.
- **InsightFace / ArcFace:** modelo preentrenado de reconocimiento facial.
- **FastAPI:** backend y endpoints REST.
- **Gradio:** interfaz web.
- **PostgreSQL + pgvector:** almacenamiento y consulta de embeddings.
- **Docker Compose:** ejecución de los servicios.

## Estructura del trabajo

El proyecto se divide en tres etapas.

### Parte 1 — Dataset y preprocesamiento

Notebook: `1-pre-processing.ipynb`

Esta etapa contempla el análisis exploratorio del dataset, control de calidad, detección y alineación facial, clustering, data augmentation y partición de los datos.

**Dataset utilizado:** [Completar nombre, fuente, cantidad de identidades e imágenes]

**Enlace al dataset o a los datos procesados:** [Completar si corresponde]

### Parte 2 — Entrenamiento y comparación

Notebook: `2-train.ipynb`

Se desarrolló una CNN residual propia para obtener embeddings faciales y se comparó con InsightFace (ArcFace, w600k_r50).

Se realizaron tres experimentos con la CNN:

| Experimento | Configuración | Parámetros | EER en val_open |
|---|---|---:|---:|
| A1 | Softmax + GAP | 1,36 M | 0,2773 |
| A2 | ArcFace + GAP | 1,36 M | 0,2682 |
| A3 | ArcFace + Flatten | 7,65 M | 0,2283 |

El experimento A3 obtuvo el menor EER de validación y fue seleccionado como la mejor configuración de la CNN propia.

### Comparación final de modelos

Los modelos se evaluaron sobre identidades no utilizadas durante el entrenamiento.

| Métrica | CNN propia (A3) | InsightFace |
|---|---:|---:|
| AUC | 0,8746 | 1,0000 |
| EER | 0,2126 | 0,0017 |
| TAR @ FAR=1e-3 | 0,1136 | 0,9975 |
| Top-1 accuracy | 0,7131 | 1,0000 |
| Precision macro | 0,4249 | 0,9687 |
| Recall macro | 0,6363 | 0,9958 |
| F1 macro | 0,4391 | 0,9813 |
| Desconocidos rechazados | 0,0414 | 0,9619 |
| Inferencia CPU (ms/rostro) | 7,61 | 141,81 |
| Tamaño (MB) | 30,61 | 174,38 |

**Modelo seleccionado: InsightFace w600k_r50.**

InsightFace obtuvo mejores resultados de verificación e identificación, especialmente en el reconocimiento de personas desconocidas.

La CNN propia demostró ventajas de velocidad y tamaño, aunque presentó mayores dificultades para distinguir identidades no registradas.

### Parte 3 — Integración en la aplicación

Archivo principal: `src/lib/services/face_service.py`

La aplicación integra tres operaciones fundamentales:

1. `detect_faces()`: detecta los rostros presentes en una imagen.
2. `align_face()`: obtiene los rostros alineados.
3. `extract_embedding_from_face()`: genera embeddings faciales para su comparación.

Los embeddings de las personas registradas se almacenan mediante PostgreSQL con pgvector.

El frontend permite registrar identidades y reconocer personas en imágenes nuevas.

## Modelos y descarga

Los archivos de pesos `.onnx` y `.pth` no se incluyen en el repositorio.

**Enlace de descarga:** https://drive.google.com/drive/folders/11dP6O0u9go8eXXwyO7UdAEzoUM_bPg1x?usp=sharing
Archivos de reconocimiento y detección necesarios para ejecutar la aplicación:

- `w600k_r50.onnx`: modelo de reconocimiento facial.
- `det_10g.onnx` : modelo de deteccion. 

Se recomienda conservar también el modelo de la CNN propia exportado como `custom_cnn.onnx`, para reproducir las comparaciones.

Los modelos deben colocarse en la carpeta `models/`, respetando los nombres esperados por la configuración.

Los modelos preentrenados de InsightFace están sujetos a sus condiciones de licencia para investigación y uso no comercial.

## Ejecución del proyecto

### Requisitos

- Git
- Docker Desktop
- Docker Compose

### 1. Clonar el repositorio

```bash
git clone https://github.com/laureanodiez/tuia-face-recognition-app.git
cd tuia-face-recognition-app
```

### 2. Descargar los modelos

Descargar los archivos desde el enlace de Google Drive y colocarlos dentro de `models/`.

### 3. Configurar las variables de entorno

Crear el archivo `.env` a partir del archivo de ejemplo correspondiente a Docker y revisar las variables de configuración.

Para el modelo seleccionado en la notebook:

```ini
MODEL_NAME=w600k_r50.onnx
EMBEDDING_DIM=512
FACE_SIZE=112
SIMILARITY_METRIC=cosine
SIMILARITY_THRESHOLD=0.220
```

También se deben configurar las rutas del detector y la conexión a PostgreSQL según los archivos de ejemplo del proyecto.

### 4. Construir y levantar los contenedores

```bash
docker compose up -d --build
```

### 5. Acceder a la aplicación

Una vez iniciados los servicios:

- **Frontend:** http://127.0.0.1:8080
- **Backend (Swagger):** http://127.0.0.1:8000/docs
- **Health:** http://127.0.0.1:8000/health

### 6. Verificar los contenedores

```bash
docker compose ps
```

Para consultar los logs del backend:

```bash
docker compose logs backend --tail=100
```

## Pruebas funcionales

Se comprobó el funcionamiento de la aplicación mediante el registro de varias imágenes de una identidad y su posterior reconocimiento en una fotografía diferente.

Para reproducir la validación completa se recomienda probar:

- Registro de varias identidades.
- Reconocimiento con imágenes diferentes de las utilizadas durante el registro.
- Identificación de múltiples rostros.
- Rechazo de identidades desconocidas.
- Imágenes sin rostros y registros con más de un rostro.

## Limitaciones y posibles mejoras

La CNN propia presenta limitaciones para identificar correctamente personas desconocidas. En la evaluación open-set rechazó correctamente el 4,14 % de las muestras desconocidas.

Entre las posibles mejoras se encuentran aumentar la diversidad de los datos de entrenamiento, explorar nuevas arquitecturas y evaluar distintos criterios de calibración del umbral de similitud.

Los resultados reportados corresponden al dataset y protocolo utilizados en este trabajo y no garantizan el mismo rendimiento en otros escenarios.

## Consideraciones de privacidad

Las imágenes faciales y sus embeddings constituyen información biométrica sensible.

Se recomienda utilizar imágenes con consentimiento de sus titulares y evitar publicar datos personales, fotografías de terceros, bases de datos de rostros o archivos de configuración que contengan secretos.

## Referencias

- [InsightFace](https://github.com/deepinsight/insightface)
- [ArcFace — Deng et al., 2019](https://arxiv.org/abs/1801.07698)
- [PyTorch](https://pytorch.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [pgvector](https://github.com/pgvector/pgvector)
