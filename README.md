# Fuga de clientes Telco: del dataset al despliegue

Construimos una solución completa de Machine Learning que estima la probabilidad de que un cliente de una compañía de telecomunicaciones cancele el servicio (churn). El proyecto cubre todo el ciclo: datos, entrenamiento reproducible con `train.py`, interfaz en Streamlit, imagen Docker y despliegue en una máquina virtual de Google Cloud.

**Objetivo de negocio:** identificar a tiempo a los clientes con alto riesgo de irse para priorizarlos en una campaña de retención.

**Autores:** Miguel Melo y David Castro

## Ruta elegida

**Ruta A: aprendizaje supervisado.** Es una clasificación binaria: la variable objetivo `Churn` vale `Yes` si el cliente se fue y `No` si se quedó.

## Dataset

- **Fuente:** IBM Telco Customer Churn, de uso público ([repositorio de IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)). Una copia está en `data/telco_churn.csv`, así que no hace falta descargar nada.
- **Unidad de observación:** cada fila es un cliente.
- **Tamaño:** 7.043 clientes y 21 columnas.
- **Objetivo:** `Churn`. El 26,5 % de los clientes se fue (1.869 de 7.043), así que las clases están desbalanceadas.
- **Datos faltantes:** solo `TotalCharges` tiene 11 vacíos. Vienen como texto en blanco y corresponden exactamente a los clientes con `tenure = 0`.

## Variables utilizadas

Usamos 17 variables que se conocen de cualquier cliente activo:

| Tipo | Variables |
|---|---|
| Numéricas (2) | `tenure` (meses de antigüedad), `MonthlyCharges` (cargo mensual en USD) |
| Perfil (3) | `SeniorCitizen`, `Partner`, `Dependents` |
| Contrato y facturación (3) | `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Servicios (9) | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

Excluimos tres columnas a propósito:

- **`customerID`:** es un identificador y no describe al cliente.
- **`TotalCharges`:** es casi el producto de antigüedad por cargo mensual (correlación de 0,83 con `tenure`). Pedirlo en la app sería redundante y dejaría ingresar combinaciones incoherentes.
- **`gender`:** no aporta señal (26,9 % de fuga en mujeres frente a 26,2 % en hombres) y es un atributo sensible.

## Preparación de datos

Todo el preprocesamiento vive dentro de un `Pipeline` de scikit-learn, así la app aplica exactamente las mismas transformaciones que el entrenamiento:

- **Numéricas:** imputación con la mediana y estandarización (`StandardScaler`).
- **Categóricas:** imputación con la moda y codificación one-hot (`OneHotEncoder(handle_unknown="ignore")`).
- `SeniorCitizen` viene como 0/1 y lo convertimos a `No`/`Yes` para tratarlo igual que las demás categóricas.

Partimos los datos en 80 % entrenamiento (5.634 clientes) y 20 % prueba (1.409), estratificando por `Churn` y con `random_state=42`.

## Modelo y justificación

Comparamos dos modelos con validación cruzada estratificada de 5 folds, usando solo el conjunto de entrenamiento. Los dos usan `class_weight="balanced"` para compensar el desbalance:

| Modelo | F1 de fuga en CV |
|---|---|
| Regresión logística | 0,628 ± 0,022 |
| Random forest (300 árboles, `min_samples_leaf=5`) | 0,631 ± 0,013 |

El random forest gana por 0,003, una diferencia mucho menor que la variación entre folds. Por eso aplicamos una regla de parsimonia: si el modelo más simple queda a menos de una desviación estándar del mejor, nos quedamos con el simple. **Elegimos la regresión logística**: rinde igual, se interpreta directamente con sus coeficientes y el artefacto pesa 10 KB en lugar de 16 MB.

## Métricas y evaluación

Evaluamos el modelo elegido **una sola vez** sobre el conjunto de prueba:

| Métrica | Valor |
|---|---|
| Recall de fuga | 0,78 |
| Precisión de fuga | 0,51 |
| F1 de fuga | 0,62 |
| ROC-AUC | 0,839 |
| Accuracy | 0,74 |

Matriz de confusión (filas: real, columnas: predicho):

| | Predicho: se queda | Predicho: se va |
|---|---|---|
| **Real: se queda** | 756 | 279 |
| **Real: se va** | 83 | 291 |

**Por qué estas métricas y no solo accuracy:** como solo el 26,5 % se va, un modelo que dijera siempre "se queda" tendría 73,5 % de accuracy sin detectar a nadie. Para retención importa más no dejar escapar a quien se va (recall) que evitar algunas falsas alarmas, porque contactar a un cliente que no pensaba irse cuesta menos que perder uno. Por eso elegimos el modelo con F1 de la clase positiva y reportamos recall y ROC-AUC.

**Qué aprendió el modelo:** los coeficientes más grandes empujan hacia la fuga cuando el cliente tiene fibra óptica, contrato mes a mes, paga con cheque electrónico o tiene servicios de streaming. Empujan hacia quedarse un contrato de dos años, más antigüedad e internet DSL. Esto coincide con lo que se ve en las tablas cruzadas del notebook.

Las métricas exactas quedan en `metrics.json` cada vez que se ejecuta `train.py`.

## Interpretación de la salida

La app muestra:

- **Clase predicha:** "Se va" si la probabilidad de fuga es de 50 % o más, y "Se queda" en otro caso.
- **Probabilidad de fuga:** la estimación del modelo para un cliente con ese perfil. Sirve para ordenar a los clientes por riesgo, no como certeza individual.

**Ojo con la escala.** Como entrenamos con `class_weight="balanced"`, el modelo da más peso a los clientes que se van y sus probabilidades salen infladas: el promedio predicho es 41,5 % cuando la tasa real es 26,5 %. En los clientes de prueba con probabilidad entre 70 % y 80 %, en realidad se fue el 47 %. El orden sí es confiable (ROC-AUC 0,839), así que la usamos para priorizar clientes. La versión web presenta el número como "puntaje de riesgo" y al lado muestra la tasa real de fuga observada para ese tramo.

La app solo ofrece combinaciones que existen en los datos. Si el cliente no tiene teléfono, "Varias líneas" queda fijo en "Sin servicio telefónico". Si no tiene internet, los seis servicios de internet quedan fijos en "Sin servicio de internet". Los valores por defecto del formulario son los más frecuentes del dataset.

## Estructura del proyecto

```
telco-churn-streamlit/
├── app.py                  # interfaz Streamlit (solo carga el modelo)
├── train.py                # entrenamiento reproducible
├── export_web.py           # exporta el modelo a web/model.json
├── model.joblib            # pipeline entrenado + metadatos para la app
├── metrics.json            # métricas de la última ejecución de train.py
├── requirements.txt        # dependencias con versiones fijas
├── requirements-dev.txt    # requirements.txt + pytest
├── Dockerfile
├── .dockerignore
├── data/
│   └── telco_churn.csv
├── tests/
│   ├── test_app.py         # pruebas de la app con streamlit.testing
│   ├── test_web_export.py  # web/model.json coincide con model.joblib
│   ├── check_web.mjs       # la web da la misma probabilidad que sklearn
│   └── web_cases.json      # 300 casos de verificación
├── web/                    # versión web estática (Vercel)
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   ├── churn-model.js      # inferencia en JavaScript
│   └── model.json          # coeficientes exportados
└── colab/
    └── telco_churn_colab.ipynb   # revisión del dataset y pruebas en Colab
```

Fijamos las versiones de `requirements.txt` a propósito: `model.joblib` solo carga de forma confiable con la misma versión de scikit-learn con la que lo entrenamos (1.9.1).

## Cómo ejecutar `train.py`

Necesitamos Python 3.12.

```bash
git clone https://github.com/melo11miguel/telco-churn-streamlit.git
cd telco-churn-streamlit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py
```

Debe imprimir la validación cruzada, el modelo elegido, la matriz de confusión y terminar con `Modelo guardado en model.joblib`. Con `random_state=42` las métricas salen idénticas en cada ejecución.

Para correr las pruebas (cargan la app, llenan el formulario y comparan el resultado con el pipeline):

```bash
python -m pytest -q
```

## Cómo ejecutar Streamlit

```bash
streamlit run app.py
```

Y abrir http://localhost:8501. La app no vuelve a entrenar: solo carga `model.joblib`.

**En Google Colab:** abrir `colab/telco_churn_colab.ipynb` (en Colab: Archivo, Abrir notebook, GitHub, pegar la URL del repositorio) y ejecutar las celdas en orden. El notebook clona el repo, revisa el dataset, entrena, corre las pruebas y levanta Streamlit con un túnel temporal de localtunnel.

## Cómo construir la imagen Docker

Desde la carpeta del proyecto:

```bash
docker build -t ml-streamlit-app .
docker images
```

El punto final indica que el contexto de construcción es la carpeta actual. La imagen parte de `python:3.12-slim`, instala `requirements.txt` y copia el proyecto (sin lo que excluye `.dockerignore`).

## Cómo ejecutar el contenedor

Primera prueba interactiva (detener con Ctrl+C):

```bash
docker run --rm -p 8501:8501 ml-streamlit-app
```

En segundo plano y con reinicio automático:

```bash
docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
docker ps
docker logs ml-streamlit
```

`-p 8501:8501` conecta el puerto 8501 de la máquina con el del contenedor. Streamlit escucha en `0.0.0.0` para aceptar conexiones externas. La imagen incluye un `HEALTHCHECK`, así que después de unos 20 segundos `docker ps` debe mostrar el estado `(healthy)`.

## Despliegue en Google Cloud (Compute Engine)

1. En Google Cloud Console, seleccionar el proyecto, abrir **Compute Engine** y habilitar la API si la consola lo pide.
2. Crear una instancia de VM:
   - Imagen: **Ubuntu 24.04 LTS** (x86/64).
   - Tipo de máquina: `e2-small` (2 GB de RAM) es suficiente.
   - Con dirección IP externa.
   - En **Redes**, agregar la etiqueta de red `streamlit-server`.
3. Abrir la terminal con el botón **SSH** e instalar Docker y Git:

   ```bash
   sudo apt update
   sudo apt install -y docker.io git
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```

   Cerrar la ventana SSH y volver a entrar para usar Docker sin `sudo`.
4. Clonar, construir y ejecutar:

   ```bash
   git clone https://github.com/melo11miguel/telco-churn-streamlit.git
   cd telco-churn-streamlit
   ls -la
   docker build -t ml-streamlit-app .
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   docker ps
   ```

5. Abrir el puerto 8501 en el firewall. Desde Cloud Shell:

   ```bash
   gcloud compute firewall-rules create allow-streamlit-8501 \
     --allow=tcp:8501 --direction=INGRESS --target-tags=streamlit-server
   ```

   O desde la consola: **Red de VPC, Firewall, Crear regla**, con destino la etiqueta `streamlit-server`, rango de origen `0.0.0.0/0` y TCP 8501.
6. Abrir `http://IP_EXTERNA_DE_LA_VM:8501` en el navegador.

Este acceso público es solo para el laboratorio. En producción haría falta HTTPS, autenticación y restringir los orígenes.

**Al terminar, detener o eliminar la VM** para que no siga generando costos.

### Actualizar la aplicación desplegada

El contenedor no se actualiza solo: hay que construir una imagen nueva y crear un contenedor nuevo.

```bash
git pull
docker stop ml-streamlit
docker rm ml-streamlit
docker build -t ml-streamlit-app .
docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
```

### Diagnóstico rápido

| Situación | Qué revisar |
|---|---|
| `docker build` falla | Leer el error completo; confirmar que `requirements.txt` y `model.joblib` están en la carpeta. |
| El contenedor se detiene | `docker ps -a` y `docker logs ml-streamlit`. |
| Funciona en la VM pero no desde fuera | La regla de firewall, la etiqueta `streamlit-server` en la VM y la IP externa. Desde la VM, `curl localhost:8501/_stcore/health` debe responder `ok`. |
| Los cambios no aparecen | Seguir los pasos de actualización de arriba. |

## Versión web en Vercel

Además de la app de Streamlit (la que pide la guía, con Docker en GCP), publicamos una versión web con un diseño pensado para alguien de negocio. Streamlit no puede correr en Vercel, porque necesita un servidor Python encendido todo el tiempo. Por eso aprovechamos que el modelo es una regresión logística: su predicción es una suma ponderada que el navegador calcula directamente con los coeficientes del pipeline entrenado. No hay servidor y ningún dato que se ingrese sale del equipo de quien usa la página.

La página muestra:

- **Puntaje de riesgo** de 0 a 100, con nivel bajo (menos de 30), medio (de 30 a 50) o alto (50 o más, el mismo umbral de la app de Streamlit).
- **Tasa real de fuga** de los clientes de prueba con un puntaje parecido.
- **Qué explica el puntaje:** los cinco factores que más lo suben o lo bajan frente al cliente promedio, en puntos.
- **Qué podría retenerlo:** cuánto bajaría el puntaje al cambiar una sola cosa (contrato, forma de pago o servicios de soporte). Son asociaciones del modelo, no efectos causales.
- **Confiabilidad:** métricas de prueba en lenguaje sencillo y el gráfico de puntaje frente a fuga real, con su tabla.

Si reentrenamos el modelo, hay que regenerar y verificar la versión web:

```bash
python train.py
python export_web.py
python -m pytest -q          # incluye la prueba de que web/model.json está al día
node tests/check_web.mjs     # la web y sklearn dan la misma probabilidad
```

`export_web.py` se niega a exportar si `train.py` llegara a elegir el random forest, porque la web solo reproduce la regresión logística. Para verla en local: `python -m http.server 8000 --directory web` y abrir http://localhost:8000.

En Vercel el proyecto apunta a la carpeta `web/` como directorio raíz, sin paso de construcción. Cada push a `main` la vuelve a publicar.

## Ejemplo de entrada y salida

Valores obtenidos con la app y el modelo de este repositorio. Los campos que no aparecen quedan en su valor por defecto (el más frecuente del dataset).

**Caso 1: cliente nuevo de alto riesgo**

| Campo en la app | Valor |
|---|---|
| Antigüedad (meses) | 1 |
| Cargo mensual (USD) | 95 |
| Tipo de contrato | Mes a mes |
| Servicio de internet | Fibra óptica |
| Método de pago | Cheque electrónico |
| Seguridad en línea / Soporte técnico | No / No |
| Factura electrónica | Sí |

Salida esperada: **Clase predicha: Se va. Probabilidad de fuga: 79,2 %.**

**Caso 2: cliente antiguo de bajo riesgo**

| Campo en la app | Valor |
|---|---|
| Antigüedad (meses) | 70 |
| Cargo mensual (USD) | 25 |
| Tipo de contrato | Dos años |
| Servicio de internet | No (los seis servicios de internet quedan en "Sin servicio de internet") |
| Método de pago | Transferencia bancaria (automática) |
| Factura electrónica | No |

Salida esperada: **Clase predicha: Se queda. Probabilidad de fuga: 1,3 %.**

**Caso 3: formulario sin tocar** (antigüedad 29, cargo 70,35, mes a mes, fibra óptica, cheque electrónico). Salida esperada: **Se va, 72,1 %.** Tiene sentido: el perfil más frecuente del dataset combina justo los factores de mayor riesgo.

## Prueba cruzada

Otra pareja debe poder reproducir el proyecto siguiendo solo este README, sin instrucciones verbales.

| Paso | Resultado | Observaciones |
|---|---|---|
| Clonar el repositorio | | |
| `docker build` | | |
| `docker run` | | |
| Probar el caso 1 del ejemplo | | |
| Instrucciones faltantes o errores | | |

Pareja que hizo la prueba: ____________ Fecha: ____________
