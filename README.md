# Fuga de clientes Telco: del dataset al despliegue

Construimos una solución completa de Machine Learning que estima el riesgo de que un cliente de una compañía de telecomunicaciones cancele el servicio (churn). El proyecto cubre todo el ciclo: datos, entrenamiento reproducible con `train.py`, una app web que usa el modelo entrenado, imagen Docker y despliegue.

Con autorización del docente, reemplazamos la interfaz de Streamlit que propone la guía por una app web publicada en Vercel: **https://radar-fuga-telco.vercel.app**

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

Todo el preprocesamiento vive dentro de un `Pipeline` de scikit-learn, así las mismas transformaciones del entrenamiento se aplican a cualquier cliente nuevo:

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

El random forest gana por 0,003, una diferencia mucho menor que la variación entre folds. Por eso aplicamos una regla de parsimonia: si el modelo más simple queda a menos de una desviación estándar del mejor, nos quedamos con el simple. **Elegimos la regresión logística**: rinde igual, se interpreta directamente con sus coeficientes, el artefacto pesa 10 KB en lugar de 16 MB y, como su predicción es una suma ponderada, la app web la puede calcular exactamente en el navegador.

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

## La app web

Streamlit necesita un servidor Python encendido todo el tiempo, y Vercel no lo ofrece. Por eso aprovechamos que el modelo es una regresión logística: `export_web.py` toma el pipeline entrenado de `model.joblib` y guarda en `web/model.json` las medianas y modas de imputación, la media y escala de cada numérica, las categorías del one-hot, los coeficientes y el intercepto. `web/churn-model.js` repite esas mismas operaciones en el navegador y aplica la sigmoide. El resultado es idéntico al de `model.predict_proba` (lo verificamos en 300 casos, ver Pruebas), no hay servidor y ningún dato que se ingrese sale del equipo de quien usa la página.

La app no vuelve a entrenar: solo lee `model.json`.

### Interpretación de la salida

- **Puntaje de riesgo** de 0 a 100: la salida del modelo multiplicada por 100.
- **Nivel:** bajo (menos de 30), medio (de 30 a 50) o alto (50 o más). Con 50 o más, la app dice "Se prevé que se vaya", que es la clase positiva del modelo con el umbral de 0,5.
- **Tasa real de fuga:** cuántos de los clientes de prueba con un puntaje parecido se fueron de verdad.
- **Qué explica el puntaje:** los cinco factores que más lo suben (rojo) o lo bajan (azul) frente al cliente promedio de entrenamiento, en puntos.
- **Qué podría retenerlo:** cuánto bajaría el puntaje al cambiar una sola cosa (contrato, forma de pago o servicios de soporte). Son asociaciones del modelo, no efectos causales garantizados.
- **Confiabilidad:** las métricas de prueba en lenguaje sencillo y el gráfico de puntaje frente a fuga real, con su tabla.

**Por qué decimos "puntaje" y no "probabilidad".** Como entrenamos con `class_weight="balanced"`, el modelo da más peso a los clientes que se van y su salida queda inflada: el promedio predicho es 41,5 % cuando la tasa real es 26,5 %. Por ejemplo, de los clientes de prueba con puntaje entre 70 y 80, en realidad se fue el 47 %. El orden sí es confiable (ROC-AUC 0,839), así que el puntaje sirve para priorizar clientes, y al lado mostramos siempre la tasa real del tramo.

La app solo ofrece combinaciones que existen en los datos. Si el cliente no tiene teléfono, "Varias líneas" queda bloqueado en "Sin servicio telefónico". Si no tiene internet, los seis servicios de internet quedan bloqueados en "Sin servicio de internet". Los valores por defecto del formulario son los más frecuentes del dataset.

## Estructura del proyecto

```
telco-churn-streamlit/
├── train.py                # entrenamiento reproducible
├── export_web.py           # exporta el modelo a web/model.json
├── model.joblib            # pipeline entrenado + metadatos
├── metrics.json            # métricas de la última ejecución de train.py
├── requirements.txt        # dependencias de Python con versiones fijas
├── requirements-dev.txt    # requirements.txt + pytest
├── package.json            # pruebas de la web (npm test)
├── Dockerfile              # imagen nginx que sirve web/
├── .dockerignore
├── data/
│   └── telco_churn.csv
├── web/                    # la app (Vercel y Docker sirven esta carpeta)
│   ├── index.html
│   ├── styles.css
│   ├── app.js              # interfaz
│   ├── churn-model.js      # inferencia en JavaScript
│   └── model.json          # parámetros exportados del pipeline
├── tests/
│   ├── test_web_export.py  # web/model.json coincide con model.joblib
│   ├── check_web.mjs       # la web da la misma probabilidad que sklearn
│   ├── ui.test.cjs         # la interfaz responde bien a los clics
│   └── web_cases.json      # 300 casos de verificación
└── colab/
    └── telco_churn_colab.ipynb   # revisión del dataset, entrenamiento y pruebas en Colab
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
python export_web.py
```

`train.py` debe imprimir la validación cruzada, el modelo elegido, la matriz de confusión y terminar con `Modelo guardado en model.joblib`. Con `random_state=42` las métricas salen idénticas en cada ejecución. `export_web.py` actualiza `web/model.json`; se niega a exportar si `train.py` llegara a elegir el random forest, porque la web solo reproduce la regresión logística.

## Cómo ejecutar la app

En local, sin instalar nada más:

```bash
python -m http.server 8000 --directory web
```

Y abrir http://localhost:8000. Para probar sin descargar nada: https://radar-fuga-telco.vercel.app.

**En Google Colab:** abrir https://colab.research.google.com/github/melo11miguel/telco-churn-streamlit/blob/main/colab/telco_churn_colab.ipynb y ejecutar todas las celdas. El notebook clona el repo, revisa el dataset, entrena, exporta, corre las pruebas y abre la app con un túnel temporal.

**En Vercel** el proyecto apunta a la carpeta `web/` como directorio raíz, sin paso de construcción. Cada push a `main` la vuelve a publicar.

## Pruebas

Necesitamos Node.js 20 o superior para las pruebas de la web.

```bash
python -m pytest -q   # web/model.json está al día con model.joblib
npm install
npm test              # misma probabilidad que sklearn en 300 casos + prueba de la interfaz
```

`npm test` también corre contra la versión publicada: `node tests/ui.test.cjs https://radar-fuga-telco.vercel.app/`.

## Cómo construir la imagen Docker

La imagen parte de `nginx:1.27-alpine` y solo copia la carpeta `web/`, porque el modelo ya va exportado en `web/model.json`. Desde la carpeta del proyecto:

```bash
docker build -t radar-fuga-telco .
docker images
```

El punto final indica que el contexto de construcción es la carpeta actual.

## Cómo ejecutar el contenedor

Primera prueba interactiva (detener con Ctrl+C):

```bash
docker run --rm -p 8080:80 radar-fuga-telco
```

Y abrir http://localhost:8080. En segundo plano y con reinicio automático:

```bash
docker run -d --name radar-fuga --restart unless-stopped -p 80:80 radar-fuga-telco
docker ps
docker logs radar-fuga
```

`-p 80:80` conecta el puerto 80 de la máquina con el puerto 80 del contenedor, donde escucha nginx. La imagen incluye un `HEALTHCHECK`, así que a los pocos segundos `docker ps` debe mostrar el estado `(healthy)`.

## Despliegue en Google Cloud (Compute Engine)

1. En Google Cloud Console, seleccionar el proyecto, abrir **Compute Engine** y habilitar la API si la consola lo pide.
2. Crear una instancia de VM:
   - Imagen: **Ubuntu 24.04 LTS** (x86/64).
   - Tipo de máquina: `e2-micro` alcanza, porque nginx solo sirve archivos estáticos.
   - Con dirección IP externa.
   - En **Firewall**, marcar **Permitir tráfico HTTP**. Eso le pone a la VM la etiqueta `http-server` y crea la regla de firewall que abre el puerto 80.
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
   docker build -t radar-fuga-telco .
   docker run -d --name radar-fuga --restart unless-stopped -p 80:80 radar-fuga-telco
   docker ps
   ```

5. Si no marcamos **Permitir tráfico HTTP** al crear la VM, abrir el puerto desde Cloud Shell:

   ```bash
   gcloud compute instances add-tags NOMBRE_DE_LA_VM --tags=http-server --zone=ZONA_DE_LA_VM
   gcloud compute firewall-rules create default-allow-http \
     --allow=tcp:80 --direction=INGRESS --target-tags=http-server
   ```

6. Abrir `http://IP_EXTERNA_DE_LA_VM` en el navegador.

Este acceso público es solo para el laboratorio. En producción haría falta HTTPS y un dominio propio (Vercel ya los da por defecto).

**Al terminar, detener o eliminar la VM** para que no siga generando costos.

### Actualizar la aplicación desplegada

El contenedor no se actualiza solo: hay que construir una imagen nueva y crear un contenedor nuevo.

```bash
git pull
docker stop radar-fuga
docker rm radar-fuga
docker build -t radar-fuga-telco .
docker run -d --name radar-fuga --restart unless-stopped -p 80:80 radar-fuga-telco
```

### Diagnóstico rápido

| Situación | Qué revisar |
|---|---|
| `docker build` falla | Leer el error completo; confirmar que la carpeta `web/` está en el directorio actual. |
| El contenedor se detiene | `docker ps -a` y `docker logs radar-fuga`. |
| Funciona en la VM pero no desde fuera | La regla de firewall, la etiqueta `http-server` en la VM y la IP externa. Desde la VM, `curl -I localhost` debe responder `200 OK`. |
| La página abre pero no calcula | Abrir la consola del navegador; `http://IP/model.json` debe descargar el modelo. |
| Los cambios no aparecen | Seguir los pasos de actualización de arriba. |

## Ejemplo de entrada y salida

Los tres casos se cargan con los botones de ejemplo de la parte superior de la app. Los campos que no aparecen quedan en su valor por defecto (el más frecuente del dataset).

**Caso 1: cliente nuevo de alto riesgo** (botón "Cliente nuevo, alto riesgo")

| Campo en la app | Valor |
|---|---|
| Antigüedad | 1 mes |
| Cargo mensual | US$ 95 |
| Tipo de contrato | Mes a mes |
| Servicio de internet | Fibra óptica |
| Método de pago | Cheque electrónico |
| Seguridad en línea / Soporte técnico | No / No |
| Factura electrónica | Sí |

Salida esperada: **puntaje 79/100, riesgo alto, "Se prevé que se vaya"**. De los 149 clientes de prueba con puntaje entre 70 y 80, el 47 % realmente canceló. Lo que más lo retiene: ofrecer un contrato de dos años, que baja el puntaje a 48.

**Caso 2: cliente fiel de bajo riesgo** (botón "Cliente fiel, bajo riesgo")

| Campo en la app | Valor |
|---|---|
| Antigüedad | 70 meses |
| Cargo mensual | US$ 25 |
| Tipo de contrato | Dos años |
| Servicio de internet | Sin internet (los seis servicios de internet quedan bloqueados) |
| Método de pago | Transferencia bancaria automática |
| Factura electrónica | No |

Salida esperada: **puntaje 1/100, riesgo bajo, "Se prevé que se quede"**. De los 311 clientes de prueba con puntaje entre 0 y 10, el 1 % canceló.

**Caso 3: cliente típico** (botón "Cliente típico", o el formulario sin tocar: antigüedad 29 meses, cargo US$ 70,35, mes a mes, fibra óptica, cheque electrónico). Salida esperada: **puntaje 72/100, riesgo alto.** Tiene sentido: el perfil más frecuente del dataset combina justo los factores de mayor riesgo.

## Prueba cruzada

Otra pareja debe poder reproducir el proyecto siguiendo solo este README, sin instrucciones verbales.

| Paso | Resultado | Observaciones |
|---|---|---|
| Clonar el repositorio | | |
| `docker build` | | |
| `docker run` y abrir la app | | |
| Probar el caso 1 del ejemplo | | |
| Instrucciones faltantes o errores | | |

Pareja que hizo la prueba: ____________ Fecha: ____________
