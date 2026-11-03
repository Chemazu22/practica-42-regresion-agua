# Práctica 4.2: Regresión Lineal y Correlación

Aplicación educativa en Streamlit que reproduce el análisis de consumo per cápita de agua potable (CPC) del script de la práctica. Genera datos sintéticos en DuckDB en memoria, calcula correlaciones, ajusta un modelo OLS y muestra diagnósticos, métricas, gráficas y un simulador interactivo.

## Contenido

- `app.py`: aplicación, análisis y visualizaciones.
- `requirements.txt`: dependencias para Python 3.12.
- `.streamlit/config.toml`: tema visual y opciones del servidor.

## Ejecutar localmente

Se recomienda Python 3.12 en un entorno virtual.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Abre la dirección local que Streamlit muestre en la terminal (normalmente `http://localhost:8501`). En Windows, activa el entorno con `.venv\Scripts\activate`.

## Publicar en Streamlit Community Cloud

1. Sube estos archivos a un **repositorio público de GitHub** en la rama `main`.
2. Inicia sesión en [share.streamlit.io](https://share.streamlit.io/) con GitHub y elige **Create app**.
3. Selecciona el repositorio público, la rama `main` y el archivo principal `app.py` (en la raíz).
4. En **Advanced settings**, selecciona **Python 3.12** y despliega la aplicación.
5. Cuando esté publicada, abre sus opciones y asigna un subdominio disponible, por ejemplo `practica-42-regresion-agua-chemazu22`. La dirección pública tendrá la forma `https://practica-42-regresion-agua-chemazu22.streamlit.app`.

Streamlit instala las bibliotecas declaradas en `requirements.txt`. El subdominio de ejemplo está personalizado con el usuario de GitHub de esta publicación; si no está disponible, elige otro y consulta la dirección final en el panel de la app.

## Contenido de las pestañas

- **Datos:** tabla de residencias generada en DuckDB y descarga CSV.
- **Correlación:** matriz de Pearson y asociación de cada variable con el CPC.
- **Modelo OLS:** coeficientes, errores estándar, pruebas t, valores p, intervalos de confianza, significancia con α seleccionado, comparación con los coeficientes usados para simular y resumen completo de `statsmodels`.
- **Supuestos:** prueba de normalidad Shapiro-Wilk y estadístico Durbin-Watson con interpretación según α y el rango 1.5–2.5.
- **Métricas:** R², RMSE, R² ajustado y ecuación ajustada con scikit-learn.
- **Panel de gráficas:** heatmap, tendencia humedad-consumo, reales vs. predichos, residuos vs. ajustados, Q-Q normal e histograma; descarga PNG.
- **Simulador:** consumo CPC predicho, intervalo de predicción y consumo diario total de la vivienda a partir de controles interactivos.

## Lógica reproducible

La consulta DuckDB utiliza `SELECT setseed(semilla)` y una CTE `base` para generar temperatura (18–32 °C), humedad (50–90 %), personas (3–7), consumo de bidón (0.2–2.0 lt/hab-día) y ruido uniforme de −3 a 3. La respuesta conserva la fórmula del ejercicio:

```text
CPC = 94 + 2.5·Temperatura − 0.8·Humedad − 5.2·Personas + 15·Bidón + ruido
```

Por defecto se usan semilla `0.42` y 100 residencias. La simulación está almacenada en caché con `st.cache_data`.
