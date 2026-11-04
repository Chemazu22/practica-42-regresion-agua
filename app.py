"""Aplicación interactiva para la práctica 4.2 de regresión lineal."""

import os
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "practica42_matplotlib")
)

from io import BytesIO

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson


st.set_page_config(
    page_title="Práctica 4.2 | Regresión y correlación",
    page_icon="💧",
    layout="wide",
)

VARIABLES = [
    "temperatura_max",
    "humedad_max",
    "personas_familia",
    "consumo_bidon_pc",
]
RESPUESTA = "cpc_consumo"
COEFICIENTES_SIMULACION = {
    "const": 94.0,
    "temperatura_max": 2.5,
    "humedad_max": -0.8,
    "personas_familia": -5.2,
    "consumo_bidon_pc": 15.0,
}
ETIQUETAS = {
    "const": "Intercepto",
    "temperatura_max": "Temperatura máxima",
    "humedad_max": "Humedad máxima",
    "personas_familia": "Personas por familia",
    "consumo_bidon_pc": "Consumo de bidón por persona",
}


# PASO 1: SIMULACIÓN REPRODUCIBLE DE DATOS EN DUCKDB
@st.cache_data(show_spinner="Simulando residencias en DuckDB…")
def simular_datos(semilla: float, cantidad: int) -> pd.DataFrame:
    """Genera los datos en una base DuckDB en memoria con una CTE base."""
    conexion = duckdb.connect(database=":memory:")
    try:
        conexion.execute(f"SELECT setseed({semilla:.8f});")
        conexion.execute(
            f"""
            CREATE TABLE telemetria_agua AS
            WITH base AS (
                SELECT
                    residencia_id,
                    ROUND(18 + random() * 14, 2) AS temperatura_max,
                    ROUND(50 + random() * 40, 2) AS humedad_max,
                    ROUND(3 + random() * 4, 0) AS personas_familia,
                    ROUND(0.2 + random() * 1.8, 2) AS consumo_bidon_pc,
                    random() * 6 - 3 AS ruido
                FROM generate_series(1, {int(cantidad)}) AS t(residencia_id)
            )
            SELECT
                residencia_id,
                temperatura_max,
                humedad_max,
                personas_familia,
                consumo_bidon_pc,
                ROUND(
                    94 + 2.5 * temperatura_max - 0.8 * humedad_max
                    - 5.2 * personas_familia + 15 * consumo_bidon_pc + ruido,
                    2
                ) AS cpc_consumo
            FROM base
            ORDER BY residencia_id
            """
        )
        return conexion.execute(
            "SELECT * FROM telemetria_agua ORDER BY residencia_id"
        ).df()
    finally:
        conexion.close()


def construir_panel(
    datos: pd.DataFrame,
    correlacion: pd.DataFrame,
    reales: pd.Series,
    predichos: np.ndarray,
    residuos: pd.Series,
    r2: float,
) -> bytes:
    """Construye y cierra el panel 2x3; devuelve un PNG descargable."""
    sns.set_theme(style="whitegrid")
    figura, ejes = plt.subplots(2, 3, figsize=(18, 10))
    figura.suptitle(
        "Panel de diagnóstico visual: Regresión lineal múltiple (Subtema 4.2)",
        fontsize=16,
        fontweight="bold",
    )

    sns.heatmap(
        correlacion,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar=True,
        ax=ejes[0, 0],
    )
    ejes[0, 0].set_title("1. Matriz de correlación de Pearson")

    sns.regplot(
        data=datos,
        x="humedad_max",
        y=RESPUESTA,
        color="blue",
        line_kws={"color": "red", "linewidth": 2},
        ax=ejes[0, 1],
    )
    ejes[0, 1].set_title("2. Tendencia: humedad máxima vs consumo")
    ejes[0, 1].set_xlabel("Humedad máxima (%)")
    ejes[0, 1].set_ylabel("CPC consumo (lt/hab-día)")

    ejes[0, 2].scatter(predichos, reales, alpha=0.7, color="teal")
    minimo = min(float(reales.min()), float(np.min(predichos)))
    maximo = max(float(reales.max()), float(np.max(predichos)))
    ejes[0, 2].plot([minimo, maximo], [minimo, maximo], "r--", linewidth=2)
    ejes[0, 2].set_title(f"3. Valores reales vs predichos (R² = {r2:.1%})")
    ejes[0, 2].set_xlabel("CPC predicho")
    ejes[0, 2].set_ylabel("CPC real")

    ejes[1, 0].scatter(predichos, residuos, color="purple", alpha=0.6)
    ejes[1, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    ejes[1, 0].set_title("4. Residuos vs valores ajustados")
    ejes[1, 0].set_xlabel("Valores ajustados")
    ejes[1, 0].set_ylabel("Residuos (e = y - ŷ)")

    sm.qqplot(residuos, line="45", fit=True, ax=ejes[1, 1])
    ejes[1, 1].set_title("5. Gráfico Q-Q normal de residuos")

    sns.histplot(residuos, kde=True, color="steelblue", ax=ejes[1, 2])
    ejes[1, 2].axvline(0, color="red", linestyle="--", linewidth=1.5)
    ejes[1, 2].set_title("6. Histograma de residuos")
    ejes[1, 2].set_xlabel("Residuo")

    figura.tight_layout(rect=(0, 0, 1, 0.96))
    salida = BytesIO()
    figura.savefig(salida, format="png", dpi=160, bbox_inches="tight")
    plt.close(figura)
    return salida.getvalue()


def formato_ecuacion(intercepto: float, coeficientes: np.ndarray) -> str:
    signos = ["+" if valor >= 0 else "−" for valor in coeficientes]
    return (
        f"CPC = {intercepto:.2f} {signos[0]} {abs(coeficientes[0]):.2f}·Temp "
        f"{signos[1]} {abs(coeficientes[1]):.2f}·Humedad "
        f"{signos[2]} {abs(coeficientes[2]):.2f}·Personas "
        f"{signos[3]} {abs(coeficientes[3]):.2f}·Bidón"
    )


# Panel lateral: parámetros de la práctica y datos académicos.
with st.sidebar:
    st.header("Parámetros de simulación")
    semilla = st.number_input(
        "Semilla de DuckDB",
        min_value=0.0,
        max_value=1.0,
        value=0.42,
        step=0.01,
        format="%.2f",
    )
    cantidad = st.number_input(
        "Número de residencias",
        min_value=20,
        max_value=1000,
        value=100,
        step=10,
    )
    alpha = st.selectbox("Nivel de significancia (α)", [0.01, 0.05, 0.10], index=1)
    st.divider()
    st.subheader("Datos de la materia")
    st.caption("Ingeniería en Sistemas")
    st.caption("Análisis y Exploración de Datos")
    st.caption("TecNM · INC-2505 · Unidad 4")
    st.caption("Subtema 4.2: Regresión lineal y correlación")


st.title("Práctica 4.2: Regresión Lineal y Correlación")
st.write(
    "Explora la relación entre las condiciones de la vivienda y el consumo per cápita "
    "de agua potable (CPC). Los datos se generan en DuckDB en memoria."
)

with st.spinner("Calculando correlaciones, modelos y diagnósticos…"):
    df_agua = simular_datos(float(semilla), int(cantidad))

    # PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON
    matriz_correlacion = df_agua.drop(columns=["residencia_id"]).corr(method="pearson")
    correlacion_respuesta = matriz_correlacion[RESPUESTA].drop(RESPUESTA)

    # PASO 3: REGRESIÓN LINEAL MÚLTIPLE CON STATSMODELS (OLS)
    X = df_agua[VARIABLES]
    Y = df_agua[RESPUESTA]
    X_const = sm.add_constant(X)
    modelo_ols = sm.OLS(Y, X_const).fit()

    # PASO 4: VALIDACIÓN DE SUPUESTOS DE LOS RESIDUOS
    residuos = modelo_ols.resid
    estadistico_shapiro, p_shapiro = stats.shapiro(residuos)
    estadistico_dw = durbin_watson(residuos)

    # PASO 5: MÉTRICAS Y ECUACIÓN AJUSTADA CON SCIKIT-LEARN
    modelo_sklearn = LinearRegression()
    modelo_sklearn.fit(X, Y)
    predichos = modelo_sklearn.predict(X)
    r2 = r2_score(Y, predichos)
    rmse = float(np.sqrt(mean_squared_error(Y, predichos)))
    ecuacion = formato_ecuacion(float(modelo_sklearn.intercept_), modelo_sklearn.coef_)

    # PASO 6: PANEL DE GRÁFICAS DE DIAGNÓSTICO
    panel_png = construir_panel(
        df_agua, matriz_correlacion, Y, predichos, residuos, r2
    )

st.caption(
    f"Base reproducible · Semilla {semilla:.2f} · {len(df_agua)} residencias · α = {alpha:.2f}"
)
metrica1, metrica2, metrica3, metrica4 = st.columns(4)
metrica1.metric("R²", f"{modelo_ols.rsquared:.3f}", f"{modelo_ols.rsquared:.1%}")
metrica2.metric("R² ajustado", f"{modelo_ols.rsquared_adj:.3f}")
metrica3.metric("RMSE", f"{rmse:.4f} lt/hab-día")
metrica4.metric("Prob(F)", f"{modelo_ols.f_pvalue:.3e}")

tab_datos, tab_corr, tab_ols, tab_supuestos, tab_metricas, tab_graficas, tab_simulador = st.tabs(
    [
        "Datos",
        "Correlación",
        "Modelo OLS",
        "Supuestos",
        "Métricas",
        "Panel de gráficas",
        "Simulador",
    ]
)

with tab_datos:
    st.subheader("Datos simulados de residencias")
    st.dataframe(df_agua, width="stretch", hide_index=True)
    st.download_button(
        "Descargar datos en CSV",
        data=df_agua.to_csv(index=False).encode("utf-8-sig"),
        file_name="datos_consumo_agua.csv",
        mime="text/csv",
    )

with tab_corr:
    st.subheader("Matriz de correlación de Pearson (r)")
    st.dataframe(matriz_correlacion.round(4), width="stretch")
    st.subheader("Correlación de cada variable con el CPC")
    tabla_corr = correlacion_respuesta.rename("Correlación con CPC").to_frame()
    tabla_corr.index.name = "Variable"
    st.dataframe(tabla_corr.round(4), width="stretch")
    st.caption("La correlación describe asociación lineal; por sí sola no demuestra causalidad.")

with tab_ols:
    st.subheader("Coeficientes del modelo OLS")
    intervalo = modelo_ols.conf_int(alpha=0.05)
    tabla_coef = pd.DataFrame(
        {
            "coef": modelo_ols.params,
            "std err": modelo_ols.bse,
            "t": modelo_ols.tvalues,
            "p": modelo_ols.pvalues,
            "IC 95% inferior": intervalo[0],
            "IC 95% superior": intervalo[1],
            "¿Significativo?": modelo_ols.pvalues < alpha,
        }
    )
    st.dataframe(tabla_coef.style.format(precision=4), width="stretch")
    st.subheader("Coeficientes estimados y usados en la simulación")
    comparacion = pd.DataFrame(
        {
            "Variable": list(COEFICIENTES_SIMULACION),
            "Coeficiente de simulación": list(COEFICIENTES_SIMULACION.values()),
            "Coeficiente OLS estimado": modelo_ols.params.reindex(
                COEFICIENTES_SIMULACION
            ).values,
        }
    )
    comparacion["Diferencia (estimado − simulación)"] = (
        comparacion["Coeficiente OLS estimado"] - comparacion["Coeficiente de simulación"]
    )
    comparacion["Variable"] = comparacion["Variable"].map(ETIQUETAS)
    st.dataframe(comparacion.style.format(precision=4), width="stretch", hide_index=True)
    st.subheader("Interpretación de los coeficientes")
    efecto_temp = "aumenta" if modelo_ols.params[VARIABLES[0]] >= 0 else "disminuye"
    efecto_humedad = "aumenta" if modelo_ols.params[VARIABLES[1]] >= 0 else "disminuye"
    efecto_personas = "aumenta" if modelo_ols.params[VARIABLES[2]] >= 0 else "disminuye"
    efecto_bidon = "aumenta" if modelo_ols.params[VARIABLES[3]] >= 0 else "disminuye"
    st.markdown(
        f"- **Intercepto ({modelo_ols.params['const']:.2f}):** CPC estimado cuando las cuatro variables explicativas valen cero; sirve como referencia matemática y está fuera del rango real de la muestra.\n"
        f"- **Temperatura ({modelo_ols.params[VARIABLES[0]]:.2f}):** por cada 1 °C adicional, el CPC {efecto_temp} {abs(modelo_ols.params[VARIABLES[0]]):.2f} lt/hab-día, manteniendo las demás variables constantes.\n"
        f"- **Humedad ({modelo_ols.params[VARIABLES[1]]:.2f}):** por cada punto porcentual adicional, el CPC {efecto_humedad} {abs(modelo_ols.params[VARIABLES[1]]):.2f} lt/hab-día, manteniendo lo demás constante.\n"
        f"- **Personas por familia ({modelo_ols.params[VARIABLES[2]]:.2f}):** una persona adicional se asocia con que el CPC {efecto_personas} {abs(modelo_ols.params[VARIABLES[2]]):.2f} lt/hab-día, manteniendo lo demás constante.\n"
        f"- **Bidón ({modelo_ols.params[VARIABLES[3]]:.2f}):** por cada lt/hab-día adicional de consumo de bidón, el CPC {efecto_bidon} {abs(modelo_ols.params[VARIABLES[3]]):.2f} lt/hab-día, manteniendo lo demás constante."
    )
    with st.expander("Resumen completo de statsmodels (summary())"):
        st.code(modelo_ols.summary().as_text(), language="text")

with tab_supuestos:
    st.subheader("Diagnóstico de los residuos")
    col_norm, col_ind = st.columns(2)
    with col_norm:
        st.metric("Shapiro-Wilk · estadístico", f"{estadistico_shapiro:.4f}")
        st.metric("Shapiro-Wilk · valor p", f"{p_shapiro:.4f}")
        if p_shapiro > alpha:
            st.success(
                f"Con α = {alpha:.2f}, no se rechaza la normalidad de los residuos (p > α)."
            )
        else:
            st.warning(
                f"Con α = {alpha:.2f}, se rechaza la normalidad de los residuos (p ≤ α)."
            )
    with col_ind:
        st.metric("Durbin-Watson", f"{estadistico_dw:.4f}")
        if 1.5 <= estadistico_dw <= 2.5:
            st.success("DW está entre 1.5 y 2.5; no se observa autocorrelación relevante.")
        else:
            st.warning("DW está fuera del rango 1.5–2.5; conviene revisar la independencia.")
    st.caption(
        "Shapiro-Wilk evalúa normalidad. Durbin-Watson se interpreta considerando el orden de las observaciones."
    )

with tab_metricas:
    st.subheader("Evaluación del ajuste")
    st.write(f"**Coeficiente de determinación (R²):** {r2:.2%}")
    st.write(f"**RMSE:** {rmse:.4f} lt/hab-día")
    st.write(f"**R² ajustado (OLS):** {modelo_ols.rsquared_adj:.4f}")
    st.write("**Ecuación del modelo ajustado:**")
    st.code(ecuacion)
    st.write(
        f"El modelo explica aproximadamente {r2:.1%} de la variabilidad observada del CPC "
        "en estos datos simulados."
    )

with tab_graficas:
    st.subheader("Panel de diagnóstico visual")
    st.image(panel_png, caption="Seis vistas del análisis y diagnóstico del modelo.", width="stretch")
    st.download_button(
        "Descargar panel de gráficas en PNG",
        data=panel_png,
        file_name="panel_regresion_agua.png",
        mime="image/png",
    )

with tab_simulador:
    st.subheader("Estimación para una vivienda")
    st.write("Ajusta las variables para obtener el CPC estimado y un intervalo de predicción.")
    c1, c2 = st.columns(2)
    with c1:
        temperatura = st.slider("Temperatura máxima (°C)", 18.0, 32.0, 25.0, 0.1)
        humedad = st.slider("Humedad máxima (%)", 50.0, 90.0, 70.0, 0.1)
    with c2:
        personas = st.slider("Personas en la familia", 3, 7, 4, 1)
        bidon = st.slider("Consumo de bidón (lt/hab-día)", 0.2, 2.0, 1.0, 0.01)
    entrada = pd.DataFrame(
        [[temperatura, humedad, personas, bidon]], columns=VARIABLES
    )
    entrada_const = sm.add_constant(entrada, has_constant="add")
    prediccion = float(modelo_ols.predict(entrada_const).iloc[0])
    intervalo_pred = modelo_ols.get_prediction(entrada_const).summary_frame(alpha=alpha).iloc[0]
    inferior = float(intervalo_pred["obs_ci_lower"])
    superior = float(intervalo_pred["obs_ci_upper"])
    total_vivienda = prediccion * personas
    res1, res2, res3 = st.columns(3)
    res1.metric("CPC predicho", f"{prediccion:.2f} lt/hab-día")
    res2.metric("Intervalo de predicción", f"[{inferior:.2f}, {superior:.2f}]")
    res3.metric("Consumo total de la vivienda", f"{total_vivienda:.2f} lt/día")
    st.caption(
        f"Intervalo de predicción para una observación nueva al {(1-alpha)*100:.0f}% de confianza; "
        "el consumo total se calcula como CPC predicho × número de personas."
    )
# Resultados de referencia transcritos del PDF; no corresponden al ajuste de esta muestra.
st.info("Resultados del profesor transcritos del PDF de referencia")
st.caption("Valores informativos copiados del documento entregado por el profesor; las métricas de la app se calculan por separado con la simulación activa.")
ref_cols = st.columns(4)
ref_cols[0].metric("R² · referencia", "0.957")
ref_cols[1].metric("R² ajustado · referencia", "0.955")
ref_cols[2].metric("RMSE · referencia", "2.0152 lt/hab-día")
ref_cols[3].metric("Prob(F) · referencia", "2.84e-64")
st.write("**Shapiro-Wilk:** estadístico 0.9881 · p = 0.5097 &nbsp;&nbsp; | &nbsp;&nbsp; **Durbin-Watson:** 2.0298")
st.write("**Ecuación reportada:** CPC = 94.49 + 2.48·Temp − 0.79·Humedad − 5.18·Personas + 14.89·Bidón")
st.subheader("Coeficientes OLS transcritos del PDF")
coef_profesor = pd.DataFrame(
    {
        "coef": [94.4925, 2.4828, -0.7938, -5.1802, 14.8872],
        "std err": [1.996, 0.052, 0.017, 0.161, 0.407],
        "t": [47.337, 48.169, -47.530, -32.228, 36.561],
        "p": ["<0.001"] * 5,
        "IC 95% inferior": [90.529, 2.380, -0.827, -5.500, 14.079],
        "IC 95% superior": [98.456, 2.585, -0.761, -4.861, 15.695],
    },
    index=["const", "temperatura_max", "humedad_max", "personas_familia", "consumo_bidon_pc"],
)
st.dataframe(coef_profesor, width="stretch")
