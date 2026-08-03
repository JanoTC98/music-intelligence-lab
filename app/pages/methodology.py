"""Methodology and limitations page."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Metodología y limitaciones")
    st.caption("Decisiones, definiciones y restricciones del proyecto.")

    st.subheader("Definiciones de variables")
    st.markdown(
        """
| Variable | Significado |
|---|---|
| **valence** | Positividad musical percibida (0 = negativo, 1 = positivo). |
| **danceability** | Idoneidad para bailar según tempo, ritmo y estabilidad. |
| **energy** | Intensidad y actividad percibida de la canción. |
| **speechiness** | Presencia de palabra hablada (0 = musical, 1 = hablado). |
| **acousticness** | Confianza de que la grabación es acústica (no electrónica). |
| **instrumentalness** | Probabilidad de ausencia de voz (1 = instrumental). |
| **liveness** | Presencia de audiencia en vivo (mayor valor, mayor presencia). |
| **loudness** | Volumen medio en decibelios (dB). |
| **tempo** | Velocidad estimada en pulsaciones por minuto (BPM). |
| **mode** | Modalidad: 0 = menor, 1 = mayor. |
| **key** | Tonalidad estimada (-1 a 11 en el modelo de campos). |
| **time_signature** | Compás estimado (p. ej. 4/4 → 4). |
| **popularity** | Metadato contextual; no participa en la similitud inicial. |
"""
    )

    st.subheader("Cómo funciona el proyecto")
    st.markdown(
        """
- **Identidad musical:** cada grabación se agrupa con un
  `recording_group_id` exacto (huella estable + SHA-256). Es la unidad de
  recomendación y de entrenamiento; nunca se dividen grupos entre conjuntos.
- **Recomendador por canción:** vecinos más cercanos sobre características
  escaladas (similitud coseno). La popularidad, duración y géneros son filtros
  opcionales.
- **Recomendador por preferencias:** distancia euclidiana ponderada respecto a
  un perfil editable. Los pesos van de 0 a 3; peso 0 ignora la variable.
- **Clasificación:** laboratorios experimentales multietiqueta (114 géneros) y
  multiclase (género dominante). El test está congelado y solo se usa una vez.
"""
    )

    st.subheader("Limitaciones importantes")
    st.markdown(
        """
- El dataset de 114.000 filas tiene un bloque por género (1.000 filas cada
  uno); **no permite estimar la prevalencia real de géneros en Spotify**.
- Las **similitudes y puntuaciones no son probabilidades** de gusto ni de
  pertenencia a un género.
- `popularity = 0` se conserva; no indica falta de calidad.
- El análisis acústico incompleto se excluye de los recomendadores y del
  clasificador baseline.
- Los clasificadores son **módulos experimentales**; no bloquean el MVP.
- No se almacenan datos personales ni se integra con cuentas de Spotify.
- Esta aplicación **no tiene afiliación oficial con Spotify**.
"""
    )

    st.subheader("Decisiones cerradas del proyecto")
    st.markdown(
        """
- El CSV original es inmutable.
- La unidad de modelado es `recording_group_id`.
- El clasificador principal es multietiqueta; el secundario, multiclase.
- El test se congela y no se usa durante la selección de modelos.
- La aplicación carga artefactos; **no entrena** durante la interacción.
"""
    )


if st.runtime.exists():
    render()
