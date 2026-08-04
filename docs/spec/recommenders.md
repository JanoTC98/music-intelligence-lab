# Recomendadores

## Recomendador por canción

- Basado en vecinos y distancias.
- Usa features acústicas normalizadas (StandardScaler) y distancia coseno.
- Excluye la propia grabación del resultado.
- Deduplica por `recording_group_id`.
- Filtros desactivados por defecto (explícito, género, duración, artista, popularidad mínima).

## Recomendador por preferencias

- Usa distancia ponderada.
- Variables no configuradas tienen peso 0.
- Permite presets editables.

## Reglas esenciales

- Similitud no significa probabilidad de gusto.
- Puntuaciones no calibradas no se llaman probabilidades.
- Streamlit no reentrena modelos durante una interacción.
- Artefactos ausentes producen un error claro, no entrenamiento silencioso.