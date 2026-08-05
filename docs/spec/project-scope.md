# Alcance del proyecto

> **Proyecto:** Spotify Music Intelligence  
> **Estado:** implementación activa  
> **Versión de referencia:** 2.0

## Producto central

El recomendador musical basado en contenido que sugiere canciones similares a partir de una canción de entrada o de preferencias del usuario.

## Módulos complementarios

- Clasificación multietiqueta de géneros (laboratorio supervisado).
- Clasificación multiclase de género acústico dominante (experimento).
- Aplicación web multipágina en Streamlit.
- Pipeline reproducible de datos y modelos.

## Fuera de alcance

- Persistencia opcional de eventos y feedback en MySQL (esquemas en `database/`, especificación en `docs/spec/persistence.md`).
- Power BI y cualquier dashboard de Power BI.
- Cambios de tecnología principal sin aprobación del propietario.
- Modificación del dataset bruto.
- Reentrenamiento de modelos durante interacciones de usuario en Streamlit.