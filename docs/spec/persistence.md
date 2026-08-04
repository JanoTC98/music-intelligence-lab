# Persistencia

## MySQL (principal)

- **Versión mínima:** MySQL >= 8.0.16 (requerido para restricciones `CHECK`).
- **Host:** `127.0.0.1`.
- **Usuario:** `spotify_app` con host `127.0.0.1`.
- **Base de datos:** `spotify_app`.
- **Tablas:** eventos, feedback, sesiones, y tablas de auditoría.
- **Carácter opcional:** la persistencia es opcional y el tracking debe fallar de forma abierta.

## SQL Server (alternativa)

- SQL Server es una alternativa, no el motor oficial.
- No se añade `pyodbc` a dependencias de producción.
- Se documenta como requisito externo.

## Codificación de contraseñas

Si la contraseña contiene caracteres reservados de URL como `@`, `:`, `/`, `?` o `#`, debe codificarse antes de insertarla en `DATABASE_URL`. Alternativamente, el código debe construir la URL con `sqlalchemy.URL.create()`.

## Scripts

- `database/mysql/00_create_database_and_user.sql`
- `database/mysql/01_create_schema.sql`
- `database/mysql/02_verify_schema.sql`
- `database/sqlserver/00_create_database_and_login.sql`
- `database/sqlserver/01_create_schema.sql`
- `database/sqlserver/02_verify_schema.sql`