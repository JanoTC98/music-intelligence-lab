# Database

## MySQL (principal)

- **Versión mínima:** MySQL >= 8.0.16 (requerido para restricciones `CHECK`).
- **Host:** `127.0.0.1`.
- **Usuario:** `spotify_app` con host `127.0.0.1`.
- **Base de datos:** `spotify_app`.

### Scripts

| Script | Descripción |
|---|---|
| `database/mysql/00_create_database_and_user.sql` | Crea la base de datos y el usuario |
| `database/mysql/01_create_schema.sql` | Crea las seis tablas |
| `database/mysql/02_verify_schema.sql` | Verifica la integridad del esquema |

### Codificación de contraseñas

Si la contraseña contiene caracteres reservados de URL como `@`, `:`, `/`, `?` o `#`, debe codificarse antes de insertarla en `DATABASE_URL`. Alternativamente, el código debe construir la URL con `sqlalchemy.URL.create()`.

La persistencia es **opcional** y el tracking debe fallar de forma abierta.

## SQL Server (alternativa)

SQL Server es una alternativa, no el motor oficial. Los scripts se encuentran en `database/sqlserver/`. No se añade `pyodbc` a dependencias de producción; se documenta como requisito externo.