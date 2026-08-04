-- SQL Server database and login setup for Spotify Music Intelligence
-- SQL Server is an alternative, not the official engine.
-- Requires an external driver (e.g., pyodbc) not included in production dependencies.

CREATE DATABASE spotify_app;
GO

CREATE LOGIN spotify_app WITH PASSWORD = '<password>';
GO

CREATE USER spotify_app FOR LOGIN spotify_app;
GO

ALTER ROLE db_datareader ADD MEMBER spotify_app;
ALTER ROLE db_datawriter ADD MEMBER spotify_app;
GO