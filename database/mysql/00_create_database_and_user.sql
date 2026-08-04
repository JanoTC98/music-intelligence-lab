-- MySQL database and user setup for Spotify Music Intelligence
-- MySQL >= 8.0.16 required (CHECK constraints)

CREATE DATABASE IF NOT EXISTS spotify_app
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'spotify_app'@'127.0.0.1'
    IDENTIFIED BY '<password>';

GRANT ALL PRIVILEGES ON spotify_app.* TO 'spotify_app'@'127.0.0.1';

FLUSH PRIVILEGES;