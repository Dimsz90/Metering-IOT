CREATE DATABASE IF NOT EXISTS gas_meter_db;
USE gas_meter_db;

CREATE TABLE IF NOT EXISTS devices (
    device_id VARCHAR(50) PRIMARY KEY,
    last_seen DATETIME,
    last_reading VARCHAR(50),
    confidence FLOAT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS readings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    device_id VARCHAR(50),
    reading VARCHAR(50) NOT NULL,
    confidence FLOAT,
    image_filename VARCHAR(255),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX (device_id),
    INDEX (timestamp),
    FOREIGN KEY (device_id) REFERENCES devices(device_id) ON DELETE CASCADE
);