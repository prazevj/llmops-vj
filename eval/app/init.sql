-- Database initialization script
CREATE DATABASE IF NOT EXISTS apa_evaluation;
USE apa_evaluation;

-- Create connections table
CREATE TABLE IF NOT EXISTS connections (
    connection_id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    connection_type VARCHAR(50) NOT NULL,
    connection_config JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create metrics table
CREATE TABLE IF NOT EXISTS metrics (
    metrics_id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    evaluation_id VARCHAR(36),
    metrics_name VARCHAR(255) NOT NULL,
    description TEXT,
    metric_type VARCHAR(100) NOT NULL,
    configuration JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_evaluation_id (evaluation_id),
    INDEX idx_metrics_name (metrics_name)
);

-- Create evaluations table
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    evaluation_name VARCHAR(255) NOT NULL,
    metrics_id VARCHAR(36) NOT NULL,
    metrics_name VARCHAR(255) NOT NULL,
    prompt_id VARCHAR(36) NOT NULL,
    prompt_name VARCHAR(255) NOT NULL,
    prompt_template TEXT NOT NULL,
    user_instructions TEXT,
    golden_instructions TEXT,
    metadata_required JSON,
    model_id VARCHAR(36) NOT NULL,
    evaluation_scheduler VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_evaluation_name (evaluation_name),
    INDEX idx_metrics_id (metrics_id),
    INDEX idx_prompt_id (prompt_id),
    INDEX idx_model_id (model_id),
    FOREIGN KEY (metrics_id) REFERENCES metrics(metrics_id) ON DELETE RESTRICT
);

-- Insert sample data
INSERT INTO connections (connection_type, connection_config) VALUES
('LLM', JSON_OBJECT(
    'model_name', 'gpt-3.5-turbo',
    'model_provider', 'openai',
    'api_key', 'your-api-key-here',
    'base_url', 'https://api.openai.com/v1',
    'temperature', 0.7,
    'max_tokens', 1000,
    'top_p', 1.0,
    'frequency_penalty', 0.0,
    'presence_penalty', 0.0
));

INSERT INTO metrics (metrics_name, description, metric_type, configuration) VALUES
('Accuracy', 'Measures the accuracy of responses', 'classification', JSON_OBJECT('threshold', 0.8)),
('Relevance', 'Measures relevance to user query', 'scoring', JSON_OBJECT('scale', '1-10'));
