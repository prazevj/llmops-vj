import pymysql
import ssl
import os
from contextlib import contextmanager
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', ''),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', ''),
    'charset': 'utf8mb4',
    'ssl_ca': os.getenv('DB_SSL_CA', ''),
    'ssl_disabled': False,
    'ssl_verify_cert': True,
    'ssl_verify_identity': True
}

@contextmanager
def get_db_connection():
    """Database connection context manager with SSL support"""
    connection = None
    try:
        # Only configure SSL if SSL CA is provided
        ssl_context = None
        if DB_CONFIG.get('ssl_ca') and os.path.exists(DB_CONFIG['ssl_ca']):
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            ssl_context.load_verify_locations(DB_CONFIG['ssl_ca'])
            
            connection = pymysql.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database'],
                charset=DB_CONFIG['charset'],
                ssl_ca=DB_CONFIG['ssl_ca'],
                ssl_disabled=DB_CONFIG['ssl_disabled'],
                ssl_verify_cert=DB_CONFIG['ssl_verify_cert'],
                ssl_verify_identity=DB_CONFIG['ssl_verify_identity'],
                autocommit=True
            )
        else:
            # Connect without SSL if no SSL CA is provided
            connection = pymysql.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database'],
                charset=DB_CONFIG['charset'],
                autocommit=True
            )
        
        yield connection
        
    except pymysql.Error as e:
        logger.error(f"Database error: {str(e)}")
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
    finally:
        if connection:
            connection.close()

def init_database():
    """Initialize database tables if they don't exist"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create metrics table first (referenced by evaluations)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    metrics_id VARCHAR(36) PRIMARY KEY,
                    metrics_name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT,
                    metric_type ENUM('accuracy', 'precision', 'recall', 'f1', 'custom') NOT NULL,
                    configuration JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_metrics_name (metrics_name),
                    INDEX idx_metric_type (metric_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id VARCHAR(36) PRIMARY KEY,
                    evaluation_name VARCHAR(255) NOT NULL,
                    metrics_id VARCHAR(36) NOT NULL,
                    metrics_name VARCHAR(255) NOT NULL,
                    prompt_id VARCHAR(36) NOT NULL,
                    prompt_name VARCHAR(255) NOT NULL,
                    prompt_template TEXT NOT NULL,
                    user_instructions TEXT NOT NULL,
                    golden_instructions TEXT NOT NULL,
                    metadata_required JSON,
                    model_id VARCHAR(36) NOT NULL,
                    evaluation_scheduler VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_evaluation_name (evaluation_name),
                    INDEX idx_metrics_id (metrics_id),
                    INDEX idx_prompt_id (prompt_id),
                    INDEX idx_model_id (model_id),
                    FOREIGN KEY (metrics_id) REFERENCES metrics(metrics_id) ON DELETE RESTRICT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            cursor.close()
            logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
