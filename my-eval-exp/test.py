# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from .routers import evaluations, metrics, model_config, health
from .database import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting APA Evaluation Management API...")
    try:
        # Test database connection on startup
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        logger.info("Database connection established successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise
    
    yield
    
    logger.info("Shutting down APA Evaluation Management API...")

app = FastAPI(
    title="APA Evaluation Management API",
    version="1.0.0",
    description="API for managing evaluations and metrics",
    lifespan=lifespan
)

# Include routers
app.include_router(evaluations.router, prefix="/api", tags=["evaluations"])
app.include_router(metrics.router, prefix="/api", tags=["metrics"])
app.include_router(model_config.router, prefix="/api", tags=["model-config"])
app.include_router(health.router, tags=["health"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "APA Evaluation Management API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)

# app/models.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
import json
import uuid
from datetime import datetime

def validate_uuid(uuid_string: str) -> bool:
    """Validate if a string is a valid UUID"""
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False

class EvaluationBase(BaseModel):
    evaluation_name: str 
    metrics_id: str 
    metrics_name: str 
    prompt_id: str 
    prompt_name: str 
    prompt_template: str 
    user_instructions: str 
    golden_instructions: str 
    metadata_required: str
    model_id: str 
    evaluation_scheduler: str 

    @field_validator('metadata_required')
    @classmethod
    def validate_metadata_json(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError('metadata_required must be valid JSON')
        return v

    @field_validator('metrics_id', 'prompt_id', 'model_id')
    @classmethod
    def validate_uuid_fields(cls, v):
        if not validate_uuid(v):
            raise ValueError('Must be a valid UUID')
        return v

class EvaluationCreate(EvaluationBase):
    pass

class EvaluationResponse(EvaluationBase):
    evaluation_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class MetricsCreate(BaseModel):
    evaluation_id: str
    metrics_name: str
    description: str
    metric_type: str
    configuration: Dict[str, Any]

class MetricsResponse(MetricsCreate):
    evaluation_id: str
    metrics_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class PromptTemplateUpdate(BaseModel):
    evaluation_id: str
    prompt_template: str 

class UserInstructionsUpdate(BaseModel):
    evaluation_id: str
    user_instructions: str 

class ModelConfig(BaseModel):
    model_name: str
    model_provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None

class CuratedInstructions(BaseModel):
    final_prompt: str
    model_conn_config: ModelConfig
    template_used: str
    golden_instructions: str
    variables_used: Dict[str, str]
    metadata: Dict[str, Any]

class GoldenInstructionsInput(BaseModel):
    golden_instructions: str 
    context: Optional[str] = None
    user_query: Optional[str] = None

# app/database.py
import pymysql
import ssl
import os
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Database connection context manager"""
    connection = None
    try:
        # You'll need to configure your database connection parameters
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'apa_evaluation'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
        yield connection
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise
    finally:
        if connection:
            connection.close()

# app/utils.py
import uuid
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def generate_uuid() -> str:
    """Generate a new UUID string"""
    return str(uuid.uuid4())

def parse_json_field(value: str) -> Dict[str, Any]:
    """Safely parse JSON field from database"""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        if isinstance(value, str):
            if value.startswith('{') and not value.startswith('{"'):
                return {}
            return json.loads(value)
        return {}
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON field: {value}, error: {str(e)}")
        return {}

# app/routers/evaluations.py
from fastapi import APIRouter, HTTPException, status
from typing import List
import pymysql
import json
import logging

from ..models import EvaluationCreate, EvaluationResponse, PromptTemplateUpdate, UserInstructionsUpdate
from ..database import get_db_connection
from ..utils import generate_uuid, parse_json_field, validate_uuid

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/evaluations", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation(evaluation: EvaluationCreate):
    """Create a new evaluation"""
    evaluation_id = generate_uuid()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # Check if metrics_id exists
            cursor.execute("SELECT metrics_id FROM metrics WHERE metrics_id = %s", (evaluation.metrics_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=400, detail="Metrics ID does not exist")
            
            # Insert evaluation
            insert_query = """
                INSERT INTO evaluations (
                    evaluation_id, evaluation_name, metrics_id, metrics_name, 
                    prompt_id, prompt_name, prompt_template, user_instructions, 
                    golden_instructions, metadata_required, model_id, evaluation_scheduler
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                evaluation_id, evaluation.evaluation_name, evaluation.metrics_id,
                evaluation.metrics_name, evaluation.prompt_id, evaluation.prompt_name,
                evaluation.prompt_template, evaluation.user_instructions,
                evaluation.golden_instructions, json.dumps(evaluation.metadata_required),
                evaluation.model_id, evaluation.evaluation_scheduler
            ))
            
            # Fetch the created evaluation
            cursor.execute("SELECT * FROM evaluations WHERE evaluation_id = %s", (evaluation_id,))
            result = cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                evaluation_dict = dict(zip(columns, result))
                evaluation_dict['metadata_required'] = parse_json_field(evaluation_dict['metadata_required'])
                return EvaluationResponse(**evaluation_dict)
            
        except pymysql.IntegrityError as e:
            raise HTTPException(status_code=400, detail=f"Database integrity error: {str(e)}")
        finally:
            cursor.close()

@router.get("/evaluations/{evaluation_id}/prompt-template", response_model=dict)
async def get_prompt_template(evaluation_id: str):
    """Get prompt template for an evaluation"""
    if not validate_uuid(evaluation_id):
        raise HTTPException(status_code=400, detail="Invalid evaluation ID format")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT prompt_template FROM evaluations WHERE evaluation_id = %s", 
                (evaluation_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Evaluation not found")
            
            prompt_template = result[0]
            
            return {
                "evaluation_id": evaluation_id,
                "prompt_template": prompt_template
            }
        
        finally:
            cursor.close()

@router.get("/evaluations", response_model=List[EvaluationResponse])
async def list_evaluations(skip: int = 0, limit: int = 100):
    """List all evaluations with pagination"""
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, skip)
            )
            results = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            evaluations_list = []
            
            for result in results:
                evaluation_dict = dict(zip(columns, result))
                evaluation_dict['metadata_required'] = parse_json_field(evaluation_dict['metadata_required'])
                evaluations_list.append(EvaluationResponse(**evaluation_dict))
            
            return evaluations_list
            
        finally:
            cursor.close()

@router.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(evaluation_id: str):
    """Get evaluation by ID"""
    if not validate_uuid(evaluation_id):
        raise HTTPException(status_code=400, detail="Invalid evaluation ID format")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM evaluations WHERE evaluation_id = %s", (evaluation_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Evaluation not found")
            
            columns = [desc[0] for desc in cursor.description]
            evaluation_dict = dict(zip(columns, result))
            evaluation_dict['metadata_required'] = parse_json_field(evaluation_dict['metadata_required'])
            
            return EvaluationResponse(**evaluation_dict)
            
        finally:
            cursor.close()

@router.put("/evaluations/{evaluation_id}/prompt-template")
async def update_prompt_template(evaluation_id: str, update_data: PromptTemplateUpdate):
    """Update prompt template for an evaluation"""
    if not validate_uuid(evaluation_id):
        raise HTTPException(status_code=400, detail="Invalid evaluation ID format")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # Check if evaluation exists
            cursor.execute("SELECT evaluation_id FROM evaluations WHERE evaluation_id = %s", (evaluation_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Evaluation not found")
            
            # Update prompt template
            cursor.execute(
                "UPDATE evaluations SET prompt_template = %s, updated_at = NOW() WHERE evaluation_id = %s",
                (update_data.prompt_template, evaluation_id)
            )
            
            return {"message": "Prompt template updated successfully", "evaluation_id": evaluation_id}
            
        finally:
            cursor.close()

@router.delete("/evaluations/{evaluation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(evaluation_id: str):
    """Delete evaluation by ID"""
    if not validate_uuid(evaluation_id):
        raise HTTPException(status_code=400, detail="Invalid evaluation ID format")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # Check if evaluation exists
            cursor.execute("SELECT evaluation_id FROM evaluations WHERE evaluation_id = %s", (evaluation_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Evaluation not found")
            
            # Delete evaluation
            cursor.execute("DELETE FROM evaluations WHERE evaluation_id = %s", (evaluation_id,))
            
        finally:
            cursor.close()

# app/routers/metrics.py
from fastapi import APIRouter, HTTPException, status
from typing import List
import logging

from ..models import MetricsCreate, MetricsResponse
from ..database import get_db_connection
from ..utils import generate_uuid, parse_json_field, validate_uuid

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/metrics", response_model=List[MetricsResponse])
async def list_metrics(skip: int = 0, limit: int = 100):
    """List all metrics with pagination"""
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT * FROM metrics ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, skip)
            )
            results = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            metrics_list = []
            
            for result in results:
                metrics_dict = dict(zip(columns, result))
                metrics_dict['configuration'] = parse_json_field(metrics_dict['configuration'])
                metrics_list.append(MetricsResponse(**metrics_dict))
            
            return metrics_list
            
        finally:
            cursor.close()

@router.post("/metrics", response_model=MetricsResponse, status_code=status.HTTP_201_CREATED)
async def create_metrics(metrics: MetricsCreate):
    """Create new metrics"""
    metrics_id = generate_uuid()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            insert_query = """
                INSERT INTO metrics (
                    metrics_id, evaluation_id, metrics_name, description, 
                    metric_type, configuration
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                metrics_id, metrics.evaluation_id, metrics.metrics_name,
                metrics.description, metrics.metric_type, 
                json.dumps(metrics.configuration)
            ))
            
            # Fetch the created metrics
            cursor.execute("SELECT * FROM metrics WHERE metrics_id = %s", (metrics_id,))
            result = cursor.fetchone()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                metrics_dict = dict(zip(columns, result))
                metrics_dict['configuration'] = parse_json_field(metrics_dict['configuration'])
                return MetricsResponse(**metrics_dict)
            
        finally:
            cursor.close()

@router.get("/metrics/{metrics_id}", response_model=MetricsResponse)
async def get_metrics(metrics_id: str):
    """Get metrics by ID"""
    if not validate_uuid(metrics_id):
        raise HTTPException(status_code=400, detail="Invalid metrics ID format")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM metrics WHERE metrics_id = %s", (metrics_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Metrics not found")
            
            columns = [desc[0] for desc in cursor.description]
            metrics_dict = dict(zip(columns, result))
            metrics_dict['configuration'] = parse_json_field(metrics_dict['configuration'])
            
            return MetricsResponse(**metrics_dict)
            
        finally:
            cursor.close()

@router.delete("/metrics/{metrics_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_metrics(metrics_id: str):
    """Delete metrics by ID"""
    if not validate_uuid(metrics_id):
        raise HTTPException(status_code=400, detail="Invalid metrics ID format")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # Check if metrics exists
            cursor.execute("SELECT metrics_id FROM metrics WHERE metrics_id = %s", (metrics_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Metrics not found")
            
            # Check if metrics is being used by any evaluations
            cursor.execute("SELECT COUNT(*) FROM evaluations WHERE metrics_id = %s", (metrics_id,))
            count = cursor.fetchone()[0]
            if count > 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Cannot delete metrics. It is being used by {count} evaluation(s)"
                )
            
            # Delete metrics
            cursor.execute("DELETE FROM metrics WHERE metrics_id = %s", (metrics_id,))
            
        finally:
            cursor.close()

# app/routers/model_config.py
from fastapi import APIRouter, HTTPException
import json
import logging

from ..models import ModelConfig
from ..database import get_db_connection

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/model-config", response_model=ModelConfig)
async def get_model_config():
    """
    Retrieve model configuration from connections table where connection_type='LLM'
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Query the connections table for LLM type connections
            cursor.execute("""
                SELECT connection_config 
                FROM connections 
                WHERE connection_type = 'LLM' 
                ORDER BY created_at DESC 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                raise HTTPException(
                    status_code=404, 
                    detail="No LLM connection configuration found"
                )
            
            # Parse the connection_config JSON
            config_json = result[0]
            if isinstance(config_json, str):
                config = json.loads(config_json)
            else:
                config = config_json
            
            # Map the configuration to ModelConfig structure
            model_config = ModelConfig(
                model_name=config.get('model_name', ''),
                model_provider=config.get('model_provider', ''),
                api_key=config.get('api_key'),
                base_url=config.get('base_url'),
                temperature=config.get('temperature'),
                max_tokens=config.get('max_tokens'),
                top_p=config.get('top_p'),
                frequency_penalty=config.get('frequency_penalty'),
                presence_penalty=config.get('presence_penalty'),
                parameters=config.get('parameters', {})
            )
            
            return model_config
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid JSON in connection config")
    except Exception as e:
        logger.error(f"Error retrieving model config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model config: {str(e)}")

@router.put("/model-config", response_model=ModelConfig)
async def update_model_config(model_config: ModelConfig):
    """Update model configuration in connections table"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Convert ModelConfig to dict for JSON serialization
            config_dict = model_config.model_dump()
            config_json = json.dumps(config_dict)
            
            # Update the connection configuration
            cursor.execute("""
                UPDATE connections 
                SET connection_config = %s, updated_at = NOW() 
                WHERE connection_type = 'LLM'
            """)
            
            if cursor.rowcount == 0:
                # If no existing LLM connection, create one
                cursor.execute("""
                    INSERT INTO connections (connection_type, connection_config)
                    VALUES ('LLM', %s)
                """, (config_json,))
            
            cursor.close()
            
            return model_config
            
    except Exception as e:
        logger.error(f"Error updating model config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating model config: {str(e)}")

# app/routers/health.py
from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging

from ..database import get_db_connection

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        return {
            "status": "healthy", 
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

# app/__init__.py
# Empty file to make app a package

# requirements.txt
"""
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pymysql==1.1.0
python-multipart==0.0.6
"""
