from fastapi import APIRouter, HTTPException
import json
from datetime import datetime
import logging

from config.database import get_db_connection
from models.schemas import ModelConfig, CuratedInstructions, GoldenInstructionsInput
from routers.evaluations import get_prompt_template

logger = logging.getLogger(__name__)

router = APIRouter()

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
        logger.
