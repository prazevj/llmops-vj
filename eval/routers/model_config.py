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
        logger.error(f"Error retrieving model config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model config: {str(e)}")

@router.post("/apa-golden-instructions", response_model=CuratedInstructions)
async def derive_curated_instructions(instructions_input: GoldenInstructionsInput):
    """
    Derive curated instructions by combining prompt template and golden instructions.
    This endpoint constructs the final prompt using:
    1. Default prompt template from prompt-template endpoint
    2. Golden instructions provided by user
    3. Model configuration from model-config endpoint
    """
    try:
        # Get the prompt template - Note: This would need an evaluation_id in practice
        # For now, we'll use a default template or handle this differently
        # template_response = await get_prompt_template(evaluation_id)
        # prompt_template = template_response["prompt_template"]
        
        # Placeholder template - in practice this should come from an evaluation
        prompt_template = """
        Instructions: {golden_instructions}
        
        Context: {context}
        
        User Query: {user_query}
        
        Please provide your response based on the above instructions and context.
        """
        
        # Get the model configuration
        model_conn_config = await get_model_config()
        
        # Prepare variables for template substitution
        variables_used = {
            "golden_instructions": instructions_input.golden_instructions,
            "context": instructions_input.context or "No specific context provided",
            "user_query": instructions_input.user_query or "Awaiting user query"
        }
        
        # Construct the final prompt by replacing placeholders
        final_prompt = prompt_template.format(**variables_used)
        
        # Create metadata about the prompt construction
        metadata = {
            "prompt_length": len(final_prompt),
            "template_variables": list(variables_used.keys()),
            "construction_timestamp": datetime.utcnow().isoformat(),
            "model_provider": model_conn_config.model_provider,
            "model_name": model_conn_config.model_name
        }
        
        # Return the curated instructions
        return CuratedInstructions(
            final_prompt=final_prompt,
            model_conn_config=model_conn_config,
            template_used=prompt_template,
            golden_instructions=instructions_input.golden_instructions,
            variables_used=variables_used,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error deriving curated instructions: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error deriving curated instructions: {str(e)}"
        )
