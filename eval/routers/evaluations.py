from fastapi import APIRouter, HTTPException, status
from typing import List
import pymysql
import json
from datetime import datetime
import logging

from config.database import get_db_connection
from models.schemas import (
    EvaluationCreate, EvaluationResponse, PromptTemplateUpdate, 
    UserInstructionsUpdate, CuratedInstructions, GoldenInstructionsInput
)
from utils.helpers import generate_uuid, validate_uuid, parse_json_field
from routers.model_config import get_model_config

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
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

@router.get("", response_model=List[EvaluationResponse])
async def list_evaluations(skip: int = 0, limit: int = 100):
    """List all evaluations with pagination"""
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT evaluation_id, evaluation_name, metrics_id, metrics_name, prompt_id, prompt_name, "
                "prompt_template, user_instructions, metadata_required, golden_instructions, model_id, evaluation_scheduler, "
                "created_at, updated_at FROM evaluations ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, skip)
            )
            results = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            evaluations = []
            
            for result in results:
                evaluation_dict = dict(zip(columns, result))
                # Set metadata_required to empty dict since we're not selecting it
                evaluation_dict['metadata_required'] = '{}' 
                evaluations.append(EvaluationResponse(**evaluation_dict))
            
            return evaluations
            
        finally:
            cursor.close()

@router.get("/{evaluation_id}", response_model=EvaluationResponse)
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

@router.get("/{evaluation_id}/prompt-template", response_model=dict)
async def get_prompt_template(evaluation_id: str):
    """Get prompt template for an evaluation"""
    if not validate_uuid(evaluation_id):
        raise HTTPException(status_code=400, detail="Invalid evaluation ID format")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # Get prompt template for the evaluation
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

@router.put("/{evaluation_id}/prompt-template", response_model=dict)
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
                "UPDATE evaluations SET prompt_template = %s, updated_at = CURRENT_TIMESTAMP WHERE evaluation_id = %s",
                (update_data.prompt_template, evaluation_id)
            )
            
            return {"message": "Prompt template updated successfully", "evaluation_id": evaluation_id}
            
        finally:
            cursor.close()

@router.put("/{evaluation_id}/user-instructions", response_model=dict)
async def update_user_instructions(evaluation_id: str, update_data: UserInstructionsUpdate):
    """Update user instructions for an evaluation"""
    if not validate_uuid(evaluation_id):
        raise HTTPException(status_code=400, detail="Invalid evaluation ID format")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # Check if evaluation exists
            cursor.execute("SELECT evaluation_id FROM evaluations WHERE evaluation_id = %s", (evaluation_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Evaluation not found")
            
            # Update user instructions
            cursor.execute(
                "UPDATE evaluations SET user_instructions = %s, updated_at = CURRENT_TIMESTAMP WHERE evaluation_id = %s",
                (update_data.user_instructions, evaluation_id)
            )
            
            return {"message": "User instructions updated successfully", "evaluation_id": evaluation_id}
            
        finally:
            cursor.close()

@router.delete("/{evaluation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(evaluation_id: str):
    """Delete an evaluation by ID"""
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
