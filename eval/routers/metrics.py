from fastapi import APIRouter, HTTPException, status
from typing import List
import pymysql
import json
import logging

from config.database import get_db_connection
from models.schemas import MetricsCreate, MetricsResponse
from utils.helpers import generate_uuid, validate_uuid, parse_json_field

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("", response_model=MetricsResponse, status_code=status.HTTP_201_CREATED)
async def create_metrics(metrics: MetricsCreate):
    """Create new metrics"""
    metrics_id = generate_uuid()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            insert_query = """
                INSERT INTO metrics (metrics_id, metrics_name, description, metric_type, configuration)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                metrics_id, 
                metrics.metrics_name, 
                metrics.description,
                metrics.metric_type, 
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
                
        except pymysql.IntegrityError as e:
            if "Duplicate entry" in str(e):
                raise HTTPException(status_code=400, detail="Metrics name already exists")
            raise HTTPException(status_code=400, detail=f"Database integrity error: {str(e)}")
        finally:
            cursor.close()

@router.get("", response_model=List[MetricsResponse])
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

@router.get("/{metrics_id}", response_model=MetricsResponse)
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

@router.delete("/{metrics_id}", status_code=status.HTTP_204_NO_CONTENT)
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
