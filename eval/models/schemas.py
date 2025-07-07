from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
from utils.helpers import validate_uuid

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
