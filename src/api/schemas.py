"""Data schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class ContentCategory(str, Enum):
    """Content moderation categories."""
    SAFE = "safe"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SPAM = "spam"
    INAPPROPRIATE = "inappropriate"


class RiskLevel(str, Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNCERTAIN = "uncertain"


class UserMetadata(BaseModel):
    """User metadata schema."""
    followers: int = Field(..., ge=0, description="Number of followers")
    following: int = Field(..., ge=0, description="Number of following")
    account_age_days: int = Field(..., ge=0, description="Account age in days")
    verification_status: bool = Field(..., description="Account verification status")
    likes: int = Field(default=0, ge=0, description="Number of likes on content")
    comments: int = Field(default=0, ge=0, description="Number of comments")
    shares: int = Field(default=0, ge=0, description="Number of shares")
    post_hour: int = Field(..., ge=0, le=23, description="Hour of posting (0-23)")
    is_weekend: bool = Field(..., description="Posted on weekend")
    has_image: bool = Field(default=False, description="Content has image")
    image_width: int = Field(default=0, ge=0, description="Image width in pixels")
    image_height: int = Field(default=0, ge=0, description="Image height in pixels")


class ContentRequest(BaseModel):
    """Request model for content moderation."""
    text: str = Field(..., max_length=1000, description="Content text")
    user_metadata: UserMetadata = Field(..., description="User metadata")
    

class ContentResponse(BaseModel):
    """Response model for content moderation."""
    prediction: ContentCategory = Field(..., description="Predicted content category")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    category_scores: Dict[str, float] = Field(..., description="Scores for each category")
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class BatchContentRequest(BaseModel):
    """Batch request model for multiple content items."""
    items: List[ContentRequest] = Field(..., max_items=100, description="Content items to process")


class BatchContentResponse(BaseModel):
    """Batch response model."""
    results: List[ContentResponse] = Field(..., description="Results for each content item")
    total_processed: int = Field(..., description="Total number of items processed")
    total_processing_time_ms: float = Field(..., description="Total processing time")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str = Field(..., description="Type of model")
    modalities: List[str] = Field(..., description="Supported modalities")
    categories: List[str] = Field(..., description="Prediction categories")
    fusion_strategy: str = Field(..., description="Multi-modal fusion strategy")
    total_parameters: Optional[int] = Field(None, description="Total model parameters")
    input_requirements: Dict[str, str] = Field(..., description="Input format requirements")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
