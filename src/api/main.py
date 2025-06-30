"""FastAPI application for multi-modal AI content moderation."""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
import io
import json

# Handle optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Handle optional PIL import
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# Import our models (these would need to be implemented)
# from ..models import MultiModalModel
# from ..data.preprocessors import MultiModalPreprocessor

app = FastAPI(
    title="Multi-Modal Content Moderation API",
    description="AI-powered content moderation for social media using text, images, and metadata",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ContentRequest(BaseModel):
    """Request model for content moderation."""
    text: str
    user_metadata: Dict[str, Any]
    

class ContentResponse(BaseModel):
    """Response model for content moderation."""
    prediction: str
    confidence: float
    category_scores: Dict[str, float]
    risk_level: str
    explanation: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool


# Global variables for model and preprocessor
model = None
preprocessor = None


@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup."""
    global model, preprocessor
    
    # In a real implementation, you would load the trained model here
    if TORCH_AVAILABLE:
        # model = torch.load("path/to/trained/model.pth")
        # preprocessor = load_preprocessor("path/to/preprocessor.pkl")
        print("PyTorch available - ready for full ML functionality")
    else:
        print("PyTorch not available - running in demo mode")
    
    print("Multi-modal content moderation API started successfully!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model is not None
    )


@app.post("/predict/text", response_model=ContentResponse)
async def predict_text_only(request: ContentRequest):
    """Predict content safety from text and metadata only."""
    try:
        # In a real implementation, you would:
        # 1. Preprocess the text and metadata
        # 2. Run inference with the model
        # 3. Return structured predictions
        
        # Dummy prediction for demonstration
        categories = ["safe", "hate_speech", "harassment", "spam", "inappropriate"]
        scores = np.random.dirichlet(np.ones(5))
        predicted_idx = np.argmax(scores)
        
        category_scores = {cat: float(score) for cat, score in zip(categories, scores)}
        confidence = float(scores[predicted_idx])
        
        # Determine risk level
        if confidence > 0.8:
            risk_level = "high" if predicted_idx > 0 else "low"
        elif confidence > 0.6:
            risk_level = "medium"
        else:
            risk_level = "uncertain"
        
        return ContentResponse(
            prediction=categories[predicted_idx],
            confidence=confidence,
            category_scores=category_scores,
            risk_level=risk_level,
            explanation=f"Text analysis indicates {categories[predicted_idx]} content with {confidence:.1%} confidence"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/multimodal", response_model=ContentResponse)
async def predict_multimodal(
    text: str = Form(...),
    user_metadata: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """Predict content safety from text, image, and metadata."""
    try:
        # Parse user metadata
        try:
            metadata = json.loads(user_metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid user_metadata JSON")
        
        # Process image if provided
        image_data = None
        if image:
            try:
                image_bytes = await image.read()
                image_pil = Image.open(io.BytesIO(image_bytes))
                # Convert to RGB if necessary
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')
                image_data = np.array(image_pil)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        # In a real implementation, you would:
        # 1. Preprocess all modalities
        # 2. Run multi-modal inference
        # 3. Combine predictions with fusion strategy
        
        # Dummy multi-modal prediction
        categories = ["safe", "hate_speech", "harassment", "spam", "inappropriate"]
        
        # Simulate better performance with multi-modal data
        if image_data is not None:
            # Higher confidence when image is available
            scores = np.random.dirichlet(np.ones(5) * 2)
        else:
            scores = np.random.dirichlet(np.ones(5))
        
        predicted_idx = np.argmax(scores)
        category_scores = {cat: float(score) for cat, score in zip(categories, scores)}
        confidence = float(scores[predicted_idx])
        
        # Determine risk level
        if confidence > 0.8:
            risk_level = "high" if predicted_idx > 0 else "low"
        elif confidence > 0.6:
            risk_level = "medium"
        else:
            risk_level = "uncertain"
        
        modalities_used = ["text", "metadata"]
        if image_data is not None:
            modalities_used.append("image")
        
        explanation = f"Multi-modal analysis using {', '.join(modalities_used)} indicates {categories[predicted_idx]} content"
        
        return ContentResponse(
            prediction=categories[predicted_idx],
            confidence=confidence,
            category_scores=category_scores,
            risk_level=risk_level,
            explanation=explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-modal prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "model_type": "multi_modal_transformer",
        "modalities": ["text", "image", "tabular"],
        "categories": ["safe", "hate_speech", "harassment", "spam", "inappropriate"],
        "fusion_strategy": "attention",
        "input_requirements": {
            "text": "string, max 512 characters",
            "image": "RGB image, max 5MB",
            "metadata": "JSON object with user features"
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Modal Content Moderation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
