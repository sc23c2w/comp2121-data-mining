from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI(title="Local LLM API")

# Load a small model for text classification
classifier = pipeline(
    "text-classification", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1 if not torch.cuda.is_available() else 0
)

class TextRequest(BaseModel):
    text: str

class AnalysisRequest(BaseModel):
    title: str
    content: str

@app.post("/classify")
async def classify_text(request: TextRequest):
    try:
        result = classifier(request.text)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_article")
async def analyze_article(request: AnalysisRequest):
    try:
        # Analyze title and content
        title_result = classifier(request.title)[0]
        content_result = classifier(request.content)[0]
        
        # Simple misinformation likelihood score
        # Just an example - you'd want more sophisticated analysis
        combined_score = (title_result["score"] + content_result["score"]) / 2
        
        return {
            "title_analysis": {
                "label": title_result["label"],
                "score": title_result["score"]
            },
            "content_analysis": {
                "label": content_result["label"],
                "score": content_result["score"]
            },
            "combined_score": combined_score,
            "potentially_misleading": combined_score > 0.7
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}