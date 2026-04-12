import os
import sys
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Ensure the bridge directory is in sys.path
BRIDGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "geo-ai-ui", "server"))
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

# Import the core prediction logic
from predict_bridge import predict

app = FastAPI(title="Geo AI System API")

# Enable CORS for frontend development (Vite usually runs on 5173 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _to_bool(val: str) -> bool:
    return val.lower() in ("true", "1", "yes")

@app.post("/api/predict")
async def run_predict(
    image: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    api_key: Optional[str] = Form(None),
    confidence: float = Form(0.25),
    show_vegetation: str = Form("true"),
    show_ruins: str = Form("true"),
    show_structures: str = Form("true"),
    show_boulders: str = Form("true"),
    show_others: str = Form("true"),
    use_ai_insight: str = Form("false")
):
    # Create a temporary file to store the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        print(f"DEBUG: Processing image {image.filename} at {lat}, {lon}")
        # Run inference using the established bridge logic
        result = predict(
            image_path=tmp_path,
            lat=lat,
            lon=lon,
            api_key=api_key or "",
            confidence=confidence,
            class_visibility={
                "vegetation": _to_bool(show_vegetation),
                "ruins": _to_bool(show_ruins),
                "structures": _to_bool(show_structures),
                "boulders": _to_bool(show_boulders),
                "others": _to_bool(show_others),
            },
            use_ai_insight=_to_bool(use_ai_insight),
        )
        return result
    except Exception as e:
        import traceback
        err_msg = f"Inference Error: {type(e).__name__}: {str(e)}"
        stack = traceback.format_exc()
        # Log to a file we can read
        with open("backend_errors.log", "a") as f:
            f.write(f"\n--- ERROR {image.filename} ---\n{stack}\n")
        print(f"DEBUG: {err_msg}")
        raise HTTPException(status_code=500, detail={"error": err_msg, "traceback": stack})
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "geo-ai-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
