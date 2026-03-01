from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sam import SAM2Handler
from inpainting_cloud import CloudInpainter
from furniture_placement import FurniturePlacer
import sys
import os
import base64
import shutil
from pathlib import Path
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"))
app = FastAPI(title="Spatial Shift API", version="1.0.0")

# CORS configuration (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, should restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure paths
CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploads")

# Create necessary directories
for dir_path in [OUTPUT_DIR, UPLOAD_DIR, 
                 OUTPUT_DIR / "furniture", 
                 OUTPUT_DIR / "backgrounds", 
                 OUTPUT_DIR / "placed"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Global initialization (avoid reloading models)
sam_engine = SAM2Handler(CHECKPOINT, MODEL_CONFIG)
inpainter = CloudInpainter()
placer = FurniturePlacer()

print("✅ Spatial Shift API started successfully")


def image_to_base64(image_path: str) -> str:
    """Convert image to Base64 encoding"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "Spatial Shift API",
        "status": "running",
        "endpoints": ["/upload", "/segment", "/place_furniture", "/full_pipeline"]
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload indoor photo
    Returns: {"image_id": "xxx", "filename": "xxx.jpg"}
    """
    try:
        # Generate unique ID
        image_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        save_path = UPLOAD_DIR / f"{image_id}{file_ext}"
        
        # Save file
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "image_id": image_id,
            "filename": file.filename,
            "path": str(save_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/segment")
async def segment_furniture(
    image_id: str = Form(...),
    x: int = Form(...),
    y: int = Form(...)
):
    """
    Step 1: Segment furniture
    
    Args:
        image_id: ID of uploaded image
        x, y: User click coordinates
    
    Returns:
        {
            "furniture_mask": "base64_image",  # Transparent PNG furniture cutout
            "score": 0.98,
            "bbox": [x1, y1, x2, y2]  # Bounding box coordinates for frontend display
        }
    """
    try:
        # Find original image
        image_files = list(UPLOAD_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_path = str(image_files[0])
        
        # SAM intelligent segmentation (automatically identify related objects like chair legs, pillows, etc.)
        furniture_output = OUTPUT_DIR / "furniture" / f"{image_id}_furniture.png"
        result_path, score, related_count = sam_engine.process_segmentation_smart(
            image_path, x, y, str(furniture_output), iou_threshold=0.1
        )
        
        # TODO: Calculate bounding box (for frontend display red box)
        # Simplified here, can calculate precise bbox through mask
        from PIL import Image
        import numpy as np
        
        img = Image.open(result_path)
        alpha = np.array(img.split()[-1])
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        
        if rows.any() and cols.any():
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
        else:
            bbox = [x-50, y-50, x+50, y+50]  # Fallback solution
        
        return {
            "furniture_mask": image_to_base64(result_path),
            "furniture_path": str(furniture_output),
            "score": float(score),
            "bbox": bbox
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/remove_furniture")
async def remove_furniture(
    image_id: str = Form(...),
    furniture_mask_path: str = Form(...)
):
    """
    Step 2: Remove furniture, generate clean background
    
    Args:
        image_id: Image ID
        furniture_mask_path: Furniture mask path obtained from segmentation
    
    Returns:
        {
            "clean_background": "base64_image",  # Clean background
            "background_path": "output/backgrounds/xxx.png"
        }
    """
    try:
        # Find original image
        image_files = list(UPLOAD_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Image not found")
        
        image_path = str(image_files[0])
        
        # Inpainting to remove furniture
        background_output = OUTPUT_DIR / "backgrounds" / f"{image_id}_clean_bg.png"
        clean_bg_path = inpainter.inpaint(
            image_path,
            furniture_mask_path,
            str(background_output)
        )
        
        return {
            "clean_background": image_to_base64(clean_bg_path),
            "background_path": clean_bg_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")


@app.post("/place_furniture")
async def place_furniture(
    image_id: str = Form(...),
    furniture_mask_path: str = Form(...),
    background_path: str = Form(...),
    new_x: int = Form(...),
    new_y: int = Form(...),
    original_x: int = Form(None),
    original_y: int = Form(None),
    use_ai: bool = Form(True)
):
    """
    Step 3: Place furniture at new location
    
    Args:
        image_id: Image ID
        furniture_mask_path: Furniture cutout path
        background_path: Clean background path
        new_x, new_y: New position coordinates
        original_x, original_y: Original position coordinates (for scale estimation)
        use_ai: Whether to use AI blending (True=SD, False=Poisson)
    
    Returns:
        {
            "final_image": "base64_image",  # Final result image
            "method": "ai_blend" or "poisson_blend"
        }
    """
    try:
        output_path = OUTPUT_DIR / "placed" / f"{image_id}_placed.png"
        
        if use_ai:
            result_path = placer.ai_blend(
                furniture_mask_path,
                background_path,
                new_x, new_y,
                original_x, original_y,
                str(output_path)
            )
            method = "ai_blend"
            
            # If AI fails, result_path might be Poisson result
            if result_path is None:
                result_path = placer.poisson_blend(
                    furniture_mask_path,
                    background_path,
                    new_x, new_y,
                    original_x, original_y,
                    str(output_path)
                )
                method = "poisson_blend (AI fallback)"
        else:
            result_path = placer.poisson_blend(
                furniture_mask_path,
                background_path,
                new_x, new_y,
                original_x, original_y,
                str(output_path)
            )
            method = "poisson_blend"
        
        return {
            "final_image": image_to_base64(result_path),
            "final_path": result_path,
            "method": method
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Furniture placement failed: {str(e)}")


@app.post("/full_pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    segment_x: int = Form(...),
    segment_y: int = Form(...),
    place_x: int = Form(...),
    place_y: int = Form(...),
    use_ai: bool = Form(True)
):
    """
    Complete pipeline: upload -> segment -> remove -> place
    
    Complete all steps at once (suitable for simple scenarios)
    """
    try:
        # 1. Upload
        upload_result = await upload_image(file)
        image_id = upload_result["image_id"]
        
        # 2. Segment
        segment_result = await segment_furniture(image_id, segment_x, segment_y)
        
        # 3. Remove
        remove_result = await remove_furniture(
            image_id, 
            segment_result["furniture_path"]
        )
        
        # 4. Place
        place_result = await place_furniture(
            image_id,
            segment_result["furniture_path"],
            remove_result["background_path"],
            place_x, place_y,
            segment_x, segment_y,
            use_ai
        )
        
        return {
            "image_id": image_id,
            "furniture_bbox": segment_result["bbox"],
            "furniture_mask": segment_result["furniture_mask"],
            "clean_background": remove_result["clean_background"],
            "final_image": place_result["final_image"],
            "method": place_result["method"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)