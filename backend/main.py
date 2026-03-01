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

# CORS配置（允许前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置路径
CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploads")

# 创建必要目录
for dir_path in [OUTPUT_DIR, UPLOAD_DIR, 
                 OUTPUT_DIR / "furniture", 
                 OUTPUT_DIR / "backgrounds", 
                 OUTPUT_DIR / "placed"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 全局初始化（避免重复加载模型）
sam_engine = SAM2Handler(CHECKPOINT, MODEL_CONFIG)
inpainter = CloudInpainter()
placer = FurniturePlacer()

print("✅ Spatial Shift API 启动成功")


def image_to_base64(image_path: str) -> str:
    """将图片转换为Base64编码"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "Spatial Shift API",
        "status": "running",
        "endpoints": ["/upload", "/segment", "/place_furniture", "/full_pipeline"]
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    上传室内照片
    Returns: {"image_id": "xxx", "filename": "xxx.jpg"}
    """
    try:
        # 生成唯一ID
        image_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        save_path = UPLOAD_DIR / f"{image_id}{file_ext}"
        
        # 保存文件
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "image_id": image_id,
            "filename": file.filename,
            "path": str(save_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/segment")
async def segment_furniture(
    image_id: str = Form(...),
    x: int = Form(...),
    y: int = Form(...)
):
    """
    步骤1: 分割家具
    
    Args:
        image_id: 上传图片的ID
        x, y: 用户点击的坐标
    
    Returns:
        {
            "furniture_mask": "base64_image",  # 透明PNG家具抠图
            "score": 0.98,
            "bbox": [x1, y1, x2, y2]  # 边界框坐标，供前端显示
        }
    """
    try:
        # 查找原图
        image_files = list(UPLOAD_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="图片不存在")
        
        image_path = str(image_files[0])
        
        # SAM智能分割（自动识别相关物体，如椅子腿、枕头等）
        furniture_output = OUTPUT_DIR / "furniture" / f"{image_id}_furniture.png"
        result_path, score, related_count = sam_engine.process_segmentation_smart(
            image_path, x, y, str(furniture_output), iou_threshold=0.1
        )
        
        # TODO: 计算边界框（用于前端显示红框）
        # 这里简化处理，可以通过mask计算精确bbox
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
            bbox = [x-50, y-50, x+50, y+50]  # 降级方案
        
        return {
            "furniture_mask": image_to_base64(result_path),
            "furniture_path": str(furniture_output),
            "score": float(score),
            "bbox": bbox
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分割失败: {str(e)}")


@app.post("/remove_furniture")
async def remove_furniture(
    image_id: str = Form(...),
    furniture_mask_path: str = Form(...)
):
    """
    步骤2: 移除家具，生成干净背景
    
    Args:
        image_id: 图片ID
        furniture_mask_path: 分割得到的家具mask路径
    
    Returns:
        {
            "clean_background": "base64_image",  # 干净背景
            "background_path": "output/backgrounds/xxx.png"
        }
    """
    try:
        # 查找原图
        image_files = list(UPLOAD_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="图片不存在")
        
        image_path = str(image_files[0])
        
        # Inpainting移除家具
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
        raise HTTPException(status_code=500, detail=f"背景移除失败: {str(e)}")


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
    步骤3: 将家具放置到新位置
    
    Args:
        image_id: 图片ID
        furniture_mask_path: 家具抠图路径
        background_path: 干净背景路径
        new_x, new_y: 新位置坐标
        original_x, original_y: 原位置坐标（用于估算缩放）
        use_ai: 是否使用AI融合（True=SD, False=Poisson）
    
    Returns:
        {
            "final_image": "base64_image",  # 最终效果图
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
            
            # 如果AI失败，result_path可能是Poisson的结果
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
        raise HTTPException(status_code=500, detail=f"家具放置失败: {str(e)}")


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
    完整流程：上传 -> 分割 -> 移除 -> 放置
    
    一次性完成所有步骤（适合简单场景）
    """
    try:
        # 1. 上传
        upload_result = await upload_image(file)
        image_id = upload_result["image_id"]
        
        # 2. 分割
        segment_result = await segment_furniture(image_id, segment_x, segment_y)
        
        # 3. 移除
        remove_result = await remove_furniture(
            image_id, 
            segment_result["furniture_path"]
        )
        
        # 4. 放置
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
        raise HTTPException(status_code=500, detail=f"流程失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)