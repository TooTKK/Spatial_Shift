"""
简化版服务器 - 使用豆包 API
流程：上传 → SAM 分割 → 豆包 Inpainting → 返回结果
"""
import os
import sys
import base64
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加 SAM2 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"))

# 导入模块
from sam import SAM2Handler
from inpainting_doubao import DoubaoInpainter

# 初始化 FastAPI
app = FastAPI(title="Spatial Shift - Doubao API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录设置
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "masks").mkdir(exist_ok=True)
(OUTPUT_DIR / "furniture").mkdir(exist_ok=True)
(OUTPUT_DIR / "backgrounds").mkdir(exist_ok=True)

# 初始化模型
print("\n" + "="*50)
print("🚀 Spatial Shift 简化版启动")
print("="*50 + "\n")

# SAM 初始化
sam_handler = SAM2Handler(
    checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
    model_config="configs/sam2.1/sam2.1_hiera_l.yaml"
)

# 豆包 Inpainting 初始化
try:
    doubao_inpainter = DoubaoInpainter()
except Exception as e:
    print(f"⚠️  豆包 API 初始化失败: {e}")
    print("   请确保在 .env 文件中设置了 ARK_API_KEY")
    doubao_inpainter = None

print("\n✅ Spatial Shift 简化版启动成功！")
print("   - SAM 2.1 分割")
print("   - 豆包 API Inpainting")
print(f"   - 服务地址: http://0.0.0.0:8001\n")


@app.get("/")
def root():
    """健康检查"""
    return {
        "status": "running",
        "version": "doubao-simplified",
        "features": ["sam-segmentation", "doubao-inpainting"]
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片
    """
    try:
        # 生成唯一 ID
        image_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix or ".jpg"
        file_path = UPLOAD_DIR / f"{image_id}{file_ext}"
        
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"📁 图片已上传: {file_path}")
        return {"image_id": image_id, "filename": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/process")
async def process_image(
    image_id: str = Form(...),
    click_x: int = Form(...),
    click_y: int = Form(...),
    place_x: int = Form(...),
    place_y: int = Form(...)
):
    """
    完整流程：SAM 分割 → 豆包生成移动效果图
    
    Args:
        image_id: 图片 ID
        click_x: 点击家具的 X 坐标
        click_y: 点击家具的 Y 坐标
        place_x: 目标位置的 X 坐标
        place_y: 目标位置的 Y 坐标
    
    Returns:
        {
            "furniture_mask": "base64_image",  # 分割的 mask
            "final_image": "base64_image"  # 豆包生成的移动效果图
        }
    """
    try:
        # 1. 查找原图
        image_files = list(UPLOAD_DIR.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="图片不存在")
        
        image_path = str(image_files[0])
        print(f"\n{'='*50}")
        print(f"🎯 开始处理图片: {image_path}")
        print(f"   家具位置: ({click_x}, {click_y})")
        print(f"   目标位置: ({place_x}, {place_y})")
        print(f"{'='*50}\n")
        
        # 2. SAM 分割
        furniture_output = OUTPUT_DIR / "furniture" / f"{image_id}_furniture.png"
        result_path, score, related_count = sam_handler.process_segmentation_smart(
            image_path, click_x, click_y, str(furniture_output)
        )
        print(f"✅ SAM 分割完成，置信度: {score:.3f}, 相关物体: {related_count}")
        
        # 3. 提取 mask
        furniture_img = Image.open(result_path)
        if furniture_img.mode == 'RGBA':
            mask_array = np.array(furniture_img.split()[-1])
        else:
            mask_array = np.array(furniture_img.convert('L'))
        
        # 保存 mask
        mask_output = OUTPUT_DIR / "masks" / f"{image_id}_mask.png"
        mask_img = Image.fromarray(mask_array, mode='L')
        mask_img.save(mask_output)
        print(f"📦 Mask 已保存: {mask_output}")
        
        # 4. 豆包生成移动效果图
        if not doubao_inpainter:
            raise HTTPException(status_code=500, detail="豆包 API 未初始化")
        
        final_output = OUTPUT_DIR / "backgrounds" / f"{image_id}_final.png"
        final_path = doubao_inpainter.inpaint(
            image_path,
            str(mask_output),
            str(final_output),
            target_x=place_x,
            target_y=place_y
        )
        
        # 5. 转为 base64 返回
        def image_to_base64(img_path):
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        
        mask_b64 = image_to_base64(mask_output)
        final_b64 = image_to_base64(final_path)
        
        print(f"\n✅ 处理完成！")
        print(f"{'='*50}\n")
        
        return {
            "image_id": image_id,
            "furniture_mask": mask_b64,
            "final_image": final_b64,
            "confidence": float(score),
            "related_objects": related_count,
            "target_position": {"x": place_x, "y": place_y}
        }
    
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("origin:app", host="0.0.0.0", port=8001, reload=True)
