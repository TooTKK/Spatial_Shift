from fastapi import FastAPI, HTTPException
from sam import SAM2Handler # 导入你写的工具类
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"))
app = FastAPI()

# 配置路径（确保这些文件都在正确的位置）
CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Hydra 在 sam2 包内查找

# 全局初始化单例，避免重复加载模型浪费显存
sam_engine = SAM2Handler(CHECKPOINT, MODEL_CONFIG)

@app.post("/segment")
async def segment_image(image_path: str, x: int, y: int):
    """通过本地路径触发分割接口"""
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="图片路径不存在")

    try:
        # 定义输出文件名
        output_name = f"furniture_{os.path.basename(image_path)}"
        output_path = os.path.join("output", output_name)
        os.makedirs("output", exist_ok=True)

        # 调用工具类进行处理
        result_path, score = sam_engine.process_segmentation(image_path, x, y, output_path)

        return {
            "message": "分割完成",
            "score": score,
            "output_file": result_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 启动命令：python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)