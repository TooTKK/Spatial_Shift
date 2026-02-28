"""
Inpainting 模块 - 使用 Replicate API 进行背景消除
"""
import os
import requests
import replicate
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class InpaintingHandler:
    def __init__(self):
        """初始化 Inpainting 处理器"""
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("❌ 未找到 REPLICATE_API_TOKEN，请检查 .env 文件")
        
        # 设置 Replicate client
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        print("✅ Replicate API 初始化成功")
    
    def create_mask_from_sam(self, sam_mask):
        """
        将 SAM 的布尔遮罩转换为 Replicate 需要的格式
        
        Args:
            sam_mask: numpy array, shape (H, W), boolean 或 0-1
        
        Returns:
            PIL Image: 白色(255)=要修复的区域，黑色(0)=保留的区域
        """
        # 确保是 0-255 的灰度图
        mask_uint8 = (sam_mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8, mode='L')
        return mask_image
    
    def inpaint_with_replicate(self, image_path, mask_path, output_path, 
                               model="google/imagen-4"):
        """
        使用 Replicate API 进行 inpainting
        
        可选模型（按优先级）：
        1. google-research/imagen-3-inpainting (Imagen 3)
        2. bytedance/sdxl-lightning-4step-inpainting (快速)
        3. stability-ai/stable-diffusion-inpainting (经典)
        
        Args:
            image_path: 原始图片路径
            mask_path: 遮罩图片路径（白色=要修复）
            output_path: 输出路径
            model: Replicate 模型名称
        
        Returns:
            str: 输出文件路径
        """
        print(f"⏳ 正在调用 Replicate API ({model})...")
        
        try:
            # 调用 Replicate API
            output = replicate.run(
                model,
                input={
                    "image": open(image_path, "rb"),
                    "mask": open(mask_path, "rb"),
                    "prompt": "clean empty room, interior background, no furniture",
                    "negative_prompt": "object, furniture, item"
                }
            )
            
            # output 是一个 URL，下载结果
            print("⏳ 下载结果...")
            response = requests.get(output)
            response.raise_for_status()
            
            # 保存结果
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"✅ Inpainting 完成！保存到: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Replicate API 调用失败: {str(e)}")
            raise
    
    def process_full_pipeline(self, image_path, mask_array, output_dir="output"):
        """
        完整流程：接收 SAM mask → 转换格式 → inpainting → 返回结果
        
        Args:
            image_path: 原始图片路径
            mask_array: SAM 输出的 mask (numpy array)
            output_dir: 输出目录
        
        Returns:
            dict: {
                "background_clean": 清理后的背景图路径,
                "mask": 遮罩路径
            }
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/masks", exist_ok=True)
        os.makedirs(f"{output_dir}/backgrounds", exist_ok=True)
        
        # 1. 转换 mask 格式
        mask_image = self.create_mask_from_sam(mask_array)
        mask_path = f"{output_dir}/masks/mask_temp.png"
        mask_image.save(mask_path)
        print(f"✅ Mask 已保存: {mask_path}")
        
        # 2. 执行 inpainting
        background_path = f"{output_dir}/backgrounds/clean_background.png"
        self.inpaint_with_replicate(image_path, mask_path, background_path)
        
        return {
            "background_clean": background_path,
            "mask": mask_path
        }


# 便捷函数
def remove_object_from_image(image_path, mask_array, output_dir="output"):
    """
    便捷函数：一步完成对象移除
    
    Args:
        image_path: 图片路径
        mask_array: SAM mask
        output_dir: 输出目录
    
    Returns:
        dict: 结果路径
    """
    handler = InpaintingHandler()
    return handler.process_full_pipeline(image_path, mask_array, output_dir)
