"""
本地 LaMa Inpainting 模块 - 使用 lama-cleaner
"""
import os
import numpy as np
from PIL import Image
import cv2
import torch

class LocalLamaInpainter:
    def __init__(self):
        """初始化本地 LaMa 模型"""
        print("⏳ 正在加载本地 LaMa 模型...")
        try:
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config, HDStrategy
            
            # 配置
            config = Config(
                ldm_steps=25,
                ldm_sampler='plms',
                hd_strategy=HDStrategy.ORIGINAL,
                hd_strategy_crop_margin=32,
                hd_strategy_crop_trigger_size=512,
                hd_strategy_resize_limit=512,
            )
            
            # 加载 LaMa 模型
            self.model = ModelManager(
                name="lama",
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
            self.config = config
            print("✅ LaMa 模型加载成功！")
            
        except ImportError:
            print("❌ 请先安装: pip install lama-cleaner")
            raise
    
    def create_mask_from_sam(self, sam_mask):
        """
        将 SAM 的布尔遮罩转换为 LaMa 需要的格式
        
        Args:
            sam_mask: numpy array, shape (H, W), boolean 或 0-1
        
        Returns:
            numpy array: 255=要修复的区域，0=保留的区域
        """
        mask_uint8 = (sam_mask * 255).astype(np.uint8)
        return mask_uint8
    
    def inpaint(self, image_path, mask_array, output_path):
        """
        使用 LaMa 进行本地 inpainting
        
        Args:
            image_path: 原始图片路径
            mask_array: SAM 输出的 mask (numpy array, boolean)
            output_path: 输出路径
        
        Returns:
            str: 输出文件路径
        """
        print("⏳ 正在执行 LaMa inpainting...")
        
        # 1. 读取图片
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # 2. 转换 mask
        mask_np = self.create_mask_from_sam(mask_array)
        
        # 确保 mask 和 image 尺寸一致
        if mask_np.shape != image_np.shape[:2]:
            mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))
        
        # 3. mask 需要是 3 通道
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, axis=2)
            mask_np = np.repeat(mask_np, 3, axis=2)
        
        # 4. 执行 inpainting (lama-cleaner API)
        result = self.model(image_np, mask_np, self.config)
        
        # 4. 保存结果
        result_image = Image.fromarray(result)
        result_image.save(output_path)
        
        print(f"✅ Inpainting 完成！保存到: {output_path}")
        return output_path
    
    def process_full_pipeline(self, image_path, mask_array, output_dir="output"):
        """
        完整流程：接收 SAM mask → inpainting → 返回结果
        
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
        
        # 1. 保存 mask（可选，用于调试）
        mask_image = Image.fromarray(self.create_mask_from_sam(mask_array))
        mask_path = f"{output_dir}/masks/mask_lama.png"
        mask_image.save(mask_path)
        print(f"📦 Mask 已保存: {mask_path}")
        
        # 2. 执行 inpainting
        background_path = f"{output_dir}/backgrounds/clean_background_lama.png"
        self.inpaint(image_path, mask_array, background_path)
        
        return {
            "background_clean": background_path,
            "mask": mask_path
        }


# 便捷函数
def remove_object_locally(image_path, mask_array, output_dir="output"):
    """
    本地 LaMa 一步完成对象移除
    
    Args:
        image_path: 图片路径
        mask_array: SAM mask
        output_dir: 输出目录
    
    Returns:
        dict: 结果路径
    """
    inpainter = LocalLamaInpainter()
    return inpainter.process_full_pipeline(image_path, mask_array, output_dir)
