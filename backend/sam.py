import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Handler:
    def __init__(self, checkpoint_path, model_config):
        # 自动选择设备：M4 优先使用 mps
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"--- SAM 2 正在初始化，使用设备: {self.device} ---")
        
        # 加载模型
        self.model = build_sam2(model_config, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

    def process_segmentation(self, image_path, x, y, output_path):
        """核心功能：加载图片 -> 预测遮罩 -> 抠出家具"""
        # 1. 读取并预处理图片
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # 2. 设置预测器图片 (这一步会生成图像特征)
        self.predictor.set_image(image_np)

        # 3. 根据点击坐标预测
        input_point = np.array([[x, y]])
        input_label = np.array([1]) # 1 表示选中

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False, # 只要最准确的一个
        )

        # 4. 处理遮罩并保存透明 PNG
        mask = masks[0]
        original_rgba = Image.open(image_path).convert("RGBA")
        
        # 将布尔型 mask 转为 255 的 Alpha 通道
        mask_alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_rgba.size)
        original_rgba.putalpha(mask_alpha)
        
        original_rgba.save(output_path)
        return output_path, float(scores[0])