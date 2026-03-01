import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2Handler:
    def __init__(self, checkpoint_path, model_config):
        # 自动选择设备：M4 优先使用 mps
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"--- SAM 2 正在初始化，使用设备: {self.device} ---")
        
        # 加载模型
        self.model = build_sam2(model_config, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        
        # 初始化自动分割器（用于智能识别）
        self.auto_generator = SAM2AutomaticMaskGenerator(
            self.model,
            points_per_side=16,  # 降低采样点，加快速度
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85
        )
        print("✅ 智能识别模式已启用")

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
    
    def process_segmentation_smart(self, image_path, x, y, output_path, iou_threshold=0.1):
        """
        智能分割：点击书桌后，自动识别桌上所有物品
        
        Args:
            image_path: 图片路径
            x, y: 点击坐标
            output_path: 输出路径
            iou_threshold: IoU阈值（默认0.1，物体与主物体有10%重叠就算相关）
        
        Returns:
            (output_path, score, related_count)
        """
        print(f"🧠 智能识别模式：分析点击位置 ({x}, {y})...")
        
        # 1. 读取图片
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        self.predictor.set_image(image_np)
        
        # 2. 识别主物体（点击的书桌）
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        main_masks, main_scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        main_mask = main_masks[0].astype(bool)  # 确保是bool类型
        main_bbox = self._get_bbox_from_mask(main_mask)
        print(f"   主物体识别完成，边界: {main_bbox}")
        
        # 3. 自动识别图片中所有物体
        print("   🔍 扫描所有物体...")
        all_masks = self.auto_generator.generate(image_np)
        print(f"   发现 {len(all_masks)} 个物体")
        
        # 4. 找出与主物体重叠的相关物体
        related_masks = [main_mask]
        for mask_data in all_masks:
            mask = mask_data['segmentation']
            bbox = self._get_bbox_from_mask(mask)
            
            # 计算与主物体的IoU
            iou = self._calculate_iou(bbox, main_bbox)
            
            if iou > iou_threshold:
                # 确保mask是bool类型
                mask_bool = mask.astype(bool) if mask.dtype != bool else mask
                related_masks.append(mask_bool)
                print(f"   ✓ 发现相关物体，IoU={iou:.2f}, bbox={bbox}")
        
        print(f"   📦 共识别到 {len(related_masks)} 个相关物体（含主物体）")
        
        # 5. 合并所有mask（使用numpy的逻辑或）
        final_mask = np.zeros_like(main_mask, dtype=bool)
        for mask in related_masks:
            # 确保尺寸一致
            if mask.shape != final_mask.shape:
                continue
            final_mask = np.logical_or(final_mask, mask)
        
        # 5.5. 智能移除地板（如果存在）
        final_mask = self._remove_floor_from_mask(final_mask, image_np)
        
        # 6. 保存透明PNG
        original_rgba = Image.open(image_path).convert("RGBA")
        mask_alpha = Image.fromarray((final_mask * 255).astype(np.uint8)).resize(original_rgba.size)
        original_rgba.putalpha(mask_alpha)
        original_rgba.save(output_path)
        
        return output_path, float(main_scores[0]), len(related_masks) - 1  # 返回额外识别的物体数
    
    def _get_bbox_from_mask(self, mask):
        """从mask获取边界框 [x1, y1, x2, y2]"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            return [int(x1), int(y1), int(x2), int(y2)]
        return [0, 0, 0, 0]
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _remove_floor_from_mask(self, mask, image_np):
        """
        智能移除地板区域（如果存在）
        
        策略：
        1. 只分析底部 30% 区域
        2. 检测是否有大面积横向连续区域（地板特征）
        3. 通过形状（宽高比）判断是否为地板
        4. 使用颜色过滤移除相似颜色的底部区域
        5. 保护机制：底部区域太小则不处理
        
        Args:
            mask: 布尔型 mask 数组
            image_np: 原始图像数组（用于颜色分析）
        
        Returns:
            处理后的 mask
        """
        from scipy import ndimage
        
        height, width = mask.shape
        
        # 1. 分析底部 30% 区域
        bottom_threshold = int(height * 0.7)  # 从 70% 高度开始往下
        bottom_mask = mask.copy()
        bottom_mask[:bottom_threshold, :] = False  # 只保留底部
        
        bottom_area = np.sum(bottom_mask)
        total_area = np.sum(mask)
        
        # 保护机制：如果底部区域太小，说明没地板
        if total_area == 0 or bottom_area / total_area < 0.15:
            print("   ℹ️  底部区域较小，无需处理")
            return mask
        
        print(f"   🔍 检测到底部区域占比 {bottom_area/total_area*100:.1f}%，分析地板...")
        
        # 2. 检查底部是否是"横向大面积连续"（地板特征）
        # 使用连通域分析
        labeled_bottom, num_features = ndimage.label(bottom_mask)
        
        if num_features == 0:
            return mask
        
        # 找到最大的连通域（可能是地板）
        largest_component_size = 0
        largest_component_label = 0
        for i in range(1, num_features + 1):
            size = np.sum(labeled_bottom == i)
            if size > largest_component_size:
                largest_component_size = size
                largest_component_label = i
        
        # 如果最大连通域占底部区域不到 50%，说明不是地板
        if largest_component_size / bottom_area < 0.5:
            print("   ℹ️  底部无大面积连续区域，无需处理")
            return mask
        
        # 3. 分析最大连通域的形状（宽高比）
        largest_component_mask = (labeled_bottom == largest_component_label)
        rows = np.any(largest_component_mask, axis=1)
        cols = np.any(largest_component_mask, axis=0)
        
        if not (rows.any() and cols.any()):
            return mask
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        component_height = y2 - y1 + 1
        component_width = x2 - x1 + 1
        aspect_ratio = component_width / component_height if component_height > 0 else 0
        
        # 地板通常是扁平的（宽高比 > 2）
        if aspect_ratio < 2:
            print(f"   ℹ️  底部区域宽高比 {aspect_ratio:.2f}，不像地板")
            return mask
        
        print(f"   ✅ 检测到地板特征（宽高比 {aspect_ratio:.2f}），开始颜色过滤...")
        
        # 4. 颜色分析：提取底部连通域的主色调
        bottom_region_pixels = image_np[largest_component_mask]
        
        if len(bottom_region_pixels) == 0:
            return mask
        
        # 计算主色调（中位数，比均值更鲁棒）
        main_color = np.median(bottom_region_pixels, axis=0)
        
        # 5. 移除与主色调相似的底部区域
        # 计算每个像素与主色调的颜色距离（欧式距离）
        color_distance = np.sqrt(np.sum((image_np - main_color) ** 2, axis=2))
        
        # 设置阈值（欧式距离 < 50 认为是相似颜色）
        similar_color_mask = color_distance < 50
        
        # 只在底部区域应用颜色过滤
        floor_to_remove = largest_component_mask & similar_color_mask
        
        # 6. 从原始 mask 中移除地板
        cleaned_mask = mask & ~floor_to_remove
        
        removed_area = np.sum(floor_to_remove)
        print(f"   ✂️  移除地板区域：{removed_area} 像素 ({removed_area/total_area*100:.1f}%)")
        
        return cleaned_mask