import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2Handler:
    def __init__(self, checkpoint_path, model_config):
        # Auto select device: M4 prioritizes mps
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"--- SAM 2 initializing, using device: {self.device} ---")
        
        # Load model
        self.model = build_sam2(model_config, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        
        # Initialize automatic segmenter (for intelligent recognition)
        self.auto_generator = SAM2AutomaticMaskGenerator(
            self.model,
            points_per_side=16,  # Reduce sampling points to speed up
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85
        )
        print("✅ Intelligent recognition mode enabled")

    def process_segmentation(self, image_path, x, y, output_path):
        """Core functionality: Load image -> Predict mask -> Extract furniture"""
        # 1. Read and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # 2. Set predictor image (this step generates image features)
        self.predictor.set_image(image_np)

        # 3. Predict based on click coordinates
        input_point = np.array([[x, y]])
        input_label = np.array([1]) # 1 means selected

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False, # Only want the most accurate one
        )

        # 4. Process mask and save transparent PNG
        mask = masks[0]
        original_rgba = Image.open(image_path).convert("RGBA")
        
        # Convert boolean mask to 255 Alpha channel
        mask_alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_rgba.size)
        original_rgba.putalpha(mask_alpha)
        
        original_rgba.save(output_path)
        return output_path, float(scores[0])
    
    def process_segmentation_smart(self, image_path, x, y, output_path, iou_threshold=0.1):
        """
        Intelligent segmentation: After clicking desk, automatically identify all items on desk
        
        Args:
            image_path: Image path
            x, y: Click coordinates
            output_path: Output path
            iou_threshold: IoU threshold (default 0.1, objects with 10% overlap with main object are considered related)
        
        Returns:
            (output_path, score, related_count)
        """
        print(f"🧠 Intelligent recognition mode: Analyzing click position ({x}, {y})...")
        
        # 1. Read image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        self.predictor.set_image(image_np)
        
        # 2. Identify main object (clicked desk)
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        main_masks, main_scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        main_mask = main_masks[0].astype(bool)  # Ensure it's bool type
        main_bbox = self._get_bbox_from_mask(main_mask)
        print(f"   Main object identification complete, bbox: {main_bbox}")
        
        # 3. Automatically identify all objects in image
        print("   🔍 Scanning all objects...")
        all_masks = self.auto_generator.generate(image_np)
        print(f"   Found {len(all_masks)} objects")
        
        # 4. Find related objects that overlap with main object
        related_masks = [main_mask]
        for mask_data in all_masks:
            mask = mask_data['segmentation']
            bbox = self._get_bbox_from_mask(mask)
            
            # Calculate IoU with main object
            iou = self._calculate_iou(bbox, main_bbox)
            
            if iou > iou_threshold:
                # Ensure mask is bool type
                mask_bool = mask.astype(bool) if mask.dtype != bool else mask
                related_masks.append(mask_bool)
                print(f"   ✓ Found related object, IoU={iou:.2f}, bbox={bbox}")
        
        print(f"   📦 Total identified {len(related_masks)} related objects (including main object)")
        
        # 5. Merge all masks (using numpy logical or)
        final_mask = np.zeros_like(main_mask, dtype=bool)
        for mask in related_masks:
            # Ensure size consistency
            if mask.shape != final_mask.shape:
                continue
            final_mask = np.logical_or(final_mask, mask)
        
        # 5.5. Intelligently remove floor (if exists)
        final_mask = self._remove_floor_from_mask(final_mask, image_np)
        
        # 6. Save transparent PNG
        original_rgba = Image.open(image_path).convert("RGBA")
        mask_alpha = Image.fromarray((final_mask * 255).astype(np.uint8)).resize(original_rgba.size)
        original_rgba.putalpha(mask_alpha)
        original_rgba.save(output_path)
        
        return output_path, float(main_scores[0]), len(related_masks) - 1  # Return count of additionally identified objects
    
    def _get_bbox_from_mask(self, mask):
        """Get bounding box from mask [x1, y1, x2, y2]"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            return [int(x1), int(y1), int(x2), int(y2)]
        return [0, 0, 0, 0]
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _remove_floor_from_mask(self, mask, image_np):
        """
        Intelligently remove floor region (if exists)
        
        Strategy:
        1. Only analyze bottom 30% region
        2. Detect if there's a large horizontally continuous area (floor feature)
        3. Judge if it's floor through shape (aspect ratio)
        4. Use color filtering to remove similar colored bottom regions
        5. Protection mechanism: don't process if bottom region is too small
        
        Args:
            mask: Boolean mask array
            image_np: Original image array (for color analysis)
        
        Returns:
            Processed mask
        """
        from scipy import ndimage
        
        height, width = mask.shape
        
        # 1. Analyze bottom 30% region
        bottom_threshold = int(height * 0.7)  # Start from 70% height downwards
        bottom_mask = mask.copy()
        bottom_mask[:bottom_threshold, :] = False  # Only keep bottom
        
        bottom_area = np.sum(bottom_mask)
        total_area = np.sum(mask)
        
        # Protection mechanism: if bottom region is too small, there's no floor
        if total_area == 0 or bottom_area / total_area < 0.15:
            print("   ℹ️  Bottom region is small, no processing needed")
            return mask
        
        print(f"   🔍 Detected bottom region ratio {bottom_area/total_area*100:.1f}%, analyzing floor...")
        
        # 2. Check if bottom is 'large horizontally continuous' (floor feature)
        # Use connected component analysis
        labeled_bottom, num_features = ndimage.label(bottom_mask)
        
        if num_features == 0:
            return mask
        
        # Find largest connected component (possibly floor)
        largest_component_size = 0
        largest_component_label = 0
        for i in range(1, num_features + 1):
            size = np.sum(labeled_bottom == i)
            if size > largest_component_size:
                largest_component_size = size
                largest_component_label = i
        
        # If largest component occupies less than 50% of bottom region, it's not floor
        if largest_component_size / bottom_area < 0.5:
            print("   ℹ️  No large continuous area at bottom, no processing needed")
            return mask
        
        # 3. Analyze shape of largest component (aspect ratio)
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
        
        # Floor is typically flat (aspect ratio > 2)
        if aspect_ratio < 2:
            print(f"   ℹ️  Bottom region aspect ratio {aspect_ratio:.2f}, doesn't look like floor")
            return mask
        
        print(f"   ✅ Detected floor feature (aspect ratio {aspect_ratio:.2f}), starting color filtering...")
        
        # 4. Color analysis: Extract main color of bottom connected component
        bottom_region_pixels = image_np[largest_component_mask]
        
        if len(bottom_region_pixels) == 0:
            return mask
        
        # Calculate main color (median, more robust than mean)
        main_color = np.median(bottom_region_pixels, axis=0)
        
        # 5. Remove bottom regions similar to main color
        # Calculate color distance of each pixel to main color (Euclidean distance)
        color_distance = np.sqrt(np.sum((image_np - main_color) ** 2, axis=2))
        
        # Set threshold (Euclidean distance < 50 considered similar color)
        similar_color_mask = color_distance < 50
        
        # Only apply color filtering in bottom region
        floor_to_remove = largest_component_mask & similar_color_mask
        
        # 6. Remove floor from original mask
        cleaned_mask = mask & ~floor_to_remove
        
        removed_area = np.sum(floor_to_remove)
        print(f"   ✂️  Removing floor region: {removed_area} pixels ({removed_area/total_area*100:.1f}%)")
        
        return cleaned_mask