<template>
  <div class="app">
    <h1 class="title">Spatial Shift</h1>
    <p class="subtitle">
      Interact with your photos by selecting and moving objects freely.
    </p>

    <!-- 上传框：只有在没有确认图片时显示 -->
    <div v-if="!confirmedUrl" class="upload-panel">
      <p class="upload-title">Upload your picture</p>

      <div class="upload-bar">
        <span class="upload-placeholder" v-if="!fileName">
          Click the button to choose a picture
        </span>
        <span class="upload-filename" v-else>{{ fileName }}</span>

        <label class="upload-button">
          Choose picture
          <input type="file" accept="image/*" @change="handleUpload" hidden />
        </label>
      </div>

      <button class="confirm-btn" @click="confirmMove" :disabled="!uploadedFile">
        Confirm
      </button>
    </div>

    <!-- 图片预览：确认后显示，替代上传框 -->
    <div v-else-if="!finalImageUrl" class="image-wrapper">
      <button class="close-btn" @click="closeImage">×</button>

      <div class="image-preview">
        <!-- 状态提示 -->
        <div v-if="processing" class="processing-overlay">
          <div class="spinner"></div>
          <p>{{ statusMessage }}</p>
        </div>

        <img 
          :src="confirmedUrl" 
          alt="预览图片" 
          @click="handleImageClick"
          :style="{ cursor: processing ? 'wait' : 'crosshair' }"
        />
        
        <!-- 点击提示 -->
        <div class="click-hints">
          <p v-if="!firstClick" class="hint">
            🖱️ Step 1: Click on the furniture you want to move
          </p>
          <p v-else-if="!secondClick" class="hint">
            🎯 Step 2: Click where you want to place it
          </p>
          <p v-if="firstClick" class="pixel-display">
            Furniture at: ({{ firstClick.x }}, {{ firstClick.y }})
          </p>
          <p v-if="secondClick" class="pixel-display">
            Target at: ({{ secondClick.x }}, {{ secondClick.y }})
          </p>
        </div>

        <!-- 在图片上显示点击标记 -->
        <div v-if="firstClick" class="click-marker first" 
             :style="{ left: firstClick.displayX + 'px', top: firstClick.displayY + 'px' }">
          1
        </div>
        <div v-if="secondClick" class="click-marker second" 
             :style="{ left: secondClick.displayX + 'px', top: secondClick.displayY + 'px' }">
          2
        </div>
      </div>
    </div>

    <!-- 最终结果 -->
    <div v-else class="image-wrapper">
      <button class="close-btn" @click="reset">×</button>
      <div class="image-preview">
        <img :src="finalImageUrl" alt="处理结果" />
        <p class="success-message">✅ Furniture moved successfully!</p>
        <button class="retry-btn" @click="retryWithSameImage">Move another furniture</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";

// 后端 API 地址
const API_BASE = "http://localhost:8000";

// 状态变量
const fileName = ref("");
const uploadedFile = ref(null);
const imageId = ref(null);
const previewUrl = ref(null);
const confirmedUrl = ref(null);

const firstClick = ref(null);
const secondClick = ref(null);
const processing = ref(false);
const statusMessage = ref("");
const finalImageUrl = ref(null);

// 处理文件上传
const handleUpload = async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  fileName.value = file.name;
  uploadedFile.value = file;
  previewUrl.value = URL.createObjectURL(file);
};

// 确认并上传到后端
const confirmMove = async () => {
  if (!uploadedFile.value) return;

  try {
    processing.value = true;
    statusMessage.value = "Uploading image...";

    // 上传图片到后端
    const formData = new FormData();
    formData.append("file", uploadedFile.value);

    const response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Upload failed");

    const data = await response.json();
    imageId.value = data.image_id;
    confirmedUrl.value = previewUrl.value;
    
    processing.value = false;
    statusMessage.value = "";
  } catch (error) {
    alert("Upload failed: " + error.message);
    processing.value = false;
  }
};

// 处理图片点击
const handleImageClick = (event) => {
  if (processing.value) return;

  const img = event.target;
  const rect = img.getBoundingClientRect();

  // 获取点击位置相对于图片显示区域的坐标
  const displayX = event.clientX - rect.left;
  const displayY = event.clientY - rect.top;

  // 计算实际图片坐标（考虑缩放）
  const scaleX = img.naturalWidth / img.width;
  const scaleY = img.naturalHeight / img.height;
  const actualX = Math.round(displayX * scaleX);
  const actualY = Math.round(displayY * scaleY);

  console.log("Click:", { displayX, displayY, actualX, actualY });

  if (!firstClick.value) {
    // 第一次点击：选择家具
    firstClick.value = { 
      x: actualX, 
      y: actualY,
      displayX: displayX,
      displayY: displayY
    };
  } else if (!secondClick.value) {
    // 第二次点击：选择目标位置
    secondClick.value = { 
      x: actualX, 
      y: actualY,
      displayX: displayX,
      displayY: displayY
    };
    
    // 开始处理
    processImage();
  }
};

// 调用后端处理图片
const processImage = async () => {
  processing.value = true;

  try {
    statusMessage.value = "🔍 Step 1/3: Segmenting furniture...";
    
    // 调用后端 full_pipeline API
    const formData = new FormData();
    formData.append("file", uploadedFile.value);
    formData.append("segment_x", firstClick.value.x);
    formData.append("segment_y", firstClick.value.y);
    formData.append("place_x", secondClick.value.x);
    formData.append("place_y", secondClick.value.y);
    formData.append("use_ai", "false"); // 使用 Alpha Blending

    statusMessage.value = "🎨 Step 2/3: Removing furniture...";
    
    const response = await fetch(`${API_BASE}/full_pipeline`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Processing failed");
    }

    statusMessage.value = "✨ Step 3/3: Placing furniture...";
    
    const data = await response.json();
    
    // 显示最终结果（Base64 转图片）
    finalImageUrl.value = `data:image/png;base64,${data.final_image}`;
    
  } catch (error) {
    alert("Processing failed: " + error.message);
    firstClick.value = null;
    secondClick.value = null;
  } finally {
    processing.value = false;
    statusMessage.value = "";
  }
};

// 关闭图片
const closeImage = () => {
  confirmedUrl.value = null;
  previewUrl.value = null;
  fileName.value = "";
  uploadedFile.value = null;
  imageId.value = null;
  firstClick.value = null;
  secondClick.value = null;
};

// 重置所有状态
const reset = () => {
  confirmedUrl.value = null;
  previewUrl.value = null;
  fileName.value = "";
  uploadedFile.value = null;
  imageId.value = null;
  firstClick.value = null;
  secondClick.value = null;
  finalImageUrl.value = null;
};

// 用同一张图片重新移动
const retryWithSameImage = () => {
  finalImageUrl.value = null;
  firstClick.value = null;
  secondClick.value = null;
};
</script>

<style scoped>
:global(body) {
  background-color: #fdf7e3;
  color: #6e6e6e;
  font-family: "Orbitron", sans-serif;
}

.app {
  max-width: 900px;
  margin: 0 auto;
  padding: 20px;
}

.title {
  font-size: 52px;
  font-weight: 700;
  letter-spacing: 3px;
  text-shadow: 0 0 10px rgba(0, 0, 0, 0.25);
}

/* 整个上传对话框区域 */
.upload-panel {
  margin-top: 24px;
  padding: 16px 18px;
  border-radius: 16px;
  background-color: #ffffff;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* 上面的提示文字 */
.upload-title {
  font-size: 14px;
  color: #a0a0a0;
}

/* 类似聊天输入框的长条 */
.upload-bar {
  border-radius: 999px;
  border: 1px solid #c7f1c4;
  background-color: hsl(0, 0%, 100%);
  padding: 8px 10px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  width: 800px;
}

/* 左侧占位文字（很淡的灰色） */
.upload-placeholder {
  font-size: 13px;
  color: #b8b8b8;
}

/* 选择完文件后显示的文件名 */
.upload-filename {
  font-size: 13px;
  color: #6e6e6e;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 右侧“选择图片”按钮 */
.upload-button {
  font-size: 13px;
  padding: 6px 12px;
  border-radius: 999px;
  background: linear-gradient(135deg, #b8e6b8, #cbdf7a);
  color: white;
  cursor: pointer;
  border: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
}

/* 画布区域 */
.canvas-container {
  border: 1px solid #ccc;
  height: 400px;
  margin-top: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-preview {
  margin-top: 20px;
  width: 100%;
  display: flex;
  justify-content: center;
  position: relative;
}

.image-preview img {
  width: 75vw;
  max-height: 75vh;
  object-fit: contain;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
}

.image-wrapper {
  position: relative;
  width: 100%;
  display: flex;
  justify-content: center;
  margin-top: 20px;
}

/* 叉叉按钮在图片外侧左上角 */
.close-btn {
  position: absolute;
  top: -10px; /* 往图片外上移 */
  left: -10px; /* 往图片外左移 */
  background: rgba(177, 45, 45, 0);
  border: none;
  border-radius: 50%;
  width: 34px;
  height: 34px;
  font-size: 20px;
  font-weight: bold;
  color: #333;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
  transition: 0.2s;
}

.close-btn:hover {
  background: white;
}

/* 图片容器 */
.image-preview {
  display: flex;
  justify-content: center;
  width: 100%;
}

/* 控制图片大小（占屏幕 75%） */
.image-preview img {
  width: 75vw;
  max-height: 75vh;
  object-fit: contain;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
}

.pixel-display {
  margin-top: 10px;
  font-size: 14px;
  color: #888;
  text-align: center;
}

/* 新增样式 */
.subtitle {
  font-size: 16px;
  color: #888;
  margin-top: -10px;
}

.confirm-btn {
  align-self: flex-end;
  padding: 8px 24px;
  border-radius: 999px;
  background: linear-gradient(135deg, #b8e6b8, #cbdf7a);
  color: white;
  border: none;
  cursor: pointer;
  font-size: 14px;
}

.confirm-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.close-btn {
  z-index: 10;
}

/* 处理中遮罩层 */
.processing-overlay {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(255, 255, 255, 0.95);
  padding: 30px 40px;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  z-index: 5;
  text-align: center;
}

/* Loading 动画 */
.spinner {
  width: 50px;
  height: 50px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #b8e6b8;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 15px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 点击提示 */
.click-hints {
  margin-top: 15px;
  text-align: center;
}

.hint {
  font-size: 16px;
  color: #4a90e2;
  font-weight: 600;
  margin-bottom: 8px;
}

/* 点击标记 */
.click-marker {
  position: absolute;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 14px;
  color: white;
  transform: translate(-50%, -50%);
  pointer-events: none;
  z-index: 3;
}

.click-marker.first {
  background: #ff6b6b;
  border: 3px solid white;
  box-shadow: 0 2px 8px rgba(255, 107, 107, 0.5);
}

.click-marker.second {
  background: #4a90e2;
  border: 3px solid white;
  box-shadow: 0 2px 8px rgba(74, 144, 226, 0.5);
}

/* 成功消息 */
.success-message {
  margin-top: 15px;
  font-size: 18px;
  color: #4caf50;
  font-weight: 600;
}

/* 重试按钮 */
.retry-btn {
  margin-top: 15px;
  padding: 10px 30px;
  border-radius: 999px;
  background: linear-gradient(135deg, #b8e6b8, #cbdf7a);
  color: white;
  border: none;
  cursor: pointer;
  font-size: 14px;
  transition: transform 0.2s;
}

.retry-btn:hover {
  transform: scale(1.05);
}

.image-preview {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
}
</style>
