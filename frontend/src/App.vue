<template>
  <div class="app">
    <h1 class="title">Spatial Shift</h1>

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

      <button class="confirm-btn" @click="confirmMove" :disabled="!previewUrl">
        Confirm
      </button>
    </div>

    <!-- 图片预览：确认后显示，替代上传框 -->
    <div v-else class="image-wrapper">
      <button class="close-btn" @click="closeImage">×</button>

      <div class="image-preview">
        <img :src="confirmedUrl" alt="预览图片" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";

const fileName = ref("");
const previewUrl = ref(null);
const confirmedUrl = ref(null);

const handleUpload = (event) => {
  const file = event.target.files[0];
  if (!file) return;

  fileName.value = file.name;
  previewUrl.value = URL.createObjectURL(file);
};

const confirmMove = () => {
  confirmedUrl.value = previewUrl.value;
};

const closeImage = () => {
  confirmedUrl.value = null;
  previewUrl.value = null;
  fileName.value = "";
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
</style>
