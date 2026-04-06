# U-Net Image Segmentation

本專案使用 **PyTorch** 實作一個 **U-Net 多類別語意分割模型**，針對影像與其對應遮罩（mask）進行訓練、驗證與測試。程式包含資料前處理、資料增強、模型訓練、Early Stopping、學習率調整，以及多種分割評估指標的計算。

> 這份程式目前以 **Jupyter Notebook** 形式撰寫，適合用於課堂作業、模型實驗記錄與分割任務原型開發。

---

## 專案特色

- 使用 **U-Net** 架構進行影像語意分割
- 支援 **8 類別** 像素分類
- 自動建立 `train / val / test` 資料切分
- 對訓練資料加入 **Data Augmentation**
- 使用 **AMP (Automatic Mixed Precision)** 加速訓練
- 搭配 **ReduceLROnPlateau** 動態調整學習率
- 使用 **Early Stopping** 避免過度訓練
- 提供 **PA、mPA、Dice、IoU、mIoU、mAP** 等評估指標
- 可視化輸入影像、標註遮罩與預測結果

---

## 專案流程

1. 讀取影像與對應 mask
2. 將資料依比例切分為訓練集、驗證集與測試集
3. 建立 `CCAgT_Dataset` 自訂資料集類別
4. 對訓練集進行 resize、翻轉、旋轉等資料增強
5. 使用 U-Net 模型進行訓練
6. 驗證集監控 loss，並保存最佳模型 `best_model.pth`
7. 在測試集上推論並輸出評估指標
8. 將分割結果與 loss 曲線儲存為圖片

---

## 資料結構

程式預期資料夾結構如下：

```bash
size_512/
├── images_crop/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── masks_crop/
    ├── class_1/
    ├── class_2/
    └── ...
```

其中：

- `images_crop/`：原始影像
- `masks_crop/`：對應的 segmentation mask
- 每張影像需有對應 mask
- 程式會自動搜尋 `.jpg` 或 `.png` 影像，並將 `.jpg` 對應成 `.png` mask

---

## 環境需求

建議使用 Python 3.9 以上，並安裝以下套件：

```bash
pip install torch torchvision numpy opencv-python pillow matplotlib tqdm
```

若使用 GPU，請安裝與 CUDA 版本相符的 PyTorch。

---

## 模型設定

Notebook 中的主要訓練參數如下：

```python
img_size = [512, 512]
LR = 1e-4
BATCH = 32
EPOCHS = 100
n_classes = 8
```

其他設定：

- Optimizer：`Adam`
- Loss Function：`CrossEntropyLoss`
- Scheduler：`ReduceLROnPlateau`
- Early Stopping patience：`10`
- Mixed Precision：啟用

---

## 資料前處理與增強

### 影像處理
- Resize 到 `512 x 512`
- 轉成 Tensor
- 使用 ImageNet mean / std 進行 normalization

### 訓練資料增強
- Random Horizontal Flip (`p=0.5`)
- Random Vertical Flip (`p=0.3`)
- Random Rotation (`±15°`)

### Mask 處理
- 轉成灰階
- 類別值限制在 `0 ~ 7`
- 超出範圍的值會被設為 `0`

---

## U-Net 架構說明

模型由以下模組組成：

- `DoubleConv`
- `Down`
- `Up`
- `OutConv`
- `UNet`

輸入為 3 通道 RGB 影像，輸出為 8 個類別的 segmentation logits。

---

## 輸出檔案

執行後會產生以下輸出：

- `best_model.pth`：驗證集表現最佳的模型權重
- `Training_and_Validation_Loss.png`：訓練與驗證 loss 曲線
- `version1_show_*.png`：影像、標註 mask 與預測結果的視覺化圖片

---

## 資料切分結果

根據 notebook 目前執行結果：

- Train data: **14685**
- Validation data: **2098**
- Test data: **4197**

---

## 測試結果

根據 notebook 中記錄的測試結果：

| Metric | Value |
|---|---:|
| PA | 0.9867 |
| cPA | 0.6125 |
| mPA | 0.6125 |
| Dice | 0.6489 |
| IoU | 0.5189 |
| mIoU | 0.5189 |
| mAP@[.5:.95:.05] | 0.2500 |

此外，訓練過程在 **epoch 60** 觸發 Early Stopping，並在過程中多次更新最佳模型。
