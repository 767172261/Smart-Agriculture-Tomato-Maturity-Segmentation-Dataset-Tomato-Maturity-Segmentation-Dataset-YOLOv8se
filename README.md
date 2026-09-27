果园西红柿分割与检测数据集，2266张，yolo和voc标注方式
3类，标注数量：
verde-未成熟: 17853
maduro-成熟: 4353
semimaduro-半成熟: 4666
image num: 2266
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c1ac5ab44fb849dab881fee54f8a421d.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/867a26592b1d492099de2aca4bed360b.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cc746447254b4fa1a3ce961fbf741ea5.png)


| 项目 | 描述 |
|------|------|
| **数据集名称** | 果园西红柿成熟度检测与分割数据集 |
| **图像总数** | **2,266 张** |
| **任务类型** | 目标检测 + 实例分割（支持检测与像素级分割） |
| **类别数量** | **3 类**（按成熟度划分） |
| **类别及标注数量** | - **verde（未成熟）**：17,853 个实例<br>- **semimaduro（半成熟）**：4,666 个实例<br>- **maduro（成熟）**：4,353 个实例<br>**总计标注实例数**：26,872 个 |
| **标注格式** | **双格式同步提供**：<br>• **YOLO 格式**：每张图像对应 `.txt` 文件，含归一化边界框（检测）或归一化多边形点（分割）<br>• **PASCAL VOC 格式**：每张图像对应 `.xml` 文件（检测框） + 可选 PNG 分割掩码（若含分割） |
| **图像特点** | - 自然果园场景（光照变化、遮挡、重叠、背景复杂）<br>- 包含不同生长阶段的西红柿果实<br>- 部分图像含密集小目标 |
| **适用任务** | - 成熟度分类 + 定位<br>- 果实计数<br>- 农业机器人采摘决策<br>- 小目标检测与实例分割算法研究 |
| **典型应用场景** | - 智能农业采摘机器人视觉系统<br>- 果园产量预估<br>- 农产品自动化分级流水线 |

> 💡 **备注**：该数据集同时支持 **目标检测**（bounding box）与 **实例分割**（polygon/mask），可用于训练 YOLOv8-seg、Mask R-CNN 等模型。

使用 **YOLOv8-seg（实例分割模型）** 训练 **果园西红柿成熟度分割数据集** 的完整流程，包含数据准备、配置、训练与推理代码。

---

## 📁 一、项目结构

```bash
tomato_seg_yolov8/
├── data/
│   ├── images/                 # 2266张图像
│   │   ├── train/
│   │   ├── val/
│   │   └── test/ (可选)
│   └── labels/                 # YOLO 分割标签（.txt）
│       ├── train/
│       ├── val/
│       └── test/
├── dataset.yaml                # 数据集配置文件
├── prepare_data.py             # （可选）划分数据集 + 转换VOC为YOLO-seg格式
└── train.py                    # 启动训练
```

> ✅ 前提：你已有 **YOLO 格式的分割标签**（每行：`class_id x1 y1 x2 y2 ... xn yn`，坐标归一化）

若只有 **VOC XML + PNG掩码**，需先转换（见下文附录）。

---

## 🔧 二、环境安装

```bash
pip install ultralytics opencv-python numpy tqdm
```

---

## 📄 三、数据集配置文件 `dataset.yaml`

```yaml
# dataset.yaml
path: ./data          # 数据集根目录
train: images/train   # 训练图像路径
val: images/val       # 验证图像路径

# 类别数与名称（顺序必须与标签中的 class_id 一致）
nc: 3
names: ['verde', 'maduro', 'semimaduro']
# 注意：根据你的标注，确保：
# verde → 0
# maduro → 1
# semimaduro → 2
# 若顺序不同，请调整 labels 中的类别ID
```

> ⚠️ **重要**：YOLO 标签中的类别 ID 必须与 `names` 列表索引一致。

---

## 🚀 四、训练脚本 `train.py`

```python
# train.py
from ultralytics import YOLO

# 加载 YOLOv8 分割预训练模型
# 可选: yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt
model = YOLO('yolov8s-seg.pt')

# 开始训练
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,              # 输入尺寸（支持多尺度训练）
    batch=16,               # 根据 GPU 显存调整（3090 可设 32）
    workers=8,
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    patience=20,            # Early stopping
    save=True,
    project='runs/tomato_seg',
    name='exp_yolov8s_seg',
    exist_ok=False
)

print("✅ 训练完成！最佳模型保存在: runs/tomato_seg/exp_yolov8s_seg/weights/best.pt")
```

> ✅ 启动训练：
```bash
python train.py
```

---

## 🔍 五、验证与评估

```python
from ultralytics import YOLO

model = YOLO('runs/tomato_seg/exp_yolov8s_seg/weights/best.pt')
metrics = model.val(data='dataset.yaml', imgsz=640)

print(f"mAP50-95 (Box): {metrics.box.map:.4f}")
print(f"mAP50-95 (Mask): {metrics.seg.map:.4f}")  # 实例分割核心指标
```

---

## 🖼️ 六、推理示例（带分割掩码可视化）

```python
# infer.py
from ultralytics import YOLO
import cv2

model = YOLO('runs/tomato_seg/exp_yolov8s_seg/weights/best.pt')

# 单图推理
results = model('data/images/val/example.jpg')

# 保存带分割结果的图像
res_plotted = results[0].plot()  # 自动绘制边界框 + 掩码
cv2.imwrite('tomato_segmented.jpg', res_plotted)

print("✅ 分割结果已保存为 tomato_segmented.jpg")
```

---

## 📊 七、训练建议（针对西红柿小目标 & 密集场景）

| 优化项 | 建议 |
|--------|------|
| **输入尺寸** | 使用 `imgsz=1280` 提升小果实召回率（显存允许前提下） |
| **数据增强** | YOLOv8 默认启用 Mosaic、Copy-Paste（对密集果实有效） |
| **类别不平衡** | 未成熟（verde）样本远多于成熟（maduro）<br>→ 可在 `train` 中设置 `class_weights`（需修改源码）或过采样少数类 |
| **后处理** | 降低 `conf=0.25`, `iou=0.45` 防止漏检重叠果实 |
| **模型选择** | - 快速部署：`yolov8n-seg`<br>- 精度优先：`yolov8m-seg` 或 `yolov8l-seg` |

---

## 📎 附录：若只有 VOC 格式，如何转 YOLO-seg？

### 情况1：有 PNG 分割掩码（推荐）
使用以下脚本将 **PNG 掩码 → YOLO 多边形**：

```python
# voc_mask_to_yolo.py
import os
import cv2
import numpy as np
from tqdm import tqdm

CLASS_MAP = {'verde': 0, 'maduro': 1, 'semimaduro': 2}

def mask_to_yolo_polygon(mask, class_id, img_w, img_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if len(cnt) < 3:  # 至少3点构成多边形
            continue
        # 归一化轮廓点
        polygon = []
        for point in cnt:
            x_norm = point[0][0] / img_w
            y_norm = point[0][1] / img_h
            polygon.extend([x_norm, y_norm])
        if polygon:
            lines.append(f"{class_id} " + " ".join(f"{x:.6f}" for x in polygon))
    return lines

# 示例：处理一个 split
def process_split(split, mask_dir, image_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    for mask_name in tqdm(os.listdir(mask_dir), desc=split):
        if not mask_name.endswith('.png'):
            continue
        mask_path = os.path.join(mask_dir, mask_name)
        img_path = os.path.join(image_dir, mask_name.replace('.png', '.jpg'))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        # 假设 mask 中：0=background, 1=verde, 2=maduro, 3=semimaduro
        lines = []
        for cls_name, cls_id in CLASS_MAP.items():
            cls_mask = (mask == (cls_id + 1)).astype(np.uint8) * 255
            lines.extend(mask_to_yolo_polygon(cls_mask, cls_id, w, h))

        with open(os.path.join(output_label_dir, mask_name.replace('.png', '.txt')), 'w') as f:
            f.write("\n".join(lines))
```


