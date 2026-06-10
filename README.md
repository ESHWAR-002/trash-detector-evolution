# Comparative Ablation Study: 5-Class Waste Localization Engine
### An Architectural Evaluation of YOLO11 vs. YOLO26 Backbones

Welcome to the engineering portfolio for the **Taco Data Set Model**. This repository documents an end-to-end computer vision research journey tracking the training, preprocessing pipelines, and real-world optimization of object detection models across five primary waste streams: **Cardboard, Glass, Metal, Paper, and Plastic**.

Rather than treating model training as a black box, this project functions as a direct comparative analysis between two distinct computer vision architectures trained on an identical data pipeline:
1. **YOLO11 (Extra Large):** A high-capacity network utilizing traditional regression layers.
2. **YOLO26 (Extra Large):** A next-generation object detection framework optimized for highly disciplined end-to-end efficiency.

---

## 📊 1. Dataset Architecture & Pipeline Curation

To maintain strict experimental control, both architectures were trained on an identical data configuration managed natively within Roboflow. 

### Data Preprocessing Configuration
* **Auto-Orient:** Applied to strip EXIF orientation metadata and standardize image orientation.
* **Resolution Scaling:** Images were resized using a strict **Stretch to 640x640** mapping to match the native input tensor constraints of both YOLO models.
* **Contrast Optimization:** Implemented **Adaptive Equalization** to dynamically normalize local image contrast, helping the model isolate edges in poorly lit environments.
* **Null Filtering:** Disabled (`Do not filter any null images`) to ensure the network explicitly learns background features from frames containing zero objects, drastically reducing false-positive rates in production.

### Data Augmentation Strategy (10x Multiplier)
To artificially expand the abstract feature space and prevent early network convergence, an augmentation matrix was applied to generate **10 outputs per training example**:
* **Horizontal Flip:** Standard structural reflection.
* **Rotation:** Randomly applied between -15° and +15° to simulate varied camera angles.
* **Brightness Adjustments:** Scaled variance between -15% and +15% to replicate real-world outdoor and indoor lighting flux.
* **Gaussian Blur:** Applied up to 0.5px to handle motion blur or out-of-focus optics.
* **Salt-and-Pepper Noise:** Injected up to 0.5% of pixels to simulate sensor grain or low-light hardware limitations.

### Final Dataset Partitions
* **Training Partition:** 11,420 images (Synthetically augmented via the 10x matrix)
* **Validation Partition:** 381 images (Pristine, unaugmented)
* **Testing Partition:** 380 images (Pristine, unaugmented)
* **Total Image Bank:** 12,181 total images

---

## 📈 2. Model 1: My First Attempt (YOLO11x)

For my first model, I deployed the high-capacity **YOLO11 Extra Large (YOLO11x)** architecture initialized with baseline COCO weights and fine-tuned it over 45 epochs. 

### Model 1 Final Performance Metrics
* **mAP@50:** 33.1%
* **Precision:** 42.6%
* **Recall:** 31.2%
* **F1-Score:** 36.0%
* <img width="1024" height="212" alt="Screenshot 2026-06-10 195218" src="https://github.com/user-attachments/assets/3aa41890-4ae1-450a-b020-56a3cee87028" />


### Baseline Per-Class Performance Breakdown (Test Set)
| Class Icon | Material Class | Test mAP@50 | Operational Performance Analysis |
| :--- | :---: | :---: | :--- |
| 📦 | **Cardboard** | **80.0%** | **Excellent:** Predictable lines and flat, opaque surfaces allowed the network to extract clean edge features easily. |
| 🥤 | **Plastic** | **57.0%** | **Moderate:** Handled rigid, solid-colored containers well, but struggled when plastic items were crushed or transparent. |
| 🪙 | **Metal** | **40.0%** | **Sub-optimal:** Crushed cans were easily localized, but heavy specular surface reflections regularly confused the classifier. |
| 📄 | **Paper** | **32.0%** | **Poor:** Crumpled or torn paper frequently lost its flat geometric boundaries, leading to misclassifications as cardboard. |
| 🍾 | **Glass** | **23.0%** | **Failure:** Drastic environmental light refraction and transparency made finding clear object boundaries nearly impossible. |
| 🔄 | **Other** | **0.0%** | **Scarcity:** Zero activations occurred due to a lack of core training examples in the initial manual annotation phase. |

---

## 🚀 3. Model 2: Upgrading the Architecture (YOLO26x)

After reviewing my baseline, I immediately launched my second model on the exact same date and data profile. I upgraded the system to the newer, next-generation **YOLO26 Extra Large (YOLO26x)** architecture to see if its NMS-free end-to-end processing and new MuSGD optimizer could handle my data bottlenecks more efficiently.

### Model 2 Final Performance Metrics
* **mAP@50:** **39.8%** *(+6.7% improvement)*
* **Precision:** **68.9%** *(+26.3% improvement)*
* **Recall:** **32.6%** *(+1.4% improvement)*
* **F1-Score:** **44.2%** *(+8.2% improvement)*

Upgrading to the YOLO26 architecture yielded a massive jump in Precision, soaring from 42.6% to 68.9%. This means Model 2 became dramatically more reliable, wiping out a huge portion of the false-positive errors that plagued my first run.
<img width="1036" height="209" alt="Screenshot 2026-06-10 195204" src="https://github.com/user-attachments/assets/00e45793-06b2-455e-ab9f-aee510b594f5" />


---

## 🔍 4. Training Artifacts & Critical Loss Analysis

### Training Curves & Logs
**MODAL 1**<img width="1028" height="677" alt="image" src="https://github.com/user-attachments/assets/cc1cec62-fd30-4187-ba10-1c74e36c2384" />
**MODAL 2**<img width="1029" height="681" alt="image" src="https://github.com/user-attachments/assets/21f81685-dd8d-460a-a016-fdc723df607a" />
### Why Both Models Failed Industrial Production Standards:
By enterprise or manufacturing deployment standards, an overall accuracy performance of **33.1% mAP (YOLO11x)** or **39.8% mAP (YOLO26x)** is classified as an absolute operational failure. To deploy an automated sorting arm or conveyor belt system, industrial pipelines require a minimum threshold of **85% to 90% mAP@50** to avoid catastrophic cross-contamination of materials. 

A granular engineering investigation of the validation logs isolates the exact mathematical reasons behind this system-wide performance failure:

1. **The Core Data-Capacity Disconnect (Under-determined Dataset):** You cannot bypass fundamental data limitations with complex architectures. The root failure of both models traces back to a massive capacity imbalance. Training an **Extra Large (x)** parameter network—which features tens of millions of structural weights—requires massive environmental diversity. Because the base corpus consisted of only ~1,900 unique baseline environments, both architectures completely exhausted the authentic feature space early.

2. **The Epoch 7 Classification Collapse (`val/cls_loss`):** The classification loss dropped quickly to its global minimum at exactly **Epoch 7 ($\sim 1.85$)** before climbing steadily upward for the rest of the 45 epochs. This proves that by Epoch 7, the model completely stopped learning generalized visual concepts of trash. For the remaining 38 epochs, it used its massive parameter weight space to memorize the unique, repeating background patterns, grain, and lighting artifacts introduced by the synthetic 10x augmentation matrix. It was no longer detecting classes; it was purely memorizing your specific training images.

3. **The Preprocessing and Blur Bottleneck:** Applying a **Stretch to 640x640** preprocessing constraint forced varied aspect ratio images into a uniform square canvas. This distorted the true geometric aspect ratios of flexible items like paper and plastic. Furthermore, combining a 0.5px Gaussian Blur with a 0.5% Salt-and-Pepper noise matrix on an already small base image pool destroyed fine edge texturing. The network was forced to optimize on blurry, pixelated artifacts, which directly caused the catastrophic collapse of complex, low-contrast, or transparent streams like **Glass (23.0% mAP)** and **Paper (32.0% mAP)**.

4. **The Synchronous Epoch 33 Gradient Optimization Cliff (`val/obj_loss`):** The objectness loss tracked smoothly until Epoch 33, where it exploded vertically from 1.82 to over 2.10. Simultaneously, the coordinate bounding-box layer (`val/box_loss`) experienced a violent upward spike at Epoch 34. This indicates a massive gradient destabilization. Because the model was already brittle from memorizing the training backgrounds, hitting a clustered sequence of highly rotated, noisy, multi-object images completely shattered the regression layers' spatial logic, causing the network to lose all structural confidence.

### The Architectural Takeaway:
While upgrading to **YOLO26x** yielded a clear structural jump in Precision (**68.9%** vs. 42.6% for YOLO11x), its ultimate failure to cross the 40% mAP line demonstrates that **data engineering beats architectural manipulation**. No matter how advanced the model's prediction heads or optimizers are, a model cannot generalize features that do not natively exist within the baseline dataset.


---

## 🔗 Live Interactive Testing
Both of my custom trained models, dataset health details, and their live browser-based webcam testing sandboxes are hosted publicly via Roboflow.

👉 **[Run Live Inference Sandbox on Roboflow Universe](https://universe.roboflow.com/eshwar-imail-edu-vn/taco-dataset-ql1ng-vhqo2/dataset/5)**
