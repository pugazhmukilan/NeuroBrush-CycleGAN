# 🎨 NeuroBrush-CycleGAN
**CycleGAN Neural Style Transfer**

## 🌟 Project Overview
This project implements a **Cycle-Consistent Generative Adversarial Network (CycleGAN)** to perform neural style transfer, transforming natural photographs into artworks styled after **Vincent van Gogh** and **Claude Monet**.

CycleGAN’s **unpaired learning** approach eliminates the need for aligned content-style image pairs, making it ideal for creative domains like digital art. The model preserves the **semantic content** of input images while applying van Gogh’s expressive brushstrokes or Monet’s impressionistic textures.

The pipeline includes dataset preparation, model training with U-Net generators and PatchGAN discriminators, and evaluation of style transfer quality using adversarial, cycle-consistency, and identity losses.

---

## 🎯 Objectives
- Implement CycleGAN to transfer van Gogh and Monet styles to natural photographs.
- Maintain structural and semantic integrity of input images during transformation.
- Produce aesthetically convincing outputs **without paired training data**.
- Demonstrate applications in **digital art**, **graphic design**, and **media production**.

---

## 🖼️ Style Transfer Example
> _Original photo (left), van Gogh style (center), Monet style (right)_

![Style Transfer Example](path_to_placeholder_image1)

---

## 🧠 Methodology

### 📁 Dataset
| Category           | Description                         | Quantity |
|--------------------|-------------------------------------|----------|
| Content Images     | Natural photographs (e.g., landscapes, portraits) | 5,727    |
| Van Gogh Paintings | Images representing the style target | 517      |
| Monet Paintings    | Images representing the style target | 1,944    |

---

### 🧹 Preprocessing
| Step             | Description |
|------------------|-------------|
| Resizing         | All images resized to **256x256** pixels. |
| Normalization    | Pixel values scaled to **[-1, 1]** using `transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)`. |
| Data Augmentation | Random horizontal flips for diversity. |
| DataLoader       | Batch size = 4, shuffling enabled. |

---

### 🏗️ Model Architecture
- **Generators**: U-Net with residual blocks (encoder-decoder + skip connections).
- **Discriminators**: PatchGAN (outputs a 30×30 patch-level real/fake probability matrix).

#### 📌 CycleGAN Architecture

> _CycleGAN architecture showing Generator G (photo → style), Generator F (style → photo), and Discriminators D_X and D_Y_

![CycleGAN Architecture](path_to_placeholder_image2)

---

### 🎯 Loss Functions

#### ✅ Adversarial Loss
Ensures outputs are realistic using **Mean Squared Error (MSE)**:

```python
gen_loss_G = MSE(D_Y(G(X)), 1)
total_disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
