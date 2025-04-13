# NeuroBrush-CycleGAN
CycleGAN Neural Style Transfer
Project Overview
This project implements a Cycle-Consistent Generative Adversarial Network (CycleGAN) to perform neural style transfer, transforming natural photographs into artworks styled after Vincent van Gogh and Claude Monet. CycleGAN's unpaired learning approach eliminates the need for aligned content-style pairs, making it ideal for artistic applications. The model preserves the semantic content of input images while applying van Gogh’s expressive brushstrokes or Monet’s impressionistic textures, producing high-quality, visually appealing outputs.

The pipeline includes dataset preparation, model training with U-Net generators and PatchGAN discriminators, and evaluation of style transfer quality. Optimized using adversarial, cycle-consistency, and identity losses, the system achieves robust performance for creative applications in digital art, graphic design, and media production.

Image Placeholder 1: Style Transfer Example

Insert an image showing a natural photograph alongside its van Gogh and Monet stylized versions. Suggested caption: "Original photo (left), van Gogh style (center), Monet style (right)."

Objectives
Implement CycleGAN to transfer van Gogh and Monet styles to natural photographs.
Maintain structural and semantic integrity of input images during transformation.
Produce aesthetically convincing outputs without paired training data.
Demonstrate applications in creative industries.
Methodology
Dataset
The dataset comprises three categories:

Category	Description	Quantity
Content Images	Natural photographs (e.g., landscapes, portraits)	5,727
Van Gogh Paintings	Images representing the style target	517
Monet Paintings	Images representing the style target	1,944
Preprocessing
Step	Description
Resizing	Images resized to 256x256 pixels for uniformity.
Normalization	Pixel values scaled to [-1, 1] using transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).
Data Augmentation	Random horizontal flips applied to enhance dataset diversity.
DataLoader	Batches of size 4 created with shuffling enabled for randomized training.
Model Architecture
Generators: U-Net with residual blocks, using an encoder-decoder structure and skip connections to preserve spatial details.
Discriminators: PatchGAN, evaluating realism at the patch level, outputting a 30x30 real/fake probability matrix for 256x256 inputs.
Image Placeholder 2: CycleGAN Architecture

Insert a diagram of the CycleGAN framework, showing Generator G (photo to style), Generator F (style to photo), and Discriminators D_X and D_Y. Suggested caption: "CycleGAN architecture for unpaired style transfer."

Loss Functions
Adversarial Loss: Ensures realistic outputs via Mean Squared Error:
python

Collapse

Wrap

Copy
gen_loss_G = MSE(D_Y(G(X)), 1)
total_disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
Cycle-Consistency Loss: Preserves content using L1 loss:
python

Collapse

Wrap

Copy
cycle_loss = L1_loss(F(G(X)), X) + L1_loss(G(F(Y)), Y)
Identity Loss: Maintains color composition:
python

Collapse

Wrap

Copy
identity_loss = L1_loss(G(Y), Y) + L1_loss(F(X), X)
Total Loss:
python

Collapse

Wrap

Copy
total_loss = adversarial_loss + 10 * cycle_loss + 0.5 * identity_loss
Training
Optimizer: Adam (learning rate=0.0002, β1=0.5, β2=0.999).
Epochs: 200–250, with loss stabilizing at 2.0–3.0.
Process: Alternates generator and discriminator updates.
Requirements
Hardware
Component	Specification	Purpose
GPU	NVIDIA RTX 4060 (24GB VRAM)	Accelerates GAN training with CUDA cores.
CPU	Intel i7-12700K	Handles data loading and preprocessing.
RAM	24GB DDR4/5	Supports large batch sizes and feature maps.
Storage	SSD (512GB–1TB NVMe)	Fast dataset loading and checkpoint saving.
OS	Windows 11 / Ubuntu 20.04+	Ensures CUDA and PyTorch compatibility.
Software
Component	Version	Role
Python	3.8+	Base language.
PyTorch	1.12.1 (CUDA 11.6)	Deep learning framework.
TorchVision	0.13.1	Image transformations and utilities.
CUDA Toolkit	11.6	GPU acceleration.
cuDNN	8.5.0	Optimized deep learning primitives.
Libraries
Library	Version	Usage
NumPy	1.21.6	Numerical operations.
Matplotlib	3.5.3	Visualization of training progress.
tqdm	4.64.1	Progress bars for training loops.
Pillow (PIL)	9.2.0	Image loading and preprocessing.
OpenCV	4.6.0	Optional: Advanced image augmentation.
Installation
Set Up Environment:
bash

Collapse

Wrap

Copy
python -m venv cyclegan_env
source cyclegan_env/bin/activate  # Linux/Mac
cyclegan_env\Scripts\activate     # Windows
pip install torch==1.12.1 torchvision==0.13.1 numpy==1.21.6 matplotlib==3.5.3 tqdm==4.64.1 pillow==9.2.0 opencv-python==4.6.0
Install CUDA:
Download CUDA Toolkit 11.6 and cuDNN 8.5.0 from NVIDIA’s website.
Follow installation instructions for your OS.
Prepare Dataset:
Organize datasets as:
text

Collapse

Wrap

Copy
datasets/
├── content_images/
├── van_gogh/
├── monet/
Preprocess images using provided scripts.
Clone Repository:
bash

Collapse

Wrap

Copy
git clone <your_repository_url>
cd cyclegan_style_transfer
Usage
Train Model:
bash

Collapse

Wrap

Copy
python train.py --dataset_path datasets/ --epochs 200 --batch_size 4 --lr 0.0002
Loss plots are saved to plots/.
Generate Images:
bash

Collapse

Wrap

Copy
python inference.py --model_path checkpoints/cyclegan.pth --input_image path/to/photo.jpg --style van_gogh
Outputs are saved to outputs/.
Image Placeholder 3: Training Progress

Insert a plot of training losses (adversarial, cycle-consistency, identity) over epochs. Suggested caption: "Training loss curves for CycleGAN."

Results
The model achieves:

Style Accuracy: Captures van Gogh’s bold strokes and Monet’s soft textures.
Visual Quality: Produces aesthetically pleasing, artwork-like images.
Content Preservation: Retains input details, validated by low cycle-consistency loss.
Image Placeholder 4: Results Comparison

Insert an image comparing an original photo with its van Gogh and Monet styled outputs. Suggested caption: "Style transfer results: Original (left), van Gogh (center), Monet (right)."

Future Work
Enhance Image Quality: Use progressive growing of GANs for sharper outputs and stable training.
Prompt-Based Multi-Style Transfer: Enable dynamic style blending via user prompts.
Video Style Transfer: Apply styles to videos with temporal consistency.
Additional Styles: Expand to other artistic styles for broader applications.
