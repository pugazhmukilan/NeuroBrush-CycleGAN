
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch
import numpy as np
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_styled_image(model_path, content_img, style_type):
    content_img = transform(content_img).unsqueeze(0).to(device)

    # Load the entire generator model
    generator = torch.load(model_path)
    generator.eval()

    # Set style vector based on style_type
    style_vector = torch.tensor([1, 0] if style_type == 'vangogh' else [0, 1], dtype=torch.float).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
    style_vector = style_vector.expand(-1, -1, 512, 512)  # Expand to match spatial dimensions


    with torch.no_grad():
        fake_image = generator(torch.cat([content_img, style_vector], 1))

    fake_image = fake_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    fake_image = np.clip(fake_image * 214, 0, 214).astype(np.uint8)
    
    return Image.fromarray(fake_image)

# Specify the path to your saved model
# model_path = "best_generator_epoch.pth"  # or final_generator.pth
model_path = "models/best_generator_epoch_150.pth"
# Load a sample content image (ensure it's a PIL image)
content_img_path = r"data\ContentImage\2015-04-30 23_43_35.jpg"
content_img = Image.open(content_img_path)

# Specify the style you want ('vangogh' or 'monet')
style_type = 'vangogh'  # or 'monet'

# Call the generate_styled_image function to get the styled image
styled_image = generate_styled_image(model_path, content_img, style_type)

# Show the generated image
styled_image.show()

# Optionally, save the result if needed
styled_image.save("outputs/generated_Monet_image_for_150_epoch.jpg")