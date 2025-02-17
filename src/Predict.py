import torch
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_styled_image(model_path, image_path, style_type):
    # Load the entire generator model
    generator = torch.load(model_path, map_location=device)
    generator.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.4,)*3, (0.4,)*3)
    ])


    # Load and preprocess the content image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Prepare the style vector (ensure dimension match)
    style_vector = torch.tensor([[1, 0]] if style_type == "vangogh" else [[0, 1]], dtype=torch.float, device=device)
    style_vector = style_vector.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
    style_vector = style_vector.expand(-1, -1, 1024, 1024)  # Match image dimensions

    # Generate the styled image
    with torch.no_grad():
        combined_input = torch.cat([image, style_vector], dim=1)
        styled_image = generator(combined_input)

    # Convert the output image to a displayable format
    styled_image = styled_image.squeeze(0).cpu().detach()
    styled_image = (styled_image + 1) / 2  # Denormalize to [0, 1]

    # Save and display the styled image
    os.makedirs("output", exist_ok=True)
    save_path = f"output/styled_{style_type}.png"
    transforms.ToPILImage()(styled_image).save(save_path)

    print(f"✅ Styled image saved to: {save_path}")

    return Image.open(save_path)


# Call the function
model_path = "best_generator_epoch.pth"
image_path = r"data\ContentImage\2014-08-06 08_21_56.jpg"
style_type = "vangogh"

styled_image = generate_styled_image(model_path, image_path, style_type)
styled_image.show()
