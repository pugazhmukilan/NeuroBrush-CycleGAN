import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ✅ Define Generator, Discriminator, and CycleGAN classes
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.enc1 = self.conv_block(3, 64, 7, stride=1, padding=3, instance_norm=False)
        self.enc2 = self.conv_block(64, 128, 3, stride=2, padding=1)
        self.enc3 = self.conv_block(128, 256, 3, stride=2, padding=1)
        self.res_blocks = torch.nn.Sequential(*[self.residual_block(256) for _ in range(6)])
        self.dec1 = self.deconv_block(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = self.deconv_block(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = torch.nn.Sequential(torch.nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3), torch.nn.Tanh())

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0, instance_norm=True):
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect')]
        if instance_norm:
            layers.append(torch.nn.InstanceNorm2d(out_channels))
        layers.append(torch.nn.ReLU(inplace=True))
        return torch.nn.Sequential(*layers)

    def residual_block(self, channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),
            torch.nn.InstanceNorm2d(channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, 3, 1, 1, padding_mode='reflect'),
            torch.nn.InstanceNorm2d(channels)
        )

    def deconv_block(self, in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding),
            torch.nn.InstanceNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        r = self.res_blocks(e3)
        d1 = self.dec1(r)
        d2 = self.dec2(d1)
        out = self.dec3(d2)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            self.conv_block(3, 64, 4, stride=2, padding=1, instance_norm=False),
            self.conv_block(64, 128, 4, stride=2, padding=1),
            self.conv_block(128, 256, 4, stride=2, padding=1),
            self.conv_block(256, 512, 4, stride=1, padding=1),
            torch.nn.Conv2d(512, 1, 4, stride=2, padding=1)
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0, instance_norm=True):
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if instance_norm:
            layers.append(torch.nn.InstanceNorm2d(out_channels))
        layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CycleGAN(torch.nn.Module):
    def __init__(self, device):
        super(CycleGAN, self).__init__()
        self.generatorS = Generator().to(device)
        self.generatorP = Generator().to(device)
        self.discriminatorS = Discriminator().to(device)
        self.discriminatorP = Discriminator().to(device)

    def forward(self, x):
        return self.generatorS(x)

def generate_vangogh_image(model_path, input_image_path):
    # ✅ Load Model Directly (Full Model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # ✅ Preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ✅ Load Input Image
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # ✅ Generate Output
    with torch.no_grad():
        output_tensor = model.generatorS(input_tensor)

    # ✅ Post-process
    output_image = output_tensor.squeeze(0).cpu().detach()
    output_image = output_image * 0.5 + 0.5  # Denormalize to [0, 1] range
    output_image = output_image.permute(1, 2, 0).numpy()

    # ✅ Convert to PIL Image
    output_image_pil = Image.fromarray((output_image * 255).astype('uint8'))
    return output_image_pil


# ✅ Generate Van Gogh-style image
model_path = "vangogh_model.pth"
input_image_path = r"data\contentimage\2013-11-10 06_42_01.jpg"
output_image = generate_vangogh_image(model_path, input_image_path)
output_image.show()