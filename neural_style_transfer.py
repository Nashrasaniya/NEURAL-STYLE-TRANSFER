import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load and preprocess images
# ---------------------------
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')

    # Resize image
    size = max(image.size)
    if size > max_size:
        scale = max_size / size
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size)

    # Transform to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Convert tensor back to displayable image
def im_convert(tensor):
    image = tensor.clone().detach().squeeze(0)
    image = image.cpu().numpy().transpose(1, 2, 0)
    
    # De-normalize
    image = image * [0.229, 0.224, 0.225]
    image = image + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)

    return image

# ---------------------------
# Feature extraction from layers
# ---------------------------
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content layer
            '28': 'conv5_1'
        }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

# ---------------------------
# Style representation using Gram Matrix
# ---------------------------
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# ---------------------------
# Style Transfer Core Function
# ---------------------------
def run_style_transfer(content_img, style_img, model, content_weight=1e4, style_weight=1e2, steps=200):
    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)

    content_features = get_features(content_img, model)
    style_features = get_features(style_img, model)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for step in range(steps):
        target_features = get_features(target, model)
        
        # Content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        # Style loss
        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss

        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
      
        total_loss.backward(retain_graph=True)

        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, Total loss: {total_loss.item():.2f}")

    return target

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Image paths
    content_path = "content.jpg"  # Make sure you have content.jpg
    style_path = "style.jpg"      # Make sure you have style.jpg

    # Load images
    content = load_image(content_path)
    style = load_image(style_path)

    # Load pre-trained VGG19 model properly
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

    # Apply Style Transfer
    output = run_style_transfer(content, style, vgg)

    # Convert and show final image
    final_img = im_convert(output)
    plt.imshow(final_img)
    plt.title("Stylized Image")
    plt.axis("off")
    plt.show()

    # Save the output image
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.imsave("output/stylized_output.png", final_img)
