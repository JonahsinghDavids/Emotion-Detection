import torchvision.transforms as transforms
from PIL import Image

# Define the preprocessing steps (resize, normalize, etc.)
def preprocess_image(image):
    """
    Preprocesses the uploaded image for the model.
    Converts the image to grayscale, resizes it, and normalizes it for ResNet.

    Args:
        image (PIL.Image): The uploaded image.

    Returns:
        Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel image
        transforms.Resize((224, 224)),  # Resize the image to 224x224 (ResNet input size)
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
    ])
    
    # Apply transformations and return the tensor
    return transform(image)
