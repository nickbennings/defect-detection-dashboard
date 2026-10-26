from torchvision.models import resnet18, ResNet18_Weights

print("Loading pretrained ResNet-18...")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
print("Model loaded successfully!")