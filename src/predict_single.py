import torch
from PIL import Image
from torchvision import transforms
from model import get_model
import sys

def predict(image_path, model_path, classes_list, model_type='resnet18'):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    model = get_model(len(classes_list))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        conf, idx = torch.max(prob, 1)
        pred_class = classes_list[idx]
        print(f"Predicted: {pred_class} (confidence: {conf.item():.2f})")