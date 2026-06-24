import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from dataset import AnimeFaceDataset, val_test_transform
from model import get_model
import torch.nn as nn

def evaluate(model_path, data_dir, split='test', model_type='resnet18'):
    dataset = AnimeFaceDataset(data_dir, split=split, transform=val_test_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    num_classes = len(dataset.class_to_idx)
    class_names = list(dataset.class_to_idx.keys())
    
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    print(classification_report(all_labels, all_preds, target_names=class_names))