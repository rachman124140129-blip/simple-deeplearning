import torch
from dataset import AnimeFaceDataset, val_test_transform
from model import get_model
import numpy as np

model = get_model(num_classes)
model.load_state_dict(torch.load('models/anime_face_classifier.pth'))
model.eval()

dataset = AnimeFaceDataset('data/dataset_split', split='val', transform=val_test_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

all_confs = []
all_correct = []
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)
        all_confs.extend(confs.numpy())
        all_correct.extend((preds == labels).numpy())

for thresh in np.arange(0.0, 1.0, 0.05):
    pred_as_unknown = (np.array(all_confs) < thresh)
    tp = sum((np.array(all_confs) >= thresh) & (np.array(all_correct) == 1))
    fp = sum((np.array(all_confs) >= thresh) & (np.array(all_correct) == 0))
    fn = sum((np.array(all_confs) < thresh) & (np.array(all_correct) == 1))
    tn = sum((np.array(all_confs) < thresh) & (np.array(all_correct) == 0))
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0
    print(f"Thresh {thresh:.2f}: Prec={precision:.2f} Rec={recall:.2f} F1={f1:.2f} TP={tp} FP={fp} FN={fn}")