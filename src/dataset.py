import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class AnimeFaceDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.transform = transform
        self.split_dir = os.path.join(data_dir, split)
        self.image_paths = []
        self.labels = []
        
        classes = set()
        
        for root, dirs, files in os.walk(self.split_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    
                    class_name = os.path.basename(root)
                    
                    self.image_paths.append(full_path)
                    classes.add(class_name)
                    
        self.classes = sorted(list(classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for path in self.image_paths:
            class_name = os.path.basename(os.path.dirname(path))
            self.labels.append(self.class_to_idx[class_name])
            
        if len(self.image_paths) == 0:
            print(f"Peringatan: Tidak ada gambar yang ditemukan di {self.split_dir}!")

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        image = Image.open(img_path).convert('RGB') 
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label