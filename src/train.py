import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import AnimeFaceDataset, train_transform, val_test_transform
from model import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt

BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'data/dataset_split'
MODEL_SAVE_PATH = 'models/anime_face_classifier.pth'

# BLOK PELINDUNG MULTIPROCESSING WINDOWS
if __name__ == '__main__':
    
    # 1. Data Loaders
    train_dataset = AnimeFaceDataset(DATA_DIR, split='train', transform=train_transform)
    val_dataset = AnimeFaceDataset(DATA_DIR, split='val', transform=val_test_transform)
    test_dataset = AnimeFaceDataset(DATA_DIR, split='test', transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.class_to_idx)
    print(f"Classes: {train_dataset.class_to_idx}")
    model = get_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 2. Training Loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # Proses Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=100*correct/total)

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # 3. Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")

        # 4. Simpan Model Terbaik
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Memastikan folder models ada
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved.")

    print(f"Best Val Acc: {best_val_acc:.2f}%")

    # 5. Visualisasi Grafik
    os.makedirs('outputs', exist_ok=True) # Memastikan folder outputs ada
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('outputs/training_curve.png')
    plt.show()

    # 6. Testing Akhir
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            
    test_acc = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")