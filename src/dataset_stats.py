import os
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

dataset_dir = 'data/dataset'

stats = defaultdict(int)
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            anime = os.path.basename(os.path.dirname(root))
            char = os.path.basename(root)
            label = f"{anime}/{char}"
            stats[label] += 1

print("=== Dataset Statistics ===")
for label, count in sorted(stats.items()):
    print(f"{label}: {count} images")
total = sum(stats.values())
print(f"Total: {total} images")

# Tampilkan grid contoh
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
axes = axes.ravel()
class_folders = []
for root, dirs, files in os.walk(dataset_dir):
    if files and all(f.endswith(('.jpg','.png')) for f in files[:1]):
        class_folders.append(root)
class_folders.sort()
for i, folder in enumerate(class_folders[:12]):
    img_files = [f for f in os.listdir(folder) if f.endswith(('.jpg','.png'))][:1]
    if img_files:
        img = cv2.imread(os.path.join(folder, img_files[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        anime = os.path.basename(os.path.dirname(folder))
        char = os.path.basename(folder)
        axes[i].set_title(f"{anime}\n{char}", fontsize=8)
        axes[i].axis('off')
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig('outputs/dataset_samples.png')
plt.show()