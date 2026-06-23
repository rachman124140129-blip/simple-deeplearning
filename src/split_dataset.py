import os
import shutil
import random
from sklearn.model_selection import train_test_split

dataset_dir = 'data/dataset'   # folder asli dengan subfolder anime/karakter
output_dir = 'data/dataset_split'
random.seed(42)

# Kumpulkan semua path gambar dan label
samples = []
class_names = set()
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(root, file)
            anime = os.path.basename(os.path.dirname(root))
            char = os.path.basename(root)
            label = f"{anime}/{char}"
            samples.append((full_path, label))
            class_names.add(label)

# Stratified split: 70% train, 20% val, 10% test
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Kelompokkan per kelas
from collections import defaultdict
samples_by_class = defaultdict(list)
for path, label in samples:
    samples_by_class[label].append(path)

train, val, test = [], [], []
for label, paths in samples_by_class.items():
    tr, temp = train_test_split(paths, test_size=1-train_ratio, random_state=42, stratify=None)
    relative_test = test_ratio / (val_ratio + test_ratio)
    v, te = train_test_split(temp, test_size=relative_test, random_state=42)
    train.extend([(p, label) for p in tr])
    val.extend([(p, label) for p in v])
    test.extend([(p, label) for p in te])

def copy_files(split_list, split_name):
    for src, label in split_list:
        dst = os.path.join(output_dir, split_name, label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(src, dst)

copy_files(train, 'train')
copy_files(val, 'val')
copy_files(test, 'test')

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")