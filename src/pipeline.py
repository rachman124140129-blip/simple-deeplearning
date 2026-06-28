import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os

from face_detector import AnimeFaceDetector
from model import get_model

class AnimeFacePipeline:
    def __init__(self, detector, model, class_to_idx, device='cpu', confidence_thresh=0.5):
        self.detector = detector
        
    def __init__(self, cascade_path, model_path, class_to_idx, model_type='resnet18', device='cpu', confidence_thresh=0.5):
        self.detector = AnimeFaceDetector(cascade_path)
        self.device = torch.device(device)
        self.confidence_thresh = confidence_thresh
        
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        
        self.model = get_model(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_face(self, face_img):
        """Menerima array numpy BGR crop wajah, mengembalikan label dan confidence."""
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            if conf.item() < self.confidence_thresh:
                return "Unknown", conf.item()
            label = self.idx_to_class[idx.item()]
            return label, conf.item()
    
    def process_image(self, image_path, output_path=None):
        """Proses gambar penuh: deteksi, klasifikasi, gambar bounding box."""
        img = cv2.imread(image_path)
        if img is None:
            print("Gambar tidak dapat dibaca.")
            return
        
        faces = self.detector.detect_from_image(img)
        if len(faces) == 0:
            print("Tidak ada wajah terdeteksi.")
            if output_path:
                cv2.imwrite(output_path, img)
            return img
        
        for (x, y, w, h) in faces:
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            label, conf = self.predict_face(crop)
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = f"{label} ({conf:.2f})" if label != "Unknown" else "Unknown"
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Hasil disimpan di {output_path}")
        return img
    
    def process_array(self, img_bgr):
        img = img_bgr.copy()
        results = []
        faces = self.detector.detect_from_image(img)
        for (x,y,w,h) in faces:
            crop = img[y:y+h, x:x+w]
            if crop.size == 0: continue
            label, conf = self.predict_face(crop)
            results.append({'bbox': [x,y,w,h], 'label': label, 'confidence': conf})
            color = (0,255,0) if label != "Unknown" else (0,0,255)
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            text = f"{label} ({conf:.2f})" if label != "Unknown" else "Unknown"
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img, results