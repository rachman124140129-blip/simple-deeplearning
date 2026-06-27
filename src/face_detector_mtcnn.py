import torch
import numpy as np
import cv2
from torchvision.ops import nms

class AnimeMTCNNDetector:
    def __init__(self, model_path='models/animeface_mtcnn.pth', device='cpu', confidence_thresh=0.9, nms_thresh=0.3):
        self.device = torch.device(device)
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        self.model = torch.hub.load('hysts/anime-face-detector', 'anime_face_detector', pretrained=True, device=self.device)
        self.model.eval()

    def detect(self, image_path):
        """Mengembalikan list bounding boxes (x1,y1,x2,y2)"""
        img = cv2.imread(image_path)
        if img is None:
            return []
        return self.detect_from_image(img)

    def detect_from_image(self, img):
        with torch.no_grad():
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            scale = 800 / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h))
            img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            detections = self.model(img_tensor)
            if len(detections) == 0:
                return []
            boxes = detections[0]['boxes'].cpu().numpy()
            scores = detections[0]['scores'].cpu().numpy()
            
            keep = scores >= self.confidence_thresh
            boxes = boxes[keep]
            scores = scores[keep]

            boxes /= scale
            
            if len(boxes) > 0:
                boxes_tensor = torch.tensor(boxes)
                scores_tensor = torch.tensor(scores)
                keep_indices = nms(boxes_tensor, scores_tensor, self.nms_thresh)
                boxes = boxes_tensor[keep_indices].numpy()
            
            result = []
            for box in boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                result.append((int(x1), int(y1), int(w), int(h)))
            return result