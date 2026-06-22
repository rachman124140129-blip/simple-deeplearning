import cv2
import os

class AnimeFaceDetector:
    def __init__(self, cascade_path='lbpcascade_animeface.xml'):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image_path):
        """Mengembalikan list of bounding boxes (x,y,w,h)"""
        img = cv2.imread(image_path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )
        return faces

    def crop_faces(self, image_path, save_dir, prefix='face'):
        """Deteksi dan crop wajah dari satu gambar, simpan dengan nama unik."""
        img = cv2.imread(image_path)
        if img is None:
            return 0
        faces = self.detect_from_image(img)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        count = 0
        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            out_name = f"{prefix}_{i:04d}.jpg"
            cv2.imwrite(os.path.join(save_dir, out_name), face)
            count += 1
        return count

    def detect_from_image(self, img):
        """Versi yang menerima array gambar (untuk efisiensi)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(24,24))