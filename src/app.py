import gradio as gr
import cv2
import numpy as np
from pipeline import AnimeFacePipeline
from face_detector_mtcnn import AnimeMTCNNDetector
from class_mapping import CLASS_TO_IDX

detector = AnimeMTCNNDetector(device='cpu')
pipeline = AnimeFacePipeline(detector=detector, model_path='models/anime_face_classifier.pth',
                            class_to_idx=CLASS_TO_IDX, device='cpu', confidence_thresh=0.6)

def process(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w*scale), int(h*scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h))
    result_img, _ = pipeline.process_array(img_bgr)
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_rgb

iface = gr.Interface(
    fn=process,
    inputs=gr.Image(type="numpy", label="Upload gambar anime"),
    outputs=gr.Image(type="numpy", label="Hasil Deteksi & Klasifikasi"),
    title="Anime Face ID - 86 & Tomodachi Game",
    description="Mendeteksi dan mengidentifikasi karakter dari anime 86 dan Tomodachi Game."
)
iface.launch()