import gradio as gr
from pipeline import AnimeFacePipeline
from class_mapping import CLASS_TO_IDX
import cv2
import numpy as np

pipeline = AnimeFacePipeline(
    cascade_path='src/lbpcascade_animeface.xml',
    model_path='models/anime_face_classifier.pth',
    class_to_idx=CLASS_TO_IDX,
    device='cpu',
    confidence_thresh=0.6
)

def detect_and_classify(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = pipeline.process_array(img_bgr)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb

iface = gr.Interface(
    fn=detect_and_classify,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Anime Face ID (86 & Tomodachi Game)",
    description="Deteksi dan identifikasi wajah karakter dari anime 86 dan Tomodachi Game."
)
iface.launch()