import cv2
import os
from face_detector import AnimeFaceDetector
from face_detector_mtcnn import AnimeMTCNNDetector

cascade_det = AnimeFaceDetector('src/lbpcascade_animeface.xml')
mtcnn_det = AnimeMTCNNDetector(confidence_thresh=0.9)

test_images = ['data/test_scene1.jpg', 'data/test_scene2.jpg', ...]

for img_path in test_images:
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    
    faces_cascade = cascade_det.detect_from_image(img)
    faces_mtcnn = mtcnn_det.detect_from_image(img)
    
    img_comp = img.copy()
    for (x,y,w,h) in faces_cascade:
        cv2.rectangle(img_comp, (x,y), (x+w,y+h), (0,255,0), 2)
    for (x,y,w,h) in faces_mtcnn:
        cv2.rectangle(img_comp, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.imwrite(f'outputs/benchmark/{img_name}_comparison.jpg', img_comp)
    
    print(f"{img_name}: Cascade={len(faces_cascade)}, MTCNN={len(faces_mtcnn)}")