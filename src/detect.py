import cv2
import sys
import os
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def batch_detect_and_crop(input_folder, output_folder, cascade_path=None):
    if cascade_path is None:
        cascade_path = os.path.join(SCRIPT_DIR, 'lbpcascade_animeface.xml')
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Mencari semua file JPG di folder input
    search_pattern = os.path.join(input_folder, '*.jpg')
    image_files = glob.glob(search_pattern)
    
    if len(image_files) == 0:
        print(f"Error: Tidak ada gambar JPG di folder '{input_folder}'.")
        return
        
    print(f"Mulai memproses {len(image_files)} gambar dari video...")
    total_wajah = 0
    
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )
        
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        
        for i, (x, y, w, h) in enumerate(faces):
            cropped_face = img[y:y+h, x:x+w]
            
            out_filename = f"{name}_face_{i}{ext}"
            out_path = os.path.join(output_folder, out_filename)
            
            cv2.imwrite(out_path, cropped_face)
            total_wajah += 1
            
    print(f"Selesai! Berhasil memotong {total_wajah} wajah dan menyimpannya di '{output_folder}'.")

if __name__ == '__main__':
    input_dir = 'data/raw_frames/86-eps01'
    output_dir = 'data/dataset/faces'
    
    batch_detect_and_crop(input_dir, output_dir)