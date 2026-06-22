import os
import sys
import argparse
from face_detector import AnimeFaceDetector

def main(input_dir, output_dir, cascade_path):
    detector = AnimeFaceDetector(cascade_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                full_path = os.path.join(root, file)
                name, _ = os.path.splitext(file)
                detector.crop_faces(full_path, output_dir, prefix=name)
    print("Ekstraksi selesai. Cek folder", output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/raw_frames')
    parser.add_argument('--output', default='data/faces/unlabeled')
    parser.add_argument('--cascade', default='src/lbpcascade_animeface.xml')
    args = parser.parse_args()
    main(args.input, args.output, args.cascade)