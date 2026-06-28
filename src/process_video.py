import cv2
import argparse
import json
from collections import defaultdict
from pipeline import AnimeFacePipeline
from class_mapping import CLASS_TO_IDX
from face_detector_mtcnn import AnimeMTCNNDetector

def main(video_path, output_video, log_file, device='cpu', frame_skip=1):
    # Inisialisasi pipeline dengan detector MTCNN
    detector = AnimeMTCNNDetector(device=device)
    pipeline = AnimeFacePipeline(detector=detector, model_path='models/anime_face_classifier.pth',
                                class_to_idx=CLASS_TO_IDX, device=device, confidence_thresh=0.6)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps / frame_skip, (width, height))
    
    frame_count = 0
    character_appearances = defaultdict(int)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        result_frame = pipeline.process_array(frame)  # pastikan method ini ada
        out.write(result_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count}")
    
    cap.release()
    out.release()
    # Simpan log
    with open(log_file, 'w') as f:
        json.dump(dict(character_appearances), f, indent=2)
    print(f"Video saved to {output_video}, log to {log_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='outputs/annotated_video.mp4')
    parser.add_argument('--log', default='outputs/character_log.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frame_skip', type=int, default=2, help='Process every N frames')
    args = parser.parse_args()
    main(args.input, args.output, args.log, args.device, args.frame_skip)