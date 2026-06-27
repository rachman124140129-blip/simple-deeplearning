import argparse
from pipeline import AnimeFacePipeline
from class_mapping import CLASS_TO_IDX

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', default='outputs/result.jpg')
parser.add_argument('--cascade', default='src/lbpcascade_animeface.xml')
parser.add_argument('--model', default='models/anime_face_classifier.pth')
parser.add_argument('--device', default='cpu')
parser.add_argument('--threshold', type=float, default=0.6)
args = parser.parse_args()

pipeline = AnimeFacePipeline(
    cascade_path=args.cascade,
    model_path=args.model,
    class_to_idx=CLASS_TO_IDX,
    device=args.device,
    confidence_thresh=args.threshold
)

if args.detector == 'mtcnn':
    detector = AnimeMTCNNDetector(model_path='models/animeface_mtcnn.pth')
else:
    detector = AnimeFaceDetector('src/lbpcascade_animeface.xml')
pipeline = AnimeFacePipeline(detector=detector)

pipeline.process_image(args.input, args.output)