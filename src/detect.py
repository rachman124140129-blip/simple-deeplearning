import cv2
import sys

def detect_anime_faces(image_path, cascade_path='src/lbpcascade_animeface.xml'):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    out_path = image_path.replace('.jpg', '_detected.jpg')
    cv2.imwrite(out_path, img)
    print(f"Ditemukan {len(faces)} wajah, hasil disimpan di {out_path}")
    return faces

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = 'data/raw_frames/sample1.jpg'
    detect_anime_faces(img_path)