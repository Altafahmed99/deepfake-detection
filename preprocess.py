# preprocess.py
import os
import cv2
from mtcnn import MTCNN

detector = MTCNN()

def crop_and_save(input_path, out_path, size=(224,224)):
    img = cv2.imread(input_path)
    if img is None:
        return False
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    if not faces:
        return False
    x,y,w,h = faces[0]['box']
    x,y = max(0,x), max(0,y)
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, size)
    cv2.imwrite(out_path, face)
    return True

def process_folder(src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for fname in os.listdir(src_folder):
        inpath = os.path.join(src_folder, fname)
        outpath = os.path.join(dest_folder, fname)
        try:
            ok = crop_and_save(inpath, outpath)
            if not ok:
                print("No face:", inpath)
        except Exception as e:
            print("Error:", inpath, e)

if __name__ == "__main__":
    # Example: prepare train real/fake from raw folders
    # Edit these paths for your dataset
    process_folder("raw_data/train/real_raw", "data/train/real")
    process_folder("raw_data/train/fake_raw", "data/train/fake")
    process_folder("raw_data/val/real_raw", "data/val/real")
    process_folder("raw_data/val/fake_raw", "data/val/fake")
    process_folder("raw_data/test/real_raw", "data/test/real")
    process_folder("raw_data/test/fake_raw", "data/test/fake")
    print("Done preprocessing.")
