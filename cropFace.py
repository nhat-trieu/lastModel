#!pip install retina-face

import cv2 
from retinaface import RetinaFace 
import os 
from tqdm import tqdm 
import sys 
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SOURCE_DIR  = '/kaggle/input/datasets/trieung11/gaf2000/GAF_2'
OUTPUT_DIR  = '/kaggle/working/GAF2_Full_Face_Cropped'

DETECTION_THRESHOLD = 0.9

DRAW_BOUNDING_BOX = True
DEBUG_BOX_DIR     = '/kaggle/working/GAF2_Face_BBoxes'

# GAF2 chỉ có Train và Val (không có Test)
SPLITS = ['Train', 'Val']
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(source_base, output_base, model):
    if not os.path.exists(source_base):
        print(f"Không tìm thấy thư mục {source_base}")
        return

    if DRAW_BOUNDING_BOX:
        os.makedirs(DEBUG_BOX_DIR, exist_ok=True)

    total_image_scanned = 0
    total_face_saved    = 0
    images_no_faces     = 0

    for split in SPLITS:
        split_dir = os.path.join(source_base, split)
        if not os.path.exists(split_dir):
            print(f"⚠️  Không tìm thấy split: {split_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing split: {split}")
        print(f"{'='*60}")

        for root, dirs, files in os.walk(split_dir):
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                continue

            relative_path = os.path.relpath(root, source_base)
            output_save_dir = os.path.join(output_base, relative_path)
            os.makedirs(output_save_dir, exist_ok=True)

            debug_save_dir = None
            if DRAW_BOUNDING_BOX:
                debug_save_dir = os.path.join(DEBUG_BOX_DIR, relative_path)
                os.makedirs(debug_save_dir, exist_ok=True)

            for filename in tqdm(image_files, desc=f"  {relative_path}"):
                total_image_scanned += 1
                img_path = os.path.join(root, filename)
                base_filename, file_extension = os.path.splitext(filename)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img_h, img_w = img.shape[:2]

                    img_for_drawing = None
                    if DRAW_BOUNDING_BOX:
                        img_for_drawing = img.copy()

                    faces = RetinaFace.detect_faces(img, model=model,
                                                    threshold=DETECTION_THRESHOLD)

                    if isinstance(faces, dict) and faces:
                        for count, (key, face_info) in enumerate(faces.items()):
                            facial_area = face_info['facial_area']
                            x1, y1, x2, y2 = (int(c) for c in facial_area)

                            x1_crop = max(0, x1)
                            y1_crop = max(0, y1)
                            x2_crop = min(img_w, x2)
                            y2_crop = min(img_h, y2)

                            cropped_face = img[y1_crop:y2_crop, x1_crop:x2_crop]
                            if cropped_face.size == 0:
                                continue

                            # ── Lưu ảnh crop ──────────────────────────────
                            output_filename = f"{base_filename}_face_{count+1}{file_extension}"
                            output_path     = os.path.join(output_save_dir, output_filename)

                            if not os.path.exists(output_path):
                                cv2.imwrite(output_path, cropped_face)
                                total_face_saved += 1

                                # ── Lưu bbox [x1,y1,x2,y2,img_w,img_h] ──
                                bbox_info     = np.array([x1, y1, x2, y2, img_w, img_h],
                                                         dtype=np.float32)
                                bbox_filename = f"{base_filename}_face_{count+1}_bbox.npy"
                                bbox_path     = os.path.join(output_save_dir, bbox_filename)
                                np.save(bbox_path, bbox_info)

                            if DRAW_BOUNDING_BOX:
                                cv2.rectangle(img_for_drawing,
                                              (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img_for_drawing, f"{count+1}",
                                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (0, 0, 255), 2)
                    else:
                        images_no_faces += 1

                    if DRAW_BOUNDING_BOX and img_for_drawing is not None:
                        debug_output_path = os.path.join(debug_save_dir, filename)
                        cv2.imwrite(debug_output_path, img_for_drawing)

                except Exception as e:
                    print(f"  ❌ Error {img_path}: {e}")

    print("\n" + "="*60)
    print("  DONE FACE DETECTION — GAF2")
    print("="*60)
    print(f"  Scanned : {total_image_scanned}")
    print(f"  Saved   : {total_face_saved}")
    print(f"  No faces: {images_no_faces}")
    print(f"  Output  : {output_base}")


if __name__ == '__main__':
    try:
        print("Loading RetinaFace model...")
        model = RetinaFace.build_model()
        process_dataset(SOURCE_DIR, OUTPUT_DIR, model)
    except Exception as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)