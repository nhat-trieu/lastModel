"""
figure2panel.py  (v2 - clean)
==============================
Output:
  - face_crops/  : thư mục chứa toàn bộ face crops (đánh số)
  - face_sheet.png : preview tất cả crops với label bên dưới
  - figure1.pdf / figure1.png : figure 2-panel cho paper

Chạy:
    python figure2panel.py \
        --image Picture_00820.jpg \
        --npz   Picture_00820.npz \
        --label Neutral \
        --output /kaggle/working/figure1.pdf
"""

import argparse, os, warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

LABEL_COLOR = {'Positive': '#27ae60', 'Neutral': '#2980b9', 'Negative': '#c0392b'}
POSITIVE_EM  = {'happy', 'surprise'}
NEGATIVE_EM  = {'angry', 'disgust', 'fear', 'sad'}

def map_emotion(e):
    e = e.lower()
    if e in POSITIVE_EM:  return 'Positive'
    if e in NEGATIVE_EM:  return 'Negative'
    return 'Neutral'

def predict_fer(crops):
    from deepface import DeepFace
    results = []
    for i, crop in enumerate(crops):
        try:
            arr = np.array(crop.convert('RGB'))
            a   = DeepFace.analyze(arr, actions=['emotion'],
                                   enforce_detection=False, silent=True)
            if isinstance(a, list): a = a[0]
            dom  = a['dominant_emotion']
            conf = a['emotion'][dom] / 100.0
            lbl  = map_emotion(dom)
        except:
            dom, conf, lbl = 'neutral', 0.0, 'Neutral'
        print(f"  Face {i:2d}: {dom:10s} → {lbl} ({conf:.2f})")
        results.append((lbl, conf, dom))
    return results

def main(image_path, npz_path, group_label, output_path):
    img  = Image.open(image_path).convert('RGB')
    iw, ih = img.size
    data = np.load(npz_path)
    boxes = data['boxes']

    # ── Crop usable faces ──────────────────────────────────────
    MIN_W, MIN_H, PAD = 50, 65, 12
    usable = []
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = [int(v) for v in b]
        if (x2-x1) >= MIN_W and (y2-y1) >= MIN_H:
            crop = img.crop((max(0,x1-PAD), max(0,y1-PAD),
                             min(iw,x2+PAD), min(ih,y2+PAD)))
            crop = crop.resize((128,128), Image.LANCZOS)
            usable.append((i, crop))
    print(f"Usable faces: {len(usable)}")
    crops = [c for _,c in usable]

    # ── Save individual crops ──────────────────────────────────
    out_dir = os.path.join(os.path.dirname(output_path), 'face_crops')
    os.makedirs(out_dir, exist_ok=True)
    for idx, (orig_i, crop) in enumerate(usable):
        crop.save(os.path.join(out_dir, f'face_{idx:02d}_orig{orig_i}.jpg'))

    # ── Predict FER ────────────────────────────────────────────
    print("\nRunning DeepFace...")
    preds = predict_fer(crops)

    # ── Select best 3 per class ────────────────────────────────
    buckets = {'Positive':[], 'Neutral':[], 'Negative':[]}
    for (orig_i, crop), (lbl, conf, raw) in zip(usable, preds):
        w,h = crop.size
        score = conf * (min(w,h)/128.0)
        buckets[lbl].append((score, crop, lbl, conf, raw, orig_i))
    for cls in buckets:
        buckets[cls].sort(key=lambda x: -x[0])

    selected = {}
    for cls in ('Positive','Neutral','Negative'):
        selected[cls] = buckets[cls][:3]

    # ── Preview sheet: all crops ───────────────────────────────
    n = len(crops)
    ncols = 8
    nrows = (n + ncols - 1) // ncols
    fig_s, axes_s = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2.4))
    fig_s.patch.set_facecolor('white')
    axes_s = axes_s.flat
    for ax, (orig_i, crop), (lbl, conf, raw) in zip(axes_s, usable, preds):
        ax.imshow(np.array(crop))
        ax.axis('off')
        ax.set_title(f'#{orig_i}\n{lbl}\n({raw[:4]})', fontsize=7,
                     color=LABEL_COLOR[lbl], pad=2)
    for ax in list(axes_s)[n:]:
        ax.axis('off')
    plt.tight_layout(pad=0.5)
    sheet_path = output_path.replace('.pdf','.sheet.png')
    fig_s.savefig(sheet_path, dpi=150, bbox_inches='tight')
    plt.close(fig_s)
    print(f"Saved crop sheet → {sheet_path}")

    # ── Main figure ────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5.5), facecolor='white')
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.5,1],
                            wspace=0.05, left=0.01, right=0.99,
                            top=0.90, bottom=0.12)

    # Left: group scene (clean, no boxes, no overlay text)
    ax_scene = fig.add_subplot(gs[0,0])
    ax_scene.imshow(np.array(img))
    ax_scene.axis('off')
    ax_scene.set_xlabel("(a) Group emotion recognition",
                         fontsize=11, fontweight='bold', labelpad=6)

    # Right: 3×3 grid
    gs_r = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0,1],
                                            hspace=0.6, wspace=0.12)
    order = ('Positive','Neutral','Negative')
    for r, cls in enumerate(order):
        faces = selected[cls]
        for c in range(3):
            ax = fig.add_subplot(gs_r[r, c])
            if c < len(faces):
                _, crop, lbl, conf, raw, orig_i = faces[c]
                ax.imshow(np.array(crop))
                for sp in ax.spines.values():
                    sp.set_edgecolor(LABEL_COLOR[lbl]); sp.set_linewidth(2.5)
                ax.set_xticks([]); ax.set_yticks([])
                # Label BELOW the image
                ax.set_xlabel(lbl, fontsize=8, color=LABEL_COLOR[lbl],
                              fontweight='bold', labelpad=3)
            else:
                ax.axis('off')

    fig.text(0.745, 0.07, "(b) Facial expression recognition",
             ha='center', fontsize=11, fontweight='bold')

    caption = (
        "Fig. 1.  Comparison of GER and FER in the wild. "
        "(a) GER of a crowd in a traditional festival scene, and "
        "(b) FER of independent faces in the scene. "
        "Although the overall group emotion is Neutral, individual facial "
        "expressions vary (Positive / Neutral / Negative), demonstrating "
        "that relying solely on FER is insufficient for accurate group emotion recognition."
    )
    fig.text(0.5, 0.01, caption, ha='center', va='bottom',
             fontsize=8, style='italic', color='#333')

    for ext in [output_path, output_path.replace('.pdf','.png')]:
        fig.savefig(ext, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved → {ext}")
    plt.close(fig)
    print("✅ Done!")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image',  default='Picture_00820.jpg')
    p.add_argument('--npz',    default='Picture_00820.npz')
    p.add_argument('--label',  default='Neutral',
                   choices=['Positive','Neutral','Negative'])
    p.add_argument('--output', default='figure1.pdf')
    a = p.parse_args()
    main(a.image, a.npz, a.label, a.output)