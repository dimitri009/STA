"""
P&ID Symbol Detection and OCR Pipeline
=======================================
Detects engineering symbols and associates OCR text (equipment tags)
with detected symbols on a P&ID diagram image.

Usage:
    python pid_pipeline.py --image path/to/diagram.jpg --yaml dataset.yaml --weights model/checkpoint_best_total.pth

Dependencies:
    pip install paddleocr rfdetr sahi scipy opencv-python-headless pillow pyyaml matplotlib
"""

import argparse
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from paddleocr import PaddleOCR
from rfdetr import RFDETRBase
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from scipy.optimize import linear_sum_assignment
# ----
import os
import json
import pandas as pd

from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# OCR quality thresholds
OCR_DET_THRESH = 0.75
OCR_REC_THRESH = 0.75

# Tiling parameters
TILE_SIZE = 1280
TILE_OVERLAP = 640

# Duplicate-detection thresholds (clustering step)
IOU_THRESHOLD = 0.20
CONTAINMENT_THRESHOLD = 0.60

# Line-merging thresholds (join vertically adjacent OCR boxes)
MERGE_VERTICAL_GAP_PX = 10
MERGE_X_OVERLAP_RATIO = 0.80

# Symbol detection
SYMBOL_DETECTION_CONFIDENCE = 0.75
SYMBOL_SLICE_SIZE = 1280
SYMBOL_SLICE_OVERLAP = 0.2

# Association
ASSOCIATION_COST_CAP = 5000
ASSOCIATION_SEARCH_MARGIN = 20

# Valid equipment-tag pattern  (e.g. FIC-101, XV-12A)
TAG_REGEX = re.compile(r"^[A-Z]{1,5}-?\d{1,5}[A-Z]?$")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image from disk and return as an RGB NumPy array."""
    img = Image.open(path)
    return np.array(img)


def crop_diagram(image: np.ndarray) -> np.ndarray:
    """
    Crop away the drawing border / title block.
    Adjust the slice values to match your document layout.
    """
    return image[160:-160, 290:-1500]


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def generate_tiles(image: np.ndarray, tile_size: int, overlap: int) -> list[dict]:
    """
    Divide *image* into overlapping tiles of *tile_size* × *tile_size* pixels.
    The last column and row of tiles are snapped to the image edge so no pixels
    are missed.

    Returns a list of dicts with keys: image, x_offset, y_offset.
    """
    h, w = image.shape[:2]
    step = tile_size - overlap

    def _anchors(dim: int) -> list[int]:
        positions = list(range(0, max(1, dim - tile_size + 1), step))
        # Ensure the last tile always reaches the edge
        last = max(0, dim - tile_size)
        if not positions or positions[-1] != last:
            positions.append(last)
        return positions

    tiles = []
    for y in _anchors(h):
        for x in _anchors(w):
            tiles.append({
                "image": image[y : y + tile_size, x : x + tile_size],
                "x_offset": x,
                "y_offset": y,
            })
    return tiles


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def build_ocr_engine() -> PaddleOCR:
    """Initialise PaddleOCR with the project settings."""
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        #text_det_thresh=OCR_DET_THRESH,
        #text_rec_score_thresh=OCR_REC_THRESH,
    )


def run_ocr_on_tiles(tiles: list[dict], ocr_engine: PaddleOCR) -> list[dict]:
    """
    Run OCR on every tile and return a flat list of detections, each with
    absolute coordinates in the full-image space.
    """
    detections = []
    for tile in tqdm(
            tiles,
            desc="OCR tiles",
            unit="tile"):
        result = ocr_engine.predict(tile["image"])
        x_off, y_off = tile["x_offset"], tile["y_offset"]

        for txt, score, poly in zip(
            result[0]["rec_texts"],
            result[0]["rec_scores"],
            result[0]["rec_polys"],
        ):
            poly = np.array(poly)
            x1, y1 = int(poly[:, 0].min()), int(poly[:, 1].min())
            x2, y2 = int(poly[:, 0].max()), int(poly[:, 1].max())

            detections.append({
                "text": txt,
                "score": float(score),
                "bbox": [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off],
                "center": ((x1 + x2) / 2 + x_off, (y1 + y2) / 2 + y_off),
            })

    return detections


# ---------------------------------------------------------------------------
# OCR post-processing: remove duplicates produced by tile overlap
# ---------------------------------------------------------------------------

def _bbox_iou(a: list, b: list) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    if xB <= xA or yB <= yA:
        return 0.0
    inter = (xB - xA) * (yB - yA)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _containment(small: list, large: list) -> float:
    """Fraction of *small* that is covered by *large*."""
    xA, yA = max(small[0], large[0]), max(small[1], large[1])
    xB, yB = min(small[2], large[2]), min(small[3], large[3])
    if xB <= xA or yB <= yA:
        return 0.0
    area_small = max(0, small[2] - small[0]) * max(0, small[3] - small[1])
    return (xB - xA) * (yB - yA) / area_small if area_small else 0.0


def _are_same_region(a: dict, b: dict) -> bool:
    iou = _bbox_iou(a["bbox"], b["bbox"])
    c1 = _containment(a["bbox"], b["bbox"])
    c2 = _containment(b["bbox"], a["bbox"])
    return iou > IOU_THRESHOLD or c1 > CONTAINMENT_THRESHOLD or c2 > CONTAINMENT_THRESHOLD


def _connected_components(n: int, neighbours: list[list[int]]) -> list[list[int]]:
    visited = [False] * n
    groups = []
    for start in range(n):
        if visited[start]:
            continue
        stack, component = [start], []
        while stack:
            k = stack.pop()
            if visited[k]:
                continue
            visited[k] = True
            component.append(k)
            stack.extend(neighbours[k])
        groups.append(component)
    return groups


def deduplicate_ocr(detections: list[dict]) -> list[dict]:
    """
    Cluster overlapping detections (artefacts of tiled inference) and keep
    the single best detection per cluster — the one with the highest
    score × text-length product.
    """
    n = len(detections)
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _are_same_region(detections[i], detections[j]):
                neighbours[i].append(j)
                neighbours[j].append(i)

    groups = _connected_components(n, neighbours)
    cleaned = []
    for group in groups:
        best = max(
            (detections[i] for i in group),
            key=lambda d: len(d["text"]) * d.get("score", 1.0),
        )
        cleaned.append(best)

    return cleaned


# ---------------------------------------------------------------------------
# OCR post-processing: merge vertically adjacent boxes into single labels
# ---------------------------------------------------------------------------

def _vertical_gap(a: dict, b: dict) -> int:
    """Pixel gap between the bottom of the upper box and the top of the lower box."""
    if a["bbox"][1] > b["bbox"][1]:
        a, b = b, a
    return b["bbox"][1] - a["bbox"][3]


def _x_overlap_ratio(a: dict, b: dict) -> float:
    ax1, ax2 = a["bbox"][0], a["bbox"][2]
    bx1, bx2 = b["bbox"][0], b["bbox"][2]
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    min_width = min(ax2 - ax1, bx2 - bx1)
    return overlap / min_width if min_width else 0.0


def _should_merge(a: dict, b: dict) -> bool:
    return (
        _vertical_gap(a, b) <= MERGE_VERTICAL_GAP_PX
        and _x_overlap_ratio(a, b) > MERGE_X_OVERLAP_RATIO
    )


def merge_adjacent_lines(detections: list[dict]) -> list[dict]:
    """Merge vertically adjacent OCR boxes that belong to the same label."""
    n = len(detections)
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _should_merge(detections[i], detections[j]):
                neighbours[i].append(j)
                neighbours[j].append(i)

    groups = _connected_components(n, neighbours)
    merged = []
    for group in groups:
        boxes = [detections[i] for i in group]
        x1 = min(b["bbox"][0] for b in boxes)
        y1 = min(b["bbox"][1] for b in boxes)
        x2 = max(b["bbox"][2] for b in boxes)
        y2 = max(b["bbox"][3] for b in boxes)
        # Sort top-to-bottom, left-to-right before joining text
        boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        merged.append({
            "text": " ".join(b["text"] for b in boxes),
            "score": max(b["score"] for b in boxes),
            "bbox": [x1, y1, x2, y2],
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "count": len(boxes),
        })
    return merged


# ---------------------------------------------------------------------------
# Symbol detection
# ---------------------------------------------------------------------------

def load_class_mapping(yaml_path: str) -> dict[int, str]:
    """Load the id→class-name mapping from a YOLO dataset YAML."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return {int(k): v for k, v in cfg["names"].items()}


def build_detection_model(weights_path: str, id_to_name: dict) -> AutoDetectionModel:
    model = RFDETRBase(pretrain_weights=weights_path)
    return AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model=model,
        confidence_threshold=SYMBOL_DETECTION_CONFIDENCE,
        category_mapping=id_to_name,
        device="cuda:0",
    )


def detect_symbols(image: np.ndarray, detection_model: AutoDetectionModel) -> list[dict]:
    """Run sliced inference and return a flat list of symbol detections."""
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=SYMBOL_SLICE_SIZE,
        slice_width=SYMBOL_SLICE_SIZE,
        overlap_height_ratio=SYMBOL_SLICE_OVERLAP,
        overlap_width_ratio=SYMBOL_SLICE_OVERLAP,
    )

    symbols = []
    for idx, obj in enumerate(result.object_prediction_list):
        b = obj.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        symbols.append({
            "id": idx,
            "class": obj.category.name,
            "score": float(obj.score.value),
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            "width": x2 - x1,
            "height": y2 - y1,
        })
    return symbols


# ---------------------------------------------------------------------------
# Symbol ↔ text association
# ---------------------------------------------------------------------------

def _expand_bbox(bbox: list, margin: int) -> list:
    x1, y1, x2, y2 = bbox
    return [x1 - margin, y1 - margin, x2 + margin, y2 + margin]


def _intersection_area(a: list, b: list) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    if xB <= xA or yB <= yA:
        return 0.0
    return float((xB - xA) * (yB - yA))


def _text_inside_symbol(sym_bbox: list, txt_bbox: list) -> bool:
    tx1, ty1, tx2, ty2 = txt_bbox
    sx1, sy1, sx2, sy2 = sym_bbox
    return tx1 >= sx1 and ty1 >= sy1 and tx2 <= sx2 and ty2 <= sy2


def _is_valid_tag(text: str) -> bool:
    return TAG_REGEX.fullmatch(text) is not None


def _association_cost(symbol: dict, text_obj: dict) -> float:
    """
    Lower cost = better match.
    Returns a large sentinel (1e9) when the text is completely outside the
    symbol's search zone.
    """
    expanded = _expand_bbox(symbol["bbox"], ASSOCIATION_SEARCH_MARGIN)
    if _intersection_area(expanded, text_obj["bbox"]) == 0:
        return 1e9

    sx, sy = symbol["center"]
    tx, ty = text_obj["center"]
    dy = ty - sy
    cost = float(np.hypot(tx - sx, dy))

    if _text_inside_symbol(symbol["bbox"], text_obj["bbox"]):
        cost -= 1000           # strong bonus for enclosed text
    if _is_valid_tag(text_obj["text"]):
        cost -= 200            # bonus for tag-like strings
    else:
        cost += 300
    if len(text_obj["text"]) > 1:
        cost -= 600            # prefer multi-character labels
    else:
        cost += 300
    cost += abs(dy)            # prefer horizontally aligned text

    return cost


def associate_symbols_to_text(
    symbols: list[dict],
    texts: list[dict],
) -> list[dict]:
    """
    Use the Hungarian algorithm to find the globally optimal one-to-one
    assignment between detected symbols and OCR text boxes.

    Pairs whose optimal cost exceeds ASSOCIATION_COST_CAP are discarded.
    """
    n_sym, n_txt = len(symbols), len(texts)
    cost_matrix = np.zeros((n_sym, n_txt))
    for i, sym in enumerate(
            tqdm(
                symbols,
                desc="Building cost matrix",
                unit="symbol"
            )
    ):
        for j, txt in enumerate(texts):
            cost_matrix[i, j] = _association_cost(sym, txt)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    associations = []
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost > ASSOCIATION_COST_CAP:
            continue
        associations.append({
            "symbol_id": symbols[r]["id"],
            "class": symbols[r]["class"],
            "tag": texts[c]["text"],
            "cost": float(cost),
            # Keep full objects for visualisation
            "_symbol": symbols[r],
            "_text": texts[c],
        })

    return associations

def save_associations_excel(
    associations,
    output_file,
):

    rows = []

    for assoc in associations:

        sym = assoc["_symbol"]
        txt = assoc["_text"]

        rows.append({

            "symbol_id":
                assoc["symbol_id"],

            "symbol_class":
                assoc["class"],

            "symbol_confidence":
                sym["score"],

            "symbol_bbox":
                str(sym["bbox"]),

            "tag":
                assoc["tag"],

            "ocr_confidence":
                txt["score"],

            "tag_bbox":
                str(txt["bbox"]),

            "association_cost":
                assoc["cost"],
        })

    df = pd.DataFrame(rows)

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Associations",
            index=False,
        )


def save_associations_json(
    associations,
    output_file,
):

    export = []

    for assoc in associations:

        sym = assoc["_symbol"]
        txt = assoc["_text"]

        export.append({

            "symbol_id":
                assoc["symbol_id"],

            "class":
                assoc["class"],

            "symbol_confidence":
                sym["score"],

            "symbol_bbox":
                sym["bbox"],

            "tag":
                assoc["tag"],

            "ocr_confidence":
                txt["score"],

            "tag_bbox":
                txt["bbox"],

            "association_cost":
                assoc["cost"],
        })

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            export,
            f,
            indent=2,
        )
# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def visualise_symbols(
    image: np.ndarray,
    symbols: list[dict],
    save_path: str = None,
):

    viz = image.copy()

    for s in symbols:

        x1, y1, x2, y2 = s["bbox"]

        cv2.rectangle(
            viz,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3,
        )

        label = f"{s['class']} {s['score']:.2f}"

        cv2.putText(
            viz,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    plt.figure(figsize=(20,20))
    plt.imshow(viz)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    #plt.show()
    plt.close()


def visualise_ocr(
    image: np.ndarray,
    detections: list[dict],
    title: str = "OCR detections",
    save_path: str | None = None,
) -> None:

    viz = image.copy()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            viz,
            d["text"],
            (x1, max(50, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

    plt.figure(figsize=(20, 20))
    plt.imshow(viz)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    #plt.show()
    plt.close()


def visualise_associations(
    image: np.ndarray,
    associations: list[dict],
    save_path: str | None = None,
) -> None:

    viz = image.copy()

    for assoc in associations:

        sym = assoc["_symbol"]
        txt = assoc["_text"]

        sx, sy = map(int, sym["center"])
        tx, ty = map(int, txt["center"])

        x1, y1, x2, y2 = sym["bbox"]
        tx1, ty1, tx2, ty2 = txt["bbox"]

        cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.rectangle(viz, (tx1, ty1), (tx2, ty2), (0, 255, 0), 3)

        cv2.line(viz, (sx, sy), (tx, ty), (255, 255, 0), 3)

        cv2.putText(
            viz,
            txt["text"],
            (sx, sy - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

    plt.figure(figsize=(20, 20))
    plt.imshow(viz)
    plt.axis("off")
    plt.title("Final Associations")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    #plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P&ID OCR + Symbol Detection Pipeline")
    parser.add_argument("--image",   required=True, help="Path to the input P&ID image")
    parser.add_argument("--yaml",    required=True, help="Path to dataset.yaml (class names)")
    parser.add_argument("--weights", required=True, help="Path to RF-DETR checkpoint (.pth)")
    parser.add_argument("--no-viz",  action="store_true", help="Skip all visualisation windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()


    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/6] Loading image and class mapping …")
    id_to_name = load_class_mapping(args.yaml)
    image = load_image(args.image)
    image_name = os.path.splitext(
        os.path.basename(args.image)
    )[0]

    output_dir = os.path.join(
        "results",
        image_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print(
        f"Output folder: {output_dir}"
    )
    diagram = crop_diagram(image)
    print(f"      Diagram size: {diagram.shape[1]}×{diagram.shape[0]} px")

    # ------------------------------------------------------------------
    # 2. OCR
    # ------------------------------------------------------------------
    print("[2/6] Running OCR …")
    ocr_engine = build_ocr_engine()
    tiles = generate_tiles(diagram, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
    print(f"      Tiles generated: {len(tiles)}")
    raw_ocr = run_ocr_on_tiles(tiles, ocr_engine)
    print(f"      Raw detections:  {len(raw_ocr)}")

    if not args.no_viz:
        #visualise_ocr(diagram, raw_ocr, title="Raw OCR detections")
        visualise_ocr(
            diagram,
            raw_ocr,
            title="Raw OCR detections",
            save_path=os.path.join(
                output_dir,
                "01_raw_ocr.png"
            )
        )

    # ------------------------------------------------------------------
    # 3. OCR post-processing
    # ------------------------------------------------------------------
    print("[3/6] Deduplicating and merging OCR results …")
    deduped = deduplicate_ocr(raw_ocr)
    merged_texts = merge_adjacent_lines(deduped)
    print(f"      After dedup:  {len(deduped)}")
    print(f"      After merge:  {len(merged_texts)}")

    if not args.no_viz:
        #visualise_ocr(diagram, merged_texts, title="Cleaned OCR detections")
        visualise_ocr(
            diagram,
            merged_texts,
            title="Cleaned OCR detections",
            save_path=os.path.join(
                output_dir,
                "02_cleaned_ocr.png"
            )
        )

    # ------------------------------------------------------------------
    # 4. Symbol detection
    # ------------------------------------------------------------------
    print("[4/6] Loading detection model and running symbol detection …")
    det_model = build_detection_model(args.weights, id_to_name)
    symbols = detect_symbols(diagram, det_model)
    visualise_symbols(
        diagram,
        symbols,
        save_path=os.path.join(
            output_dir,
            "03_symbols.png"
        )
    )
    print(f"      Symbols detected: {len(symbols)}")

    # ------------------------------------------------------------------
    # 5. Association
    # ------------------------------------------------------------------
    print("[5/6] Associating symbols with text labels …")
    associations = associate_symbols_to_text(symbols, merged_texts)
    save_associations_excel(
        associations,
        os.path.join(
            output_dir,
            "associations.xlsx"
        )
    )

    save_associations_json(
        associations,
        os.path.join(
            output_dir,
            "associations.json"
        )
    )
    print(f"      Associations found: {len(associations)}")

    # ------------------------------------------------------------------
    # 6. Output
    # ------------------------------------------------------------------
    print("[6/6] Results:")
    print(f"{'Symbol ID':>10}  {'Class':<20}  {'Tag':<20}  {'Cost':>8}")
    print("-" * 65)
    for a in associations:
        print(f"{a['symbol_id']:>10}  {a['class']:<20}  {a['tag']:<20}  {a['cost']:>8.1f}")

    if not args.no_viz:
        #visualise_associations(diagram, associations)
        visualise_associations(
            diagram,
            associations,
            save_path=os.path.join(
                output_dir,
                "04_associations.png"
            )
        )


if __name__ == "__main__":
    main()