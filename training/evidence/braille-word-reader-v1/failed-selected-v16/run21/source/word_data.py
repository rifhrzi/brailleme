"""Auditable Angelina cell/word inputs. No model predictions enter fixture selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from benchmark_angelina import LETTER_BY_MASK, PINNED_REVISION, read_split, validate_dataset_revision

WIDTH, HEIGHT, STRIDE = 480, 192, 4
GRID_W, GRID_H = WIDTH // STRIDE, HEIGHT // STRIDE
VALIDATION_FAMILIES = {"books/ola", "handwritten/ang_redmi"}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_page(root, path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    rows = np.loadtxt(path.with_suffix(".csv"), delimiter=";", ndmin=2)
    boxes = rows[:, :4] * np.array([image.width, image.height] * 2)
    masks = rows[:, 4].astype(int)
    if np.any((masks < 1) | (masks > 63)):
        raise ValueError(f"Invalid six-dot masks: {path}")
    return image, boxes.astype(np.float32), masks


def lines_from_boxes(boxes):
    """Cluster using local vertical proximity, then robustly fit each sloped line."""
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    heights = boxes[:, 3] - boxes[:, 1]
    pending = set(range(len(boxes)))
    lines = []
    while pending:
        first = min(pending, key=lambda i: centers[i, 1])
        members = [first]
        pending.remove(first)
        changed = True
        while changed:
            changed = False
            for index in sorted(pending):
                neighbors = sorted(members, key=lambda i: abs(centers[index, 0] - centers[i, 0]))[:3]
                nearest = neighbors[0]
                y = centers[nearest, 1]
                if len(members) >= 3:
                    xs = centers[members, 0]
                    if np.ptp(xs) > np.median(heights[members]) * 2:
                        slope, intercept = np.polyfit(xs, centers[members, 1], 1)
                        if abs(slope) <= .15:
                            y = slope * centers[index, 0] + intercept
                if abs(centers[index, 1] - y) < .48 * min(heights[index], heights[nearest]):
                    members.append(index)
                    pending.remove(index)
                    changed = True
        lines.append(sorted(members, key=lambda i: centers[i, 0]))
    return lines


def word_groups(boxes):
    """Word boundaries are explicit space-sized gaps, never arbitrary substrings.

    Angelina labels cells, not words. We retain an auditable geometric word
    fixture definition and label it as such rather than claim lexical ground truth.
    """
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    result = []
    lines = lines_from_boxes(boxes)
    heights = boxes[:, 3]-boxes[:, 1]
    normalized_by_line = []
    for line in lines:
        normalized_by_line.append(np.asarray([(centers[b, 0]-centers[a, 0])/max((heights[a]+heights[b])/2, 1e-6) for a, b in zip(line, line[1:])]))
    all_gaps = np.concatenate(normalized_by_line) if normalized_by_line else np.asarray([])
    def typical_pitch(gaps):
        ordered = np.sort(gaps[gaps>0])
        # Cell pitch is the lower distribution of observed center gaps; spaces
        # add a whole empty cell. Box width is not a reliable pitch proxy since
        # annotation margins and the gap between cells vary across pages.
        return float(np.median(ordered[:max(1, int(np.ceil(len(ordered)*.6)))])) if len(ordered) else .85
    page_pitch = typical_pitch(all_gaps)
    for line, normalized_gaps in zip(lines, normalized_by_line):
        if len(line) == 1:
            result.append(line)
            continue
        step = typical_pitch(normalized_gaps) if len(normalized_gaps)>=3 else page_pitch
        group = [line[0]]
        for index, gap in zip(line[1:], normalized_gaps):
            if gap > step * 1.55:
                result.append(group)
                group = []
            group.append(index)
        result.append(group)
    return result


def crop_transform(image, boxes, masks, crop):
    left, top, right, bottom = crop
    patch = image.crop(crop).resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    transformed = (boxes - np.array([left, top] * 2)) * np.array([WIDTH / (right-left), HEIGHT / (bottom-top)] * 2)
    intersects = (transformed[:, 2] > 0) & (transformed[:, 3] > 0) & (transformed[:, 0] < WIDTH) & (transformed[:, 1] < HEIGHT)
    return patch, transformed[intersects], masks[intersects]


def estimate_page_pitch(boxes):
    """Typical same-row center pitch normalized by cell height."""
    centers = (boxes[:, :2]+boxes[:, 2:])/2
    heights = boxes[:, 3]-boxes[:, 1]
    ratios = []
    for index in range(len(boxes)):
        dx = centers[:, 0]-centers[index, 0]
        scale = (heights+heights[index])/2
        close_row = np.abs(centers[:, 1]-centers[index, 1]) < scale*.65
        choices = np.flatnonzero((dx>0) & close_row)
        if len(choices):
            neighbor = min(choices, key=lambda j: dx[j])
            ratios.append(float(dx[neighbor]/scale[neighbor]))
    ratios = sorted(ratios)
    return float(np.median(ratios[:max(1, int(np.ceil(len(ratios)*.6)))])) if ratios else .85


def wrapped_line_starts(boxes, masks):
    """A line-leading continuation after a terminal hyphen is not a whole word."""
    centers = (boxes[:, :2]+boxes[:, 2:])/2
    lines = lines_from_boxes(boxes)
    lines.sort(key=lambda line: float(np.median(centers[line, 1])))
    starts = set()
    for previous, current in zip(lines, lines[1:]):
        terminal = max(previous, key=lambda i: centers[i, 0])
        if int(masks[terminal]) == 36:  # dots 3+6, '-' in Angelina src/letters.py
            starts.add(min(current, key=lambda i: centers[i, 0]))
    return starts


def boundary_guard(boxes, group, image_width, page_pitch=None):
    """Check word ends against every annotation, independent of line clustering.

    Exclude uncertain labels instead of declaring a fragment a word. No model
    prediction or image clarity enters this check.
    """
    centers = (boxes[:, :2]+boxes[:, 2:])/2
    heights = boxes[:, 3]-boxes[:, 1]
    members = set(group)
    ordered = sorted(group, key=lambda i: centers[i, 0])
    page_pitch = estimate_page_pitch(boxes) if page_pitch is None else page_pitch
    if len(ordered)>=2:
        internal = [(centers[b, 0]-centers[a, 0])/((heights[a]+heights[b])/2) for a, b in zip(ordered, ordered[1:])]
        pitch = max(page_pitch, float(np.median(internal)))
    else:
        pitch = page_pitch
    for edge, direction in ((ordered[0], -1), (ordered[-1], 1)):
        for other in range(len(boxes)):
            if other in members:
                continue
            scale = (heights[edge]+heights[other])/2
            dx = direction*(centers[other, 0]-centers[edge, 0])
            if 0 < dx <= 1.55*pitch*scale and abs(centers[other, 1]-centers[edge, 1]) < .65*scale:
                return "uncertain_boundary_neighbor"
    if boxes[ordered[0], 0] < .75*pitch*heights[ordered[0]] or image_width-boxes[ordered[-1], 2] < .75*pitch*heights[ordered[-1]]:
        return "uncertain_image_edge_boundary"
    return None


def make_target(boxes, masks, originally_complete=None, partial_geometry_supervision=False, ignore_boxes=None, visible_partial_boxes=None):
    target = np.zeros((GRID_H, GRID_W, 11), np.float32)
    intersects = (boxes[:, 2]>0) & (boxes[:, 3]>0) & (boxes[:, 0]<WIDTH) & (boxes[:, 1]<HEIGHT)
    boxes, masks = boxes[intersects], masks[intersects]
    full = (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] <= WIDTH) & (boxes[:, 3] <= HEIGHT)
    if originally_complete is not None:
        full &= np.asarray(originally_complete)[intersects]
    # A cut-off real cell is not background: omit its visible pixels from the
    # objectness loss. Complete cells below overwrite the ignored region.
    ignored = np.asarray(ignore_boxes if ignore_boxes is not None else [], np.float32).reshape(-1, 4)
    ignored = ignored[(ignored[:, 2]>0)&(ignored[:, 3]>0)&(ignored[:, 0]<WIDTH)&(ignored[:, 1]<HEIGHT)]
    # Unlisted supplemental grid slots have unknown truth. They may mask
    # objectness loss, but must never become a geometry-only partial target.
    for x0, y0, x1, y1 in np.concatenate((boxes[~full], ignored)):
        gx0, gy0 = max(0, int(x0/STRIDE)-1), max(0, int(y0/STRIDE)-1)
        gx1, gy1 = min(GRID_W, int(np.ceil(x1/STRIDE))+1), min(GRID_H, int(np.ceil(y1/STRIDE))+1)
        target[gy0:gy1, gx0:gx1, 0] = -1
    geometry_only = np.zeros(len(boxes), dtype=bool)
    if partial_geometry_supervision:
        centers = (boxes[:, :2]+boxes[:, 2:])/2
        sizes = boxes[:, 2:]-boxes[:, :2]
        actually_crosses_canvas = (boxes[:, 0]<0)|(boxes[:, 1]<0)|(boxes[:, 2]>WIDTH)|(boxes[:, 3]>HEIGHT)
        # Regress the full transformed SOURCE rectangle, never the visible
        # clipped raster rectangle. An off-canvas center or overlarge box cannot
        # be represented by this sigmoid head and remains ignored. A previously
        # cut label that moved fully inside must also remain ignored.
        geometry_only = ((~full)&actually_crosses_canvas
                         &(centers[:, 0]>=0)&(centers[:, 0]<WIDTH)
                         &(centers[:, 1]>=0)&(centers[:, 1]<HEIGHT)
                         &(sizes[:, 0]>0)&(sizes[:, 0]<WIDTH)
                         &(sizes[:, 1]>0)&(sizes[:, 1]<HEIGHT))
    # Complete targets have priority if two centers occupy the same grid point.
    indices = list(np.flatnonzero(geometry_only))+list(np.flatnonzero(full))
    fragments = np.asarray(visible_partial_boxes if visible_partial_boxes is not None else [], np.float64).reshape(-1, 4)
    if len(fragments):
        sizes = fragments[:, 2:]-fragments[:, :2]
        touches_edge = (fragments[:, 0]==0)|(fragments[:, 1]==0)|(fragments[:, 2]==WIDTH)|(fragments[:, 3]==HEIGHT)
        if (not np.isfinite(fragments).all() or not np.all(touches_edge)
                or np.any(fragments[:, :2]<0) or np.any(fragments[:, 2:]>[WIDTH, HEIGHT])
                or np.any(sizes<[4, 12]) or np.any(sizes>=[WIDTH, HEIGHT])):
            raise ValueError('Visible partial targets must be bounded edge fragments at least 4x12 pixels')
    # These separately verified DSBI fragments only teach rejection geometry.
    # Full/known source targets retain priority; every fragment bit is unknown.
    proposals = [(box, None) for box in fragments]
    proposals += [(boxes[index], None if geometry_only[index] else masks[index]) for index in indices]
    for box, mask in proposals:
        x0, y0, x1, y1 = box
        cx, cy = (x0+x1)/2/STRIDE, (y0+y1)/2/STRIDE
        gx, gy = min(GRID_W-1, int(cx)), min(GRID_H-1, int(cy))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if 0 <= gy+dy < GRID_H and 0 <= gx+dx < GRID_W:
                    target[gy+dy, gx+dx, 0] = max(target[gy+dy, gx+dx, 0], np.exp(-(dx*dx+dy*dy)/2))
        bits = [-1]*6 if mask is None else [(int(mask) >> bit) & 1 for bit in range(6)]
        target[gy, gx] = [1, cx-gx, cy-gy, (x1-x0)/WIDTH, (y1-y0)/HEIGHT] + bits
    return target


def affine_patch(image, boxes, scale_x, scale_y, translate_x=0., translate_y=0.):
    """Transform pixels and every annotation together around the image center."""
    offset_x = WIDTH/2*(1-scale_x)+translate_x
    offset_y = HEIGHT/2*(1-scale_y)+translate_y
    background = tuple(np.asarray(image).mean(axis=(0, 1)).astype(int))
    image = image.transform((WIDTH, HEIGHT), Image.Transform.AFFINE,
                            (1/scale_x, 0, -offset_x/scale_x, 0, 1/scale_y, -offset_y/scale_y),
                            Image.Resampling.BILINEAR, fillcolor=background)
    boxes = boxes*np.array([scale_x, scale_y]*2)+np.array([offset_x, offset_y]*2)
    return image, boxes


def mirror_patch_horizontal(image, boxes, masks):
    """Mirror pixels and columns together without reversing vertical relief shading."""
    mirrored = boxes.copy()
    mirrored[:, 0] = WIDTH-boxes[:, 2]
    mirrored[:, 2] = WIDTH-boxes[:, 0]
    remapped = ((masks & 7) << 3) | ((masks >> 3) & 7)
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mirrored, remapped


def rotate_patch_180(image,boxes,masks):
    """Rotate real pixels without changing relief polarity or labeled side."""
    rotated=boxes.copy()
    rotated[:,0]=WIDTH-boxes[:,2]
    rotated[:,1]=HEIGHT-boxes[:,3]
    rotated[:,2]=WIDTH-boxes[:,0]
    rotated[:,3]=HEIGHT-boxes[:,1]
    remapped=sum(((np.asarray(masks,dtype=np.int64)>>bit)&1)<<(5-bit) for bit in range(6))
    return image.transpose(Image.Transpose.ROTATE_180),rotated,remapped


def augment_patch(image, boxes, rng, camera_augmentation=False, polarity_augmentation=False, illumination_augmentation=False, originally_complete=None, point_state=None):
    """Return pixels, boxes, eligibility, and optional tracked occupied points.

    The default three-item return and RNG draws are unchanged. Supplying an
    exact point state adds a fourth return; every geometric raster updates it.
    """
    boxes = boxes.copy()
    def inside_canvas():
        return (boxes[:,0]>=0)&(boxes[:,1]>=0)&(boxes[:,2]<=WIDTH)&(boxes[:,3]<=HEIGHT)
    eligible=inside_canvas()
    if originally_complete is not None:
        original=np.asarray(originally_complete,dtype=bool)
        if original.shape != eligible.shape:
            raise ValueError("Cell eligibility must match every input box")
        eligible &= original
    controlled_light=polarity_augmentation or illumination_augmentation
    if (camera_augmentation or controlled_light) and rng.random()<.85:
        zoom = float(rng.uniform(.8, 1.7) if controlled_light else rng.uniform(.85, 1.35))
        scale_x=zoom if controlled_light else zoom*float(rng.uniform(.55, 1.25))
        translate_x, translate_y = float(rng.uniform(-.04, .04))*WIDTH, float(rng.uniform(-.04, .04))*HEIGHT
        image, boxes = affine_patch(image, boxes, scale_x, zoom, translate_x, translate_y)
        if point_state is not None:
            point_state = point_state.affine(scale_x, zoom, translate_x, translate_y)
        # The raster has already discarded off-canvas pixels. A later rotation
        # can move its annotation inside again, but cannot restore those dots.
        eligible &= inside_canvas()
    if rng.random() < .6:
        angle = float(rng.uniform(-4, 4))
        image = image.rotate(angle, Image.Resampling.BILINEAR, fillcolor=tuple(np.asarray(image).mean(axis=(0, 1)).astype(int)))
        if point_state is not None:
            point_state = point_state.rotate(angle)
        theta = -np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        corners = np.stack((boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]), axis=1)
        rotated = (corners - [WIDTH/2, HEIGHT/2]) @ rotation.T + [WIDTH/2, HEIGHT/2]
        boxes = np.concatenate((rotated.min(axis=1), rotated.max(axis=1)), axis=1)
        eligible &= inside_canvas()
    if controlled_light:
        original=np.asarray(image,dtype=np.float32)
        if rng.random()<.45:
            image=image.filter(ImageFilter.GaussianBlur(float(rng.uniform(.15,2.0))))
        pixels=np.asarray(image,dtype=np.float32)
        # Apply contrast/exposure in float32 with one final clamp, avoiding
        # irreversible intermediate saturation of a faint embossed pattern.
        center=pixels.mean(axis=(0,1),keepdims=True)
        pixels=(pixels-center)*float(rng.uniform(.3,1.6))+float(rng.uniform(.35,.8))*255
        if rng.random()<.4:
            axis=np.linspace(float(rng.uniform(.65,1)),1.,WIDTH,dtype=np.float32)
            if rng.random()<.5:axis=axis[::-1]
            pixels*=axis[None,:,None]
        pixels=np.clip(pixels,0,255)
        # This only bounds a generated augmentation; source samples and labels
        # remain included. Never invent positive patterns in a flattened image.
        if (pixels@np.asarray([.2126,.7152,.0722],np.float32)).std()/255<.01:
            pixels=original
        if polarity_augmentation and rng.random()<.5:
            pixels=255-pixels
        result = (np.clip(pixels,0,255).astype(np.uint8), boxes, eligible)
        return result if point_state is None else result+(point_state,)
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(.65, 1.3)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(.1 if camera_augmentation else .55, 1.6)))
    if rng.random() < (.45 if camera_augmentation else .25):
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(.15, 2.4 if camera_augmentation else .9))))
    pixels = np.asarray(image, dtype=np.float32)
    if rng.random() < .4:
        # Smooth directional illumination, not an opaque synthetic obstruction.
        axis = np.linspace(float(rng.uniform(.65, 1)), 1., WIDTH, dtype=np.float32)
        if rng.random() < .5:
            axis = axis[::-1]
        pixels *= axis[None, :, None]
    result = (np.clip(pixels, 0, 255).astype(np.uint8), boxes, eligible)
    return result if point_state is None else result+(point_state,)


def prepare(args):
    validate_dataset_revision(args.dataset, PINNED_REVISION)
    args.output.mkdir(parents=True, exist_ok=True)
    all_paths = sorted(set(p for split in ("books/train.txt", "books/val.txt", "handwritten/train.txt", "handwritten/val.txt") for p in read_split(args.dataset, split)))
    train = [p for p in all_paths if p.relative_to(args.dataset).parent.as_posix() not in VALIDATION_FAMILIES]
    validation = [p for p in all_paths if p not in train]
    external = read_split(args.dataset, "uploaded/test.txt")
    # Source families and exact image hashes cannot cross training/validation.
    hashes = {split: {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths} for split, paths in (("train", train), ("validation", validation), ("external", external))}
    if hashes["train"] & hashes["validation"] or hashes["train"] & hashes["external"]:
        raise ValueError("Duplicate source image crosses a split")
    split_record = {"dataset_revision": PINNED_REVISION, "seed": args.seed, "validation_families": sorted(VALIDATION_FAMILIES), "images": {k: [p.relative_to(args.dataset).as_posix() for p in v] for k, v in (("train", train), ("validation", validation), ("external", external))}}
    save_json(args.output / "source-split.json", split_record)
    rng = np.random.default_rng(args.seed)
    for split, pages, per_page in (("train", train, args.per_page), ("validation", validation, 40)):
        records = []
        folder = args.output / split
        folder.mkdir(exist_ok=True)
        for page_no, path in enumerate(pages):
            image, boxes, masks = load_page(args.dataset, path)
            lines = lines_from_boxes(boxes)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            for _ in range(per_page):
                line = lines[int(rng.integers(len(lines)))]
                start = int(rng.integers(len(line)))
                chosen = line[start:start+int(rng.integers(1, 13))]
                x0, y0 = boxes[chosen, :2].min(axis=0)
                x1, y1 = boxes[chosen, 2:].max(axis=0)
                height = float(np.median(boxes[chosen, 3] - boxes[chosen, 1]))
                crop_w = max(x1-x0+height*rng.uniform(.6, 1.6), height*rng.uniform(3.0, 4.5))
                crop_h = crop_w * HEIGHT/WIDTH
                cx, cy = (x0+x1)/2+height*rng.uniform(-.15, .15), (y0+y1)/2+height*rng.uniform(-.2, .2)
                crop = [int(round(cx-crop_w/2)), int(round(cy-crop_h/2)), int(round(cx+crop_w/2)), int(round(cy+crop_h/2))]
                if crop[0] < 0 or crop[1] < 0 or crop[2] > image.width or crop[3] > image.height:
                    continue
                patch, transformed, selected_masks = crop_transform(image, boxes, masks, crop)
                filename = f"{len(records):06d}.png"
                patch.save(folder / filename, compress_level=1)
                records.append({"image": f"{split}/{filename}", "source": path.relative_to(args.dataset).as_posix(), "crop": crop, "boxes": transformed.tolist(), "masks": selected_masks.tolist()})
            if page_no % 20 == 0:
                print(f"{split}: {page_no+1}/{len(pages)} pages, {len(records)} patches", flush=True)
        if split == "train":
            for path in read_split(args.dataset, "not_braille/train.txt"):
                with Image.open(path) as image:
                    patch = image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
                filename = f"{len(records):06d}.png"
                patch.save(folder / filename, compress_level=1)
                records.append({"image": f"{split}/{filename}", "source": path.relative_to(args.dataset).as_posix(), "crop": None, "boxes": [], "masks": []})
        save_json(args.output / f"{split}.json", records)
    for split, pages in (("validation", validation), ("external", external)):
        build_word_fixtures(args.dataset, pages, args.output, split)


def build_word_fixtures(root, pages, output, split):
    records, excluded = [], {}
    folder = output / f"{split}-words"
    folder.mkdir(exist_ok=True)
    for path in pages:
        image, boxes, masks = load_page(root, path)
        page_pitch = estimate_page_pitch(boxes)
        wrapped = wrapped_line_starts(boxes, masks)
        for group in word_groups(boxes):
            reason = None
            if not 1 <= len(group) <= 12:
                reason = "outside_length"
            elif any(int(masks[i]) not in LETTER_BY_MASK for i in group):
                reason = "unsupported_mask"
            elif any(index in wrapped for index in group):
                reason = "line_wrap_continuation"
            if not reason:
                reason = boundary_guard(boxes, group, image.width, page_pitch)
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            x0, y0 = boxes[group, :2].min(axis=0)
            x1, y1 = boxes[group, 2:].max(axis=0)
            height = float(np.median(boxes[group, 3]-boxes[group, 1]))
            # Match the user-adjustable guide (aspect 1..8). Its size is fixed
            # from annotation geometry before model inference. The larger 2.5:1
            # context can contain other cells which are outside this selection.
            guide_w = max(float(x1-x0)+height*.6, float(y1-y0)*1.25)
            guide_h = max(float(y1-y0)*1.25, guide_w/8.)
            guide_aspect = guide_w/guide_h
            crop_w = max(guide_w, guide_h*WIDTH/HEIGHT)
            crop_h = crop_w*HEIGHT/WIDTH
            cx, cy = (x0+x1)/2, (y0+y1)/2
            crop = [int(np.floor(cx-crop_w/2)), int(np.floor(cy-crop_h/2)), int(np.ceil(cx+crop_w/2)), int(np.ceil(cy+crop_h/2))]
            guide = [float(cx-guide_w/2), float(cy-guide_h/2), float(cx+guide_w/2), float(cy+guide_h/2)]
            if crop[0]<0 or crop[1]<0 or crop[2]>image.width or crop[3]>image.height:
                excluded["outside_page"] = excluded.get("outside_page", 0)+1
                continue
            patch, transformed, visible_masks = crop_transform(image, boxes, masks, crop)
            visible = (boxes[:, 2]>guide[0]) & (boxes[:, 3]>guide[1]) & (boxes[:, 0]<guide[2]) & (boxes[:, 1]<guide[3])
            visible_indices = set(np.flatnonzero(visible).tolist())
            full_visible = (boxes[:, 0]>=guide[0]) & (boxes[:, 1]>=guide[1]) & (boxes[:, 2]<=guide[2]) & (boxes[:, 3]<=guide[3])
            if visible_indices != set(group) or not np.all(full_visible[group]):
                excluded["neighboring_or_cut_cells"] = excluded.get("neighboring_or_cut_cells", 0)+1
                continue
            chosen_boxes = (boxes[group]-np.array(crop[:2]*2))*np.array([WIDTH/(crop[2]-crop[0]), HEIGHT/(crop[3]-crop[1])]*2)
            order = np.argsort(chosen_boxes[:, 0]+chosen_boxes[:, 2])
            selected_masks = masks[group][order]
            text = "".join(LETTER_BY_MASK[int(v)].upper() for v in selected_masks)
            selection = [(guide[0]-crop[0])/(crop[2]-crop[0]), (guide[1]-crop[1])/(crop[3]-crop[1]), (guide[2]-crop[0])/(crop[2]-crop[0]), (guide[3]-crop[1])/(crop[3]-crop[1])]
            filename = f"{len(records):06d}.png"
            patch.save(folder / filename, compress_level=1)
            records.append({"image": f"{split}-words/{filename}", "source": path.relative_to(root).as_posix(), "crop": crop, "guide": guide, "guideAspect": guide_aspect, "selection": selection, "annotation_rows": [int(i)+1 for i in group], "boxes": chosen_boxes[order].tolist(), "masks": selected_masks.tolist(), "context_boxes": transformed.tolist(), "context_masks": visible_masks.tolist(), "expected": text, "boundary_definition": "whole geometric word between gap>1.55*median cell pitch or annotated line edge"})
    save_json(output / f"{split}-words.json", {"scope": "public_annotation_space_bounded_words_not_lexical_or_camera_truth", "records": records, "excluded_by_ground_truth_geometry": excluded, "human_boundary_reviewed": False})
    print(f"{split}: {len(records)} space-bounded word fixtures, exclusions={excluded}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("C:/bm-datasets/AngelinaDataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-page", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260905)
    prepare(parser.parse_args())
