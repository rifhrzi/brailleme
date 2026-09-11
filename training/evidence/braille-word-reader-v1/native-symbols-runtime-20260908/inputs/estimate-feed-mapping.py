"""Offline pixel alignment only; does not read model outputs or operate the emulator."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
FEED = Path('C:/bm-symbols-apk-20260908/evidence/camera-word.png')
SCREEN = Path('C:/bm-symbols-apk-20260908/evidence/actions-prepare-v2/20260908T165746496942Z-prepare-02-e4898b.png')
SOURCE = ROOT/'assets/models/braille-word-reader-v1/native-word-fixture.png'
PLACEMENT = np.array([287., 383.])
PREVIEW = [53., 548., 975., 975.]
OUTPUT = Path(__file__).with_name('feed-mapping-v1.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(source, screenshot, bounds, blur=.45, stride=2, initial=None):
    left, top, right, bottom = bounds
    yy, xx = np.mgrid[top:bottom:stride, left:right:stride]
    observed = gaussian_filter(screenshot, blur*1.69)[yy, xx].ravel()
    smoothed = gaussian_filter(source, blur)
    points = np.stack([xx.ravel(), yy.ravel()])
    initial = np.array(initial if initial is not None else [1.69, 0., 0., 1.69, 538., 817., 1., -.02])

    def residual(parameters):
        a,b,c,d,tx,ty,gain,bias = parameters
        inverse = np.linalg.inv([[a,b],[c,d]])
        coordinates = inverse @ (points-np.array([[tx],[ty]]))
        expected = map_coordinates(smoothed, [coordinates[1], coordinates[0]], order=1, mode='nearest')
        return gain*expected+bias-observed

    result = least_squares(residual, initial, loss='soft_l1', f_scale=.008,
        bounds=([1.60,-.06,-.06,1.60,530.,809.,.5,-.2], [1.80,.06,.06,1.80,546.,825.,1.5,.2]),
        x_scale='jac', max_nfev=80, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    error = residual(result.x)
    predicted = observed+error
    return dict(parameters=result.x.tolist(), samples=len(observed), converged=bool(result.success),
        rmse_8bit=float(np.sqrt(np.mean(error**2))*255),
        correlation=float(np.corrcoef(predicted,observed)[0,1]),
        residual_p95_abs_8bit=float(np.quantile(np.abs(error),.95)*255),
        source_blur_sigma=blur, screenshot_roi=bounds, stride=stride)


def main():
    feed=np.array(Image.open(FEED).convert('RGB'))
    source=np.array(Image.open(SOURCE).convert('RGB'))
    screen=np.array(Image.open(SCREEN).convert('RGB'))
    if not np.array_equal(feed[383:640,287:737],source):
        raise ValueError('Feed placement no longer has the exact source pixels')
    weights=np.array([.2126,.7152,.0722])
    source_gray=source@weights/255.
    screen_gray=screen@weights/255.
    global_fit=fit(source_gray,screen_gray,[542,821,1023,1247])
    variants=[]
    for blur,bounds in [(.2,[542,821,1023,1247]),(.7,[542,821,1023,1247]),
                        (.45,[550,829,1015,1239]),(.45,[542,821,1023,1034]),
                        (.45,[542,1034,1023,1247]),(.45,[542,821,782,1247]),
                        (.45,[782,821,1023,1247])]:
        variants.append(fit(source_gray,screen_gray,bounds,blur=blur,initial=global_fit['parameters']))
    fits=[global_fit,*variants]
    matrices=[]
    translations=[]
    for result in fits:
        a,b,c,d,tx,ty,_,_=result['parameters']
        matrix=np.array([[a,b],[c,d]])
        matrices.append(matrix)
        translations.append(np.array([tx,ty])-matrix@PLACEMENT)
    matrix=matrices[0]
    translation=translations[0]
    x,y,w,h=PREVIEW
    corners=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    visible=(np.linalg.inv(matrix)@(corners-translation).T).T
    center=np.linalg.inv(matrix)@(np.array([x+w/2,y+h/2])-translation)
    all_corners=np.array([(np.linalg.inv(a)@(corners-t).T).T for a,t in zip(matrices,translations)])
    # Fixed native source size; no recognition result can affect this placement.
    proposed=[int(np.floor(center[0]-source.shape[1]/2)),int(np.floor(center[1]-source.shape[0]/2))]
    source_box=np.array([[proposed[0],proposed[1]],[proposed[0]+source.shape[1],proposed[1]],
                         [proposed[0]+source.shape[1],proposed[1]+source.shape[0]],[proposed[0],proposed[1]+source.shape[0]]])
    predicted_box=(matrix@source_box.T).T+translation
    report=dict(schemaVersion=1,createdAtUtc=datetime.now(timezone.utc).isoformat(),
        scope='offline_pixel_alignment_of_one_emulator_screenshot_only',
        inputs=dict(feed=dict(path=str(FEED),sha256=sha(FEED),size=[1024,1024]),
                    screenshot=dict(path=str(SCREEN),sha256=sha(SCREEN),size=[1080,2400]),
                    source=dict(path=str(SOURCE),sha256=sha(SOURCE),size=[450,257],
                                exact_feed_placement=[287,383,737,640],placement_pixels_identical=True)),
        preview_rectangle_xywh=PREVIEW,preview_rectangle_origin='Provided screenshot pixel bounds; not native camera transform metadata',
        method=dict(description='Robust grayscale pixel alignment with six affine geometry parameters plus brightness gain/bias; bilinear sampling and matched Gaussian smoothing tolerate YUV/brightness differences.',
                    pixel_coordinates='Integer pixel-center indices; rectangle extent interpretation has approximately one-pixel edge uncertainty.',
                    excludes='No model output, expected text, ADB, emulator state, or app source is used to fit geometry.'),
        fit=global_fit,sensitivity_fits=variants,
        feed_to_screenshot=dict(matrix=matrix.tolist(),translation=translation.tolist(),
                                equation='screenshot_xy = matrix @ feed_xy + translation'),
        feed_to_preview=dict(matrix=matrix.tolist(),translation=(translation-np.array([x,y])).tolist()),
        observed_visible_feed_polygon=visible.tolist(),estimated_visible_feed_center=center.tolist(),
        sensitivity=dict(note='Empirical variation across blur and spatial subsets, not a statistical confidence interval. Add at least one screenshot pixel for preview-edge uncertainty.',
                         matrix_min=np.min(matrices,axis=0).tolist(),matrix_max=np.max(matrices,axis=0).tolist(),
                         translation_min=np.min(translations,axis=0).tolist(),translation_max=np.max(translations,axis=0).tolist(),
                         visible_corner_min=np.min(all_corners,axis=0).tolist(),visible_corner_max=np.max(all_corners,axis=0).tolist()),
        proposed_calibration_composition=dict(canvas_size=[1024,1024],background_rgb=feed[0,0].tolist(),
            source=str(SOURCE),source_sha256=sha(SOURCE),source_size=[450,257],scale=1,
            placement=[*proposed,proposed[0]+450,proposed[1]+257],source_pixels_cropped=False,
            rule='Place the unchanged full source at floor(inverse-affine preview center minus half source size). Freeze this geometry before any prediction.',
            predicted_screenshot_corners=predicted_box.tolist(),new_file_created=False),
        limitations=['Only one screenshot was fitted; mapping can change with camera mode, layout, feed size, or orientation.',
                     'The right source region is clipped in this screenshot; its position is an affine extrapolation.',
                     'This measurement does not identify whether the app, emulator feed, native camera, or preview layer caused the mapping.',
                     'Full source visibility is a geometry calibration only and is not model or physical-camera acceptance.'],
        analysis_script_sha256=sha(Path(__file__)))
    with OUTPUT.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({key:report[key] for key in ['fit','feed_to_screenshot','observed_visible_feed_polygon','estimated_visible_feed_center','proposed_calibration_composition']},indent=2))


if __name__=='__main__':
    main()
