"""Fixed two-inference pending-context experiment; not actual camera evidence."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='1'
os.environ['TF_NUM_INTEROP_THREADS']='1'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import tensorflow as tf

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
directory=OUT/'approximate-pending-2026-v1'
directory.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,row):
 with p.open('x',encoding='utf8')as stream:json.dump(row,stream,indent=2);stream.write('\n')
manifest_path=Path('C:/bm-symbols-apk-20260908/evidence/calibrated-feeds-v1/manifest.json')
manifest=json.loads(manifest_path.read_text());row=next(r for r in manifest['rows']if r.get('expected_text')=='2026')
feed_path=manifest_path.parent/row['output'];assert sha(feed_path)==row['output_sha256']
mapping_path=Path(manifest['mapping_report']);assert sha(mapping_path)==manifest['mapping_report_sha256']
mapping=json.loads(mapping_path.read_text());A=np.array(mapping['feed_to_screenshot']['matrix']);t=np.array(mapping['feed_to_screenshot']['translation'])
feed=np.asarray(Image.open(feed_path).convert('RGB'))
yy,xx=np.mgrid[:640,:480];screen=np.stack([53+xx.ravel()*975/480,548+(yy.ravel()-80)*975/480]);coordinates=np.linalg.solve(A,screen-t[:,None])
frame=np.clip(np.round(np.stack([map_coordinates(feed[:,:,c].astype(float),[coordinates[1],coordinates[0]],order=1,mode='nearest').reshape(640,480)for c in range(3)],axis=2)),0,255).astype(np.uint8)
model_manifest_path=ROOT/'assets/models/braille-word-reader-v1/manifest.json';model_manifest=json.loads(model_manifest_path.read_text());model_path=model_manifest_path.parent/'word-reader.tflite';assert sha(model_path)==model_manifest['artifact']['sha256']
model=tf.lite.Interpreter(model_path=str(model_path),num_threads=1);model.allocate_tensors();inp=model.get_input_details()[0]['index'];out=model.get_output_details()[0]['index'];inferences=0
def run_pass(number,context):
 global inferences
 x,y=context['x']*480,context['y']*640;right=(context['x']+context['width'])*480;bottom=(context['y']+context['height'])*640
 x,y=max(0,int(np.floor(x))),max(0,int(np.floor(y)));right,bottom=min(480,int(np.ceil(right))),min(640,int(np.ceil(bottom)))
 crop=frame[y:bottom,x:right];resized=np.asarray(Image.fromarray(crop).resize((480,192),Image.Resampling.BILINEAR))
 tensor=resized.astype(np.float32)[None]/255.;model.set_tensor(inp,tensor);model.invoke();inferences+=1;values=model.get_tensor(out)[0]
 rgb_path=directory/f'pass-{number}.rgb';output_path=directory/f'pass-{number}.f32le'
 with rgb_path.open('xb')as f:f.write(resized.tobytes())
 with output_path.open('xb')as f:f.write(values.astype('<f4').tobytes())
 result={'requested_context':context,'actual_crop_xyxy':[x,y,right,bottom],'actual_context':{'x':x/480,'y':y/640,'width':(right-x)/480,'height':(bottom-y)/640},'input_rgb_file':str(rgb_path),'input_rgb_sha256':sha(rgb_path),'output_file':str(output_path),'output_sha256':sha(output_path)}
 save(directory/f'pass-{number}.json',result)
run_pass(1,{'x':0,'y':.35,'width':1,'height':.3})
bridge=OUT/'pending-context-bridge.cjs'
first=json.loads(subprocess.check_output(['node',str(bridge),str(directory),'first'],cwd=ROOT,text=True))
save(directory/'first-group-and-context.json',first)
if first['next_frame_context']:
 run_pass(2,first['next_frame_context'])
 comparison=json.loads(subprocess.check_output(['node',str(bridge),str(directory),'second'],cwd=ROOT,text=True))
else:comparison=first
save(directory/'comparison.json',comparison)
report={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Fixed approximate2026 global-center versus pending-context experiment; no actual frame pixels, no tuning.',
 'inferences':inferences,'maximum_inferences':2,'expected_text_for_identity_only':'2026','feed_manifest':str(manifest_path),'feed_manifest_sha256':sha(manifest_path),'feed_sha256':sha(feed_path),'mapping_sha256':sha(mapping_path),'model_sha256':sha(model_path),'model_manifest_sha256':sha(model_manifest_path),'thresholds':model_manifest['thresholds'],
 'approximation':'Frozen source affine mapped into assumed upright480x640/frame640x480right using square previewcrop80px; fixed bilinear samples and uint8 rounding; no fitted photometric adjustment for numberfeed.',
 'algorithm':'First actual TS group nearest center becomes pending; getWordTrackingContext(pending) computes next context; exactly one focused inference; same-global cached output is the control without additional inference.',
 'first_group_count':len(comparison['first']['groups']),'first_pending_cells':len(comparison['first']['state'].get('pending',{}).get('boxes',[])),
 'second_group_count':len(comparison.get('second',{}).get('groups',[])),
 'same_global_control_ready':comparison['same_global_cached_repeat_control']['ready'],'pending_context_ready':comparison.get('second',{}).get('ready'),
 'identity_comparisons':comparison.get('second',{}).get('comparisons_to_pending',[]),
 'input_scale_first':comparison['model_pixels_per_frame_pixel_first_x'],'input_scale_pending':comparison['model_pixels_per_frame_pixel_next_x'],
 'script_sha256':sha(Path(__file__)),'bridge_sha256':sha(bridge),'comparison_sha256':sha(directory/'comparison.json'),
 'limits':['Native ImageAnalysis pixels and actual adaptive observation sequence are unavailable.','Reusing one global output only tests determinism for identical pixels, not temporal camera stability.','Failure/success in this fixed approximation does not prove or exclude the live2026 cause.','No model/decoder thresholds, checkpoint, runtime app, or deployment assets changed.']}
save(directory/'report.json',report);print(json.dumps(report,indent=2))
