"""One fixed exploratory TFLite replay of an approximate frame; never native evidence."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_NUM_INTRAOP_THREADS']='1'
os.environ['TF_NUM_INTEROP_THREADS']='1'
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import tensorflow as tf

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'training/scripts'))
from word_decode import decode

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
review_path=OUT/'overlay-review-v6-tracking-masked-v2.json'
review=json.loads(review_path.read_text())
feed_path=Path(review['inputs']['feed']['path'])
assert sha(feed_path)==review['inputs']['feed']['sha256']
feed=np.asarray(Image.open(feed_path).convert('RGB'))/255.
fit=review['fit'];A=np.array(fit['matrix']);t=np.array(fit['translation'])
yy,xx=np.mgrid[:640,:480]
# Pixel centers of assumed upright480x640 frame projected through square cover.
screen=np.stack([53+xx.ravel()*(975/480),548+(yy.ravel()-80)*(975/480)])
coordinates=np.linalg.solve(A,screen-t[:,None])
frame=np.stack([map_coordinates(feed[:,:,channel],[coordinates[1],coordinates[0]],order=1,mode='nearest').reshape(640,480) for channel in range(3)],axis=2)
frame=np.clip(np.round((frame*fit['parameters'][6]+fit['parameters'][7])*255),0,255).astype(np.uint8)
crop=frame[224:416,:]
model_path=ROOT/'assets/models/braille-word-reader-v1/word-reader.tflite'
manifest=json.loads((model_path.parent/'manifest.json').read_text())
assert sha(model_path)==manifest['artifact']['sha256']
interpreter=tf.lite.Interpreter(model_path=str(model_path),num_threads=1)
interpreter.allocate_tensors();inputs=interpreter.get_input_details();outputs=interpreter.get_output_details()
tensor=crop.astype(np.float32)[None]/255.
interpreter.set_tensor(inputs[0]['index'],tensor);interpreter.invoke();pred=interpreter.get_tensor(outputs[0]['index'])[0]
result=decode(pred,manifest['thresholds'])
rows=[]
for cell in result['cells']:
 b=cell['box'];frame_box=np.array([b[0]*480,224+b[1]*192,b[2]*480,224+b[3]*192])
 screenshot_box=np.array([53+frame_box[0]*(975/480),548+(frame_box[1]-80)*(975/480),53+frame_box[2]*(975/480),548+(frame_box[3]-80)*(975/480)])
 corners=np.array([[screenshot_box[0],screenshot_box[1]],[screenshot_box[2],screenshot_box[3]]])
 source=(corners-t)@np.linalg.inv(A).T-[62,383]
 rows.append({'score':cell['score'],'model_box_xyxy':b,'upright_frame_box_xyxy':frame_box.tolist(),'approx_source_box_xyxy':source.ravel().tolist()})
target=OUT/'approximate-v6-center-window-output.f32le'
with target.open('xb')as f:f.write(pred.astype('<f4').tobytes())
input_path=OUT/'approximate-v6-center-window.rgb'
with input_path.open('xb')as f:f.write(crop.tobytes())
report={'scope':'One approximate offline camera frame, not actual captured frame/proposal evidence. No parameter tuning or additional training.',
 'model_sha256':sha(model_path),'decoder_version':manifest['decoderVersion'],'thresholds':manifest['thresholds'],
 'source_review':str(review_path),'source_review_sha256':sha(review_path),'feed_sha256':sha(feed_path),
 'assumed_frame':{'raw_size':[640,480],'orientation':'right','upright_size':[480,640],'square_preview_crop_y':80,'model_context_xyxy':[0,224,480,416]},
 'sampling':'Inverse measured feed-to-screenshot affine, bilinear sampling, fixed observed gain/bias, uint8 rounding; no cameraYUV/compression/noise simulation.',
 'input':{'path':str(input_path),'sha256':sha(input_path),'shape':[192,480,3]},
 'output':{'path':str(target),'sha256':sha(target),'shape':[48,120,11]},
 'raw_decoder':{'text':result['text'],'reason':result['reason']},'proposals':rows,
 'script_sha256':sha(Path(__file__)),
 'limits':['Native frame pixels and raw proposal coordinates were not recorded.','A fixed approximate resampling can change model predictions; disagreement is not a native mapping bug.','Whole-scene startup context contains several rows; no full-scene expected text/cell labels.','Single initial center window only; no adaptive tracking or repeated parameter search.']}
p=OUT/'approximate-v6-center-window-replay.json'
with p.open('x',encoding='utf8')as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'report':str(p),'proposals':len(rows),'raw_decoder':report['raw_decoder'],'source_boxes':[r['approx_source_box_xyxy'] for r in rows]}))
