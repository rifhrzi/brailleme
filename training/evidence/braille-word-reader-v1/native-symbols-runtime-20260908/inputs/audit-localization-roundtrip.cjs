const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { load, root, loadedSourcePaths } = require('../../scripts/word-native-parity/geometry-reference.cjs');
const image = load(path.join(root, 'src/ml/word-image-processing.ts'));
const geometry = load(path.join(root, 'src/ml/word-camera-geometry.ts'));
const contract = load(path.join(root, 'src/ml/word-model-contract.ts'));
const tracking = load(path.join(root, 'src/ml/word-tracking.ts'));
const thresholds = JSON.parse(fs.readFileSync(path.join(root, 'assets/models/braille-word-reader-v1/manifest.json'))).thresholds;
const rows = [];
for (const orientation of ['up', 'right', 'down', 'left']) for (const mirrored of [false, true]) {
  const quarter = orientation === 'right' || orientation === 'left';
  const width = quarter ? 480 : 640, height = quarter ? 640 : 480;
  const frame = { width:640, height:480, orientation, isMirrored:mirrored,
    convertFramePointToCameraPoint: ({x,y}) => ({x:2*x+17,y:2*y-31}),
    convertCameraPointToFramePoint: ({x,y}) => ({x:(x-17)/2,y:(y+31)/2}) };
  let actual;
  const source = { width, height,
    crop(x,y,right,bottom) { actual={x:x/width,y:y/height,width:(right-x)/width,height:(bottom-y)/height};
      return { dispose(){}, toRawPixelData:()=>({width:right-x,height:bottom-y,pixelFormat:'RGB',buffer:new ArrayBuffer((right-x)*(bottom-y)*3)}) }; } };
  const context = {x:.101,y:.301,width:.721,height:.321};
  const relative = {x:.35,y:.36,width:.09,height:.25};
  const values = new Float32Array(contract.WORD_MODEL_OUTPUT_LENGTH);
  const cx=relative.x+relative.width/2,cy=relative.y+relative.height/2,col=Math.floor(cx*120),row=Math.floor(cy*48);
  values.set([.99,cx*120-col,cy*48-row,relative.width,relative.height,.99,.01,.01,.01,.01,.01],(row*120+col)*11);
  const model = {runSync:()=>[values.buffer]};
  const processor={resizePixels:()=>({tensor:new Float32Array(contract.WORD_MODEL_INPUT_LENGTH).fill(.5).buffer,meanLuminance:.5,contrast:.1})};
  const result=image.detectWordImageCells(source,model,thresholds,context,processor);
  if(result.status!=='success'||result.groups.length!==1)throw Error('Synthetic actual-helper detection failed');
  const box=result.groups[0].box;
  const expected={x:actual.x+relative.x*actual.width,y:actual.y+relative.y*actual.height,width:relative.width*actual.width,height:relative.height*actual.height};
  const points=geometry.snapshotWordFrameBoxSensorPoints(frame,box);
  const returned=geometry.mapWordSensorBoxToFrame(points,frame);
  const maxError=Math.max(...Object.keys(box).map(k=>Math.abs(box[k]-expected[k])*((k==='x'||k==='width')?width:height)));
  const roundtripError=Math.max(...Object.keys(box).map(k=>Math.abs(box[k]-returned[k])*((k==='x'||k==='width')?width:height)));
  if(maxError>1e-4||roundtripError>1e-8)throw Error('Crop or frame origin lost');
  rows.push({orientation,mirrored,context,actualPixelRoundedContext:actual,expectedGlobalBox:expected,actualGlobalBox:box,cropMappingMaxErrorPixels:maxError,sensorRoundtripMaxErrorPixels:roundtripError});
}
const report={scope:'Actual TS functions with injected synthetic output and mocked native pixel/affine methods; no model inference/native runtime proof.',
  status:'no_crop_origin_or_rotation_roundtrip_bug_reproduced',rows,
  centerWindowUprightFramePixels:{width:480,height:640,previewVerticalCrop:80,context:[0,224,480,416]},
  source_sha256:Object.fromEntries([...loadedSourcePaths(),__filename].map(p=>[path.relative(root,p).replaceAll('\\','/'),crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')]))};
const out=path.join(__dirname,'v6-localization-roundtrip.json');fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({file:out,cases:rows.length,maxCropError:Math.max(...rows.map(r=>r.cropMappingMaxErrorPixels)),maxRoundtripError:Math.max(...rows.map(r=>r.sensorRoundtripMaxErrorPixels))}));
