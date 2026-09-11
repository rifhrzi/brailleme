const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {load,root,loadedSourcePaths}=require('../../scripts/word-native-parity/geometry-reference.cjs');
const decoder=load(path.join(root,'src/ml/word-decoder.ts'));
const tracking=load(path.join(root,'src/ml/word-tracking.ts'));
const geometry=load(path.join(root,'src/ml/word-camera-geometry.ts'));
const directory=path.resolve(process.argv[2]),phase=process.argv[3];
const manifest=JSON.parse(fs.readFileSync(path.join(root,'assets/models/braille-word-reader-v1/manifest.json')));
function read(number){
 const row=JSON.parse(fs.readFileSync(path.join(directory,`pass-${number}.json`)));
 const raw=fs.readFileSync(row.output_file);const values=new Float32Array(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
 const c=row.actual_context;
 const proposals=decoder.detectWordProposals(values,manifest.thresholds).map(p=>({...p,box:{x:c.x+p.box.x*c.width,y:c.y+p.box.y*c.height,width:p.box.width*c.width,height:p.box.height*c.height}}));
 const groups=tracking.groupWordProposals(proposals,480,640,manifest.thresholds,c);
 const frame={width:640,height:480,orientation:'right',isMirrored:false,
  convertCameraPointToFramePoint:({x,y})=>({x:x/2,y:y/2}),convertFramePointToCameraPoint:({x,y})=>({x:x*2,y:y*2})};
 const camera={convertCameraPointToViewPoint:({x,y})=>({x:(960-y)/2,y:x/2-80}),convertViewPointToCameraPoint:({x,y})=>({x:2*(y+80),y:960-2*x})};
 const projected=groups.map(group=>({...group,
   box:geometry.mapWordSensorBoxToPreview(geometry.snapshotWordFrameBoxSensorPoints(frame,group.box),camera,480,480),
   boxes:group.boxes.map(b=>geometry.mapWordSensorBoxToPreview(geometry.snapshotWordFrameBoxSensorPoints(frame,b),camera,480,480))}));
 return {groups:projected,proposal_count:proposals.length,actual_context:c,frame,camera};
}
const first=read(1);
const initial=tracking.advanceWordTracking(tracking.INITIAL_WORD_TRACKING_STATE,first.groups,0);
const control=tracking.advanceWordTracking(initial,first.groups,200);
let nextContext=null,previewContext=null;
if(initial.pending){
 previewContext=tracking.getWordTrackingContext(initial.pending,480,480);
 nextContext=geometry.mapWordSensorBoxToFrame(geometry.snapshotWordGuideSensorPoints(first.camera,previewContext,480,480),first.frame);
}
const report={scope:'Actual TS decoder/group/context/state functions on fixed cached approximate inputs; native point transforms are explicit640x480 right-rotation mocks.',
 first:{proposal_count:first.proposal_count,groups:first.groups,state:initial},same_global_cached_repeat_control:{state:control,ready:tracking.getWordTrackingTarget(control,200)?.ready??false,limitation:'Reuses identical cached global output, not an extra independent camera observation/inference.'},
 pending_preview_context:previewContext,next_frame_context:nextContext,
 model_pixels_per_frame_pixel_first_x:1,model_pixels_per_frame_pixel_next_x:nextContext?480/Math.ceil(nextContext.width*480):null};
if(phase==='second'){
 const second=read(2),advanced=tracking.advanceWordTracking(initial,second.groups,200);
 const median=values=>{const s=[...values].sort((a,b)=>a-b),m=Math.floor(s.length/2);return s.length%2?s[m]:(s[m-1]+s[m])/2;};
 report.second={proposal_count:second.proposal_count,groups:second.groups,state:advanced,ready:tracking.getWordTrackingTarget(advanced,200)?.ready??false,
  comparisons_to_pending:initial.pending?second.groups.map(g=>{const h=median([...initial.pending.boxes,...g.boxes].map(b=>b.height));const d=Math.abs(initial.pending.box.y+initial.pending.box.height/2-g.box.y-g.box.height/2);const iou=decoder.wordBoxIou(initial.pending.box,g.box);return {iou,row_distance:d,row_distance_limit:h*.55,passes_identity:iou>=.3&&d<=h*.55,cell_count:g.boxes.length};}):[]};
}
report.source_sha256=Object.fromEntries([...loadedSourcePaths(),__filename].map(p=>[path.relative(root,p).replaceAll('\\','/'),crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')]));
process.stdout.write(JSON.stringify(report));
