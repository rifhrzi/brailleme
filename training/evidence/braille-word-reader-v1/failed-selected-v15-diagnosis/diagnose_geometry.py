"""Saved v15 output decomposition and selected-box-only oracle; no inference."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import ast
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy.optimize import linear_sum_assignment
import psutil

OUT=Path(__file__).resolve().parent;WORK=OUT.parent;RUN=WORK/'selected-v15';DATA=WORK/'data'
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

decoder_path=RUN/'source/word_decode.py'
decoder_ast=ast.parse(decoder_path.read_text())
benchmark_ast=ast.parse((WORK.parent.parent/'training/evidence/braille-cell-classifier-v2/training-source/benchmark_angelina.py').read_text())
mapping=next(node for node in benchmark_ast.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LETTER_BY_MASK' for t in node.targets))
functions=[copy.deepcopy(node) for node in decoder_ast.body if isinstance(node,ast.FunctionDef) and node.name in ('decode','iou')]
decode_ast=next(node for node in functions if node.name=='decode')
result_index=next(i for i,node in enumerate(decode_ast.body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in node.targets))
assert ast.unparse(decode_ast.body[result_index-1]).startswith('selected.sort(')
# Execute exact trusted frozen decoder statements with NumPy globals only.
namespace={'np':np,'WIDTH':480,'HEIGHT':192,'GRID_W':120,'GRID_H':48,'STRIDE':4}
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
module=ast.fix_missing_locations(ast.Module(body=[future,mapping,*functions],type_ignores=[]))
exec(compile(module,str(decoder_path),'exec'),namespace)
extract=copy.deepcopy(decode_ast);extract.name='extract_selected';extract.body=extract.body[:result_index]
extract.body+=ast.parse('return selected, sizes, candidates').body
post=ast.FunctionDef(name='decode_selected',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in ('selected','sizes','t','selection')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(decode_ast.body[result_index-1:]),decorator_list=[])
module=ast.fix_missing_locations(ast.Module(body=[future,extract,post],type_ignores=[]))
exec(compile(module,str(decoder_path)+'::selected-stage','exec'),namespace)
decode=namespace['decode'];extract_selected=namespace['extract_selected'];decode_selected=namespace['decode_selected']

benchmark=read(RUN/'public-benchmark.json');fixtures=read(DATA/'external-words.json')['records'];t=benchmark['thresholds']
assert len(fixtures)==len(benchmark['words'])==607
assert sha(DATA/'external-words.json')==benchmark['fixture_manifest_sha256']
assert sha(RUN/'word_reader.tflite')==benchmark['model_sha256']
with np.load(RUN/'public-word-outputs.npz') as archive:qualities=archive['quality_reasons'].tolist()
assert len(qualities)==607
protocol={
 'scope':'Counterfactual selected-box geometry only, not model accuracy/calibration or an architecture forecast. No training, model loading/inference, threshold changes, dictionary or fixture changes.',
 'matching':'Greedy nearest unused final selected center <=.45 GT cell height in frozen GT left-to-right order. Also reverse greedy and maximum-cardinality minimum-distance assignment at the same cutoff; no cutoff tuning.',
 'counterfactual':'Preserve the exact post-selection/post-NMS proposal identities, count, scores and six probabilities. Replace only each matched proposal XYXY/size by its GT annotation box; retain every unmatched extra box. Sort/evaluate with exact frozen decoder post-NMS statements, thresholds and image-quality rejection. Never add missed cells, drop extras, rerun NMS, change confidence, or shift labels.',
 'strict_ceiling':'Count only full GT bijections, all correct/confident masks, no extras, with exact accepted text after GT box replacement. Also report literal text results, any lost baseline success, and the optimistic union preserving already-correct baseline results.',
 'limits':['This conditional selected-set oracle may move a center beyond its original grid cell; real11-logit outputs may not represent that correction without moving the objectness peak.', 'Geometry could alter earlier selection/NMS in a real model, but this fixed-proposal experiment deliberately cannot recover those suppressed/missed proposals.', 'Nearest-center matching is approximate; greedy/optimal disagreements and boundary geometry failures are disclosed.', 'GT geometric word boundaries are not human-reviewed lexical truth; all607 are development regression, not a fresh holdout.'],
 'hashes':{str(p.relative_to(WORK)):sha(p) for p in [RUN/'public-benchmark.json',RUN/'public-word-outputs.npz',RUN/'word_reader.tflite',decoder_path,DATA/'external-words.json']},'script_sha256':sha(__file__)}
assert not (OUT/'protocol.json').exists() and not (OUT/'report.json').exists()
save(OUT/'protocol.json',protocol)

def greedy(distance,reverse=False):
    used=set();matches={}
    for gi in (range(len(distance)-1,-1,-1) if reverse else range(len(distance))):
        choices=[(distance[gi,pi],pi) for pi in range(distance.shape[1]) if pi not in used]
        if choices:
            error,pi=min(choices)
            if error<=.45:matches[gi]=pi;used.add(pi)
    return matches

def optimal(distance):
    n,m=distance.shape
    if not m:return {}
    # Every unmatched real vertex costs10; valid pairs <=.45, invalid cost1000.
    # Cardinality dominates distance, with dummy-to-dummy free.
    costs=np.zeros((n+m,n+m));costs[:n,:m]=np.where(distance<=.45,distance,1000.)
    costs[:n,m:]=10.;costs[n:,:m]=10.
    rows,cols=linear_sum_assignment(costs)
    return {int(gi):int(pi) for gi,pi in zip(rows,cols) if gi<n and pi<m and distance[gi,pi]<=.45}

def counterfactual(selected,sizes,matches,gt,selection,quality):
    if quality:return {'text':'','reason':quality,'cells':[]}
    revised=copy.deepcopy(selected);new_sizes={id(cell):sizes[id(old)] for cell,old in zip(revised,selected)}
    for gi,pi in matches.items():
        box=gt[gi];x,y=box[:2]/[480,192];w,h=(box[2:]-box[:2])/[480,192]
        revised[pi]['box']=[float(x),float(y),float(x+w),float(y+h)]
        new_sizes[id(revised[pi])]=(float(w),float(h))
    result=decode_selected(revised,new_sizes,t,selection)
    assert len(result['cells'])==len(selected)
    assert sorted((c['score'],tuple(c['dots'])) for c in revised)==sorted((c['score'],tuple(c['dots'])) for c in selected)
    return result

def local_peaks(output):
    score=output[...,0];padded=np.pad(score,1,constant_values=-1);keep=np.ones(score.shape,bool)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx==dy==0:continue
            neighbor=padded[1+dy:49+dy,1+dx:121+dx]
            keep &= score>neighbor if dy<0 or dy==0 and dx<0 else score>=neighbor
    ys,xs=np.nonzero(keep);rows=output[ys,xs].astype(np.float64)
    centers=np.stack(((xs+rows[:,1])*4,(ys+rows[:,2])*4),axis=1)
    return rows,centers

def streamed_outputs():
    with zipfile.ZipFile(RUN/'public-word-outputs.npz') as archive:
        with archive.open('outputs.npy') as stream:
            version=np.lib.format.read_magic(stream)
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=reader(stream)
            assert shape==(607,48,120,11) and not fortran and dtype==np.dtype('float32')
            count=48*120*11*4
            for _ in range(607):
                data=stream.read(count);assert len(data)==count
                yield np.frombuffer(data,dtype=dtype).reshape(48,120,11)
            assert stream.read(1)==b''

totals=Counter();by_source=defaultdict(Counter);by_length=defaultdict(Counter);rows=[];disagreements=[]
process=psutil.Process();peak_rss=process.memory_info().rss
geometric_reasons={'clipped','too_far','multiple_rows','gap','overlapping_cells'}
for ordinal,(output,fixture,record,quality) in enumerate(zip(streamed_outputs(),fixtures,benchmark['words'],qualities)):
    for key,value in fixture.items():assert record[key]==value,(ordinal,key)
    gt=np.asarray(record['boxes'],float);centers=(gt[:,:2]+gt[:,2:])/2;heights=gt[:,3]-gt[:,1]
    selection=list(record['selection']);selection=[selection[0],selection[1],selection[0]+(selection[2]-selection[0]),selection[1]+(selection[3]-selection[1])]
    expected=record['expected'];actual=decode(output,t,record['selection']) if not quality else {'text':'','reason':quality,'cells':[]}
    assert actual=={'text':record['predicted'],'reason':record['rejection'],'cells':record['detected_cells']},ordinal
    if quality:selected=[];sizes={};candidates=[]
    else:selected,sizes,candidates=extract_selected(output,t,record['selection'])
    # Extracted original selected-stage execution must exactly reproduce the
    # archived full decoder including floating point size arithmetic.
    replay=decode_selected(selected,sizes,t,selection) if not quality else actual
    assert replay==actual,ordinal
    pred_boxes=np.asarray([p['box'] for p in selected]).reshape(-1,4)*[480,192,480,192]
    pred_centers=(pred_boxes[:,:2]+pred_boxes[:,2:])/2
    distances=np.linalg.norm(centers[:,None,:]-pred_centers[None,:,:],axis=2)/heights[:,None]
    match=greedy(distances);reverse=greedy(distances,True);best=optimal(distances)
    matched=[]
    for gi,pi in sorted(match.items()):
        cell=selected[pi];mask=sum(int(v>=.5)<<bit for bit,v in enumerate(cell['dots']))
        confidence=min(max(v,1-v) for v in cell['dots'])
        matched.append({'gt_index':gi,'proposal_index':pi,'gt_mask':record['masks'][gi],'predicted_mask':mask,
                        'correct_mask':mask==record['masks'][gi],'confident':confidence>=t['dotConfidence'] and cell['score']>=t['acceptScore'],
                        'dot_confidence':confidence,'score':cell['score'],'center_error_per_gt_height':float(distances[gi,pi])})
    missing=[];peak_values,peak_centers=local_peaks(output)
    for gi in range(len(gt)):
        if gi in match:continue
        nearby=np.where(np.linalg.norm(peak_centers-centers[gi],axis=1)/heights[gi]<=.45)[0]
        peak=int(nearby[np.argmax(peak_values[nearby,0])]) if len(nearby) else None
        score=float(peak_values[peak,0]) if peak is not None else 0.
        kind='quality_gate' if quality else 'below_proposal_threshold' if score<t['proposalScore'] else 'above_proposal_not_selected_or_matched'
        missing.append({'gt_index':gi,'annotation_row':record['annotation_rows'][gi],'gt_mask':record['masks'][gi],
                        'best_near_local_peak_score':score,'kind':kind})
    extras=[];context=np.asarray(record['context_boxes']);cc=(context[:,:2]+context[:,2:])/2;ch=context[:,3]-context[:,1]
    for pi in range(len(selected)):
        if pi in match.values():continue
        normalized=np.linalg.norm(cc-pred_centers[pi],axis=1)/ch;ci=int(np.argmin(normalized))
        target_context=bool(np.any(np.linalg.norm(centers-cc[ci],axis=1)<1e-4))
        if normalized[ci]<=.45 and not target_context:
            kind='partial_annotated_context' if np.any(context[ci,:2]<0) or np.any(context[ci,2:]>[480,192]) else 'complete_annotated_context'
        elif distances[:,pi].min()<=.45:kind='duplicate_near_target'
        else:kind='unmatched_texture_or_unannotated_cell'
        extras.append({'proposal_index':pi,'kind':kind,'nearest_context_index':ci,'nearest_context_mask':record['context_masks'][ci]})
    perfect_set=not quality and not missing and not extras and all(m['correct_mask'] and m['confident'] for m in matched)
    cf=counterfactual(selected,sizes,match,gt,selection,quality)
    strict_exact=perfect_set and cf['text']==expected
    alternatives={}
    for name,alternative in [('reverse_greedy',reverse),('optimal',best)]:
        alt=counterfactual(selected,sizes,alternative,gt,selection,quality)
        alt_good=not quality and len(alternative)==len(gt)==len(selected)
        for gi,pi in alternative.items():
            cell=selected[pi];mask=sum(int(v>=.5)<<bit for bit,v in enumerate(cell['dots']))
            alt_good &= mask==record['masks'][gi] and min(max(v,1-v) for v in cell['dots'])>=t['dotConfidence'] and cell['score']>=t['acceptScore']
        alternatives[name]={'matched_count':len(alternative),'assignment_differs':alternative!=match,'strict_exact':bool(alt_good and alt['text']==expected),'text':alt['text'],'reason':alt['reason']}
    if reverse!=match or best!=match:disagreements.append({'image':record['image'],'greedy':match,'reverse':reverse,'optimal':best,'alternatives':alternatives})
    row={'ordinal':ordinal,'image':record['image'],'source':record['source'],'expected':expected,'actual_text':actual['text'],'actual_reason':actual['reason'],'quality':quality,
         'matched':matched,'missing':missing,'extras':extras,'perfect_selected_set':bool(perfect_set),
         'rejected_only_geometric_gate':bool(perfect_set and actual['reason'] in geometric_reasons),
         'box_only_counterfactual':{'text':cf['text'],'reason':cf['reason'],'strict_complete_exact':bool(strict_exact)},'alternative_matching':alternatives}
    rows.append(row)
    count=Counter(words=1,gt_cells=len(gt),selected_cells=len(selected),matched_cells=len(match),missing_cells=len(missing),extra_cells=len(extras),wrong_matched_masks=sum(not m['correct_mask'] for m in matched),uncertain_matched_cells=sum(not m['confident'] for m in matched),correct=actual['text']==expected,accepted=bool(actual['text']),rejected=not actual['text'],quality_rejected=bool(quality),perfect_selected_set=perfect_set,
                  rejected_only_geometry=row['rejected_only_geometric_gate'],geometry_counterfactual_text_exact=cf['text']==expected,geometry_counterfactual_accepted=bool(cf['text']),geometry_counterfactual_strict_exact=strict_exact,
                  newly_strict_exact_from_rejection=strict_exact and not actual['text'],newly_strict_exact_from_wrong_acceptance=strict_exact and bool(actual['text']) and actual['text']!=expected,
                  baseline_correct_lost_by_literal_box_replacement=actual['text']==expected and cf['text']!=expected,
                  baseline_correct_without_full_bijection=actual['text']==expected and not perfect_set,
                  cf_text_exact_without_complete_correspondence=cf['text']==expected and not strict_exact,
                  complete_confident_set_still_rejected_after_gt_boxes=perfect_set and not cf['text'],
                  matching_reverse_disagreement=reverse!=match,matching_optimal_disagreement=best!=match,
                  optimal_extra_matched_cells=len(best)-len(match),optimal_counterfactual_strict_exact=alternatives['optimal']['strict_exact'],reverse_counterfactual_strict_exact=alternatives['reverse_greedy']['strict_exact'])
    if actual['reason']:count['actual_rejected_'+actual['reason']]+=1
    if row['rejected_only_geometric_gate']:count['geometry_only_'+actual['reason']]+=1
    if perfect_set and not cf['text']:count['perfect_gt_boxes_rejected_'+cf['reason']]+=1
    for m in missing:count['missing_'+m['kind']]+=1
    for extra in extras:count['extra_'+extra['kind']]+=1
    totals.update(count);by_source[record['source']].update(count);by_length[str(len(expected))].update(count)
    peak_rss=max(peak_rss,process.memory_info().rss);assert peak_rss<1_000_000_000

assert totals['correct']==benchmark['whole_words']['correct'] and totals['accepted']==benchmark['whole_words']['accepted']
assert totals['gt_cells']==2242 and totals['matched_cells']+totals['missing_cells']==2242
assert sum(totals['actual_rejected_'+reason] for reason in benchmark['whole_words']['rejection_reasons'])==totals['rejected']
summary=dict(totals)
summary['literal_box_counterfactual_exact_fraction']=totals['geometry_counterfactual_text_exact']/607
summary['strict_full_correspondence_box_counterfactual_fraction']=totals['geometry_counterfactual_strict_exact']/607
summary['optimistic_existing_success_union_new_strict_correct_count']=totals['correct']+totals['newly_strict_exact_from_rejection']+totals['newly_strict_exact_from_wrong_acceptance']
summary['optimistic_existing_success_union_new_strict_correct_fraction']=summary['optimistic_existing_success_union_new_strict_correct_count']/607
summary['literal_box_counterfactual_accepted_accuracy']=totals['geometry_counterfactual_text_exact']/totals['geometry_counterfactual_accepted'] if totals['geometry_counterfactual_accepted'] else None
for path_name,expected_hash in protocol['hashes'].items():assert sha(WORK/path_name)==expected_hash
result={'scope':protocol['scope'],'protocol_sha256':sha(OUT/'protocol.json'),'matching':protocol['matching'],'counterfactual':protocol['counterfactual'],
        'actual_frozen_word_metrics':benchmark['whole_words'],'totals':summary,'by_length':dict(by_length),'by_source':dict(by_source),
        'matching_disagreements':disagreements,'rows':rows,'limitations':protocol['limits'],
        'runtime':{'model_inference':False,'tensor_streaming':'one253440-byte NPY row at a time','sampled_peak_rss_bytes':peak_rss,'windows_peak_working_set_bytes':getattr(process.memory_info(),'peak_wset',None),'ml_modules_imported':[name for name in ('tensorflow','torch','keras') if name in sys.modules]}}
save(OUT/'report.json',result)
print(json.dumps({'totals':summary,'runtime':result['runtime'],'report_sha256':sha(OUT/'report.json')},indent=2))
