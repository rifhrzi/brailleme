"""Bounded independent scalar and saved-proof audit; no model/backend imports."""
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

OUT=Path(__file__).resolve().parent
WORK=OUT.parent
V1=WORK/'anchor-preserving-zoom-prototype-v1'
V2=WORK/'anchor-preserving-zoom-prototype-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def load_helper(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result.anchor_preserving_zoom


def bounds(box,tx,ty):
    # Independent exact-rational inequalities: 0 <= c+t+z*(q-c) <= size.
    result=[]
    for q,c,t,size in zip(box,[240,96,240,96],[tx,ty,tx,ty],[480,192,480,192]):
        d=Fraction(float(q))-c
        if d<0:result.append((Fraction(c)+Fraction(t))/-d)
        elif d>0:result.append((Fraction(size-c)-Fraction(t))/d)
    return min(result)


def apply(box,z,tx,ty):
    return np.asarray(box,np.float64)*z+np.array([240*(1-z)+tx,96*(1-z)+ty]*2)


def inside(box):
    return box[0]>=0 and box[1]>=0 and box[2]<=480 and box[3]<=192


def main():
    if (OUT/'report.json').exists():raise RuntimeError('Append-only review already exists')
    spec=read(V2/'prototype-spec.json');proof=read(V2/'proof-report.json')
    assert proof['status']=='passed' and proof['prototype_spec_sha256']==sha(V2/'prototype-spec.json')
    assert proof['script_sha256']==sha(V2/'proof.py')
    for folder,key in [('source-original','source_original_sha256'),('source-private','source_private_sha256')]:
        for name,digest in spec[key].items():assert sha(V2/folder/name)==digest
    assert sha(V2/'source-private/word_data.py')==sha(V1/'source-private/word_data.py')
    assert sha(V2/'source-private/word_train.py')==sha(V1/'source-private/word_train.py')
    assert sha(V2/'word_anchor_zoom.py')==sha(V2/'source-private/word_anchor_zoom.py')
    old=load_helper(V1/'word_anchor_zoom.py','anchor_v1')
    new=load_helper(V2/'word_anchor_zoom.py','anchor_v2')
    box=np.asarray([50,70,280,125],np.float32);z=1.2847156687768433;tx=4.0959770676002165
    old_z=old([box],[True],z,tx,0.);new_z=new([box],[True],z,tx,0.)
    assert old_z==z and apply(box,old_z,tx,0)[0]<0
    assert new_z<z and inside(apply(box,new_z,tx,0))
    inset_ulp=round((z-new_z)/np.spacing(z));assert 1<=inset_ulp<=8
    checks=0
    for box in ([0,0,480,192],[0,0,60,50],[420,142,480,192],[15,70,90,130],[400,15,470,70],[210,80,270,112]):
        for tx in (-19.2,0.,19.2):
            for ty in (-7.68,0.,7.68):
                limit=float(bounds(box,tx,ty))
                assert limit>=.92-1e-15  # .8 fallback is unreachable in the valid full-box/+/-4% domain.
                for requested in (.8,1.15,1.7):
                    actual=new([box],[True],requested,tx,ty)
                    assert .8<=actual<=requested and inside(apply(box,actual,tx,ty))
                    expected=min(requested,limit)
                    assert abs(actual-expected)<=12*abs(np.spacing(expected))
                    checks+=1
    assert new(np.empty((0,4)),[],1.7,0,0)==1.7
    assert new([[0,0,480,192]],[False],1.7,0,0)==1.7
    assert new([[-1,20,30,80]],[True],1.7,0,0)==1.7
    near=np.asarray([[210,30,230,170],[250,70,270,130]],float)
    assert new(near,[True,True],1.7,0,0)<new(near[::-1],[True,True],1.7,0,0)
    # Actual frozen Pillow affine consumes the capped scalar; no second raster.
    tree=ast.parse((V2/'source-private/word_data.py').read_text())
    affine=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='affine_patch')
    scope=dict(np=np,Image=Image,ImageEnhance=ImageEnhance,ImageFilter=ImageFilter,WIDTH=480,HEIGHT=192)
    exec(compile(ast.Module(body=[affine],type_ignores=[]),'private-affine','exec'),scope)
    _,mapped=scope['affine_patch'](Image.new('RGB',(480,192)),np.asarray([[50,70,280,125]],np.float32),new_z,new_z,4.0959770676002165,0.)
    assert inside(mapped[0])
    augment=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='augment_patch')
    assert len([n for n in ast.walk(augment) if isinstance(n,ast.Call) and ast.unparse(n.func)=='affine_patch'])==1
    assert 'eligible &= inside_canvas()' in ast.unparse(augment)
    # Independently recompute all aggregates from saved paired target records;
    # these are annotation/target counts, not model outputs or image reads.
    details=read(V2/'paired-record-results.json');assert sha(V2/'paired-record-results.json')==proof['paired_results_sha256']
    totals={name:Counter() for name in ('baseline','capped')};groups=defaultdict(lambda:{name:Counter() for name in totals})
    for row in details:
        for name,key in [('baseline','baseline'),('capped','capped_counts')]:
            totals[name].update(row[key]);groups[str(row['row_length'])][name].update(row[key])
    assert {k:dict(v) for k,v in totals.items()}==proof['totals']
    assert {k:{a:dict(b) for a,b in v.items()} for k,v in groups.items()}==proof['by_original_row_length']
    assert len(details)==1280 and len({row['record'] for row in details})==40
    assert {row['epoch'] for row in details}==set(range(32))
    assert sum(row['capped'] for row in details)==proof['proofs']['capped_affine_draws']
    assert proof['proofs']['default_off_paired_actual_patches_records']==80
    assert proof['proofs']['paired_main_rng_records']==1280
    assert proof['proofs']['irreversible_complete_and_point_visibility'] and proof['proofs']['unknowns_never_full']
    assert not any(name in sys.modules for name in ('keras','torch','tensorflow'))
    report={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
            'prototype_spec':{'path':str(V2/'prototype-spec.json'),'sha256':sha(V2/'prototype-spec.json')},
            'prototype_proof':{'path':str(V2/'proof-report.json'),'sha256':sha(V2/'proof-report.json')},
            'helper_sha256':sha(V2/'word_anchor_zoom.py'),'source_private_sha256':spec['source_private_sha256'],
            'source_original_sha256':spec['source_original_sha256'],'independent_exact_rational_containment_cases':checks,
            'v1_boundary_defect':{'box':[50,70,280,125],'zoom':z,'translate_x':4.0959770676002165,'translate_y':0.,'old_left':float(apply([50,70,280,125],old_z,4.0959770676002165,0)[0])},
            'v2_exact_boundary_regression_passed':True,'boundary_inset_ulp':inset_ulp,
            'actual_same_Pillow_affine_contains_anchor':True,'single_affine_no_extra_rng_or_raster':True,
            'default_off_and_target_survival_proof_reviewed_not_rerun':True,
            'all1280_saved_target_aggregates_recomputed':True,'original_full_targets':totals['baseline']['original_full'],
            'full_six_bit_targets_baseline':totals['baseline']['full_six_bit_targets'],'full_six_bit_targets_capped':totals['capped']['full_six_bit_targets'],
            'known_original_full_anchor_only':True,'unknown_and_previously_cut_eligibility_preserved':True,
            'source_annotation_or_model_changes_by_reviewer':False,'model_execution':False,'backend_modules_imported':[],
            'limits':['40 metadata-stratified TRAIN contexts at fixed order over32 diagnostic seed phases; not actual source-balanced epoch exposure or recognition accuracy.',
                      'Proof uses actual Patches/targets with synthetic RGB rasters, not photographed-image appearance evidence.',
                      'Only affine containment of one originally full known anchor is promised; later rotation, other cells and grid collisions can still lose targets.',
                      'Within valid full boxes and +/-4% translation the analytic cap is at least .92, so below-.8 fallback cannot be exercised without leaving declared inputs; source fallback remains explicit.',
                      'Private prototype passes for isolated training integration; final CLI/profile/source-bound focused integration tests still belong to trainer.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','report':str(OUT/'report.json'),'sha256':sha(OUT/'report.json'),'independent_cases':checks,'inset_ulp':inset_ulp}))


if __name__=='__main__':main()
