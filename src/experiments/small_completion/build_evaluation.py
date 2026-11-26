"""Reuse prepared full-test inputs; adapt class mode without changing score math."""
import difflib,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'stage2_effective_modules_20261010'
REPO=ROOT.parents[2]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(p,rel):
 q=ROOT/rel;q.parent.mkdir(parents=True,exist_ok=True)
 if not q.exists():shutil.copy2(p,q)
 assert digest(q)==digest(p)
def adapt(name,source,changes):
 before=source.read_text(encoding='utf8');s=before
 for a,b in changes:
  if a not in s:raise ValueError('Missing adapter anchor '+a[:90])
  s=s.replace(a,b)
 (ROOT/name).write_text(s,encoding='utf8')
 (ROOT/(name+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile=str(source),tofile=name)),encoding='utf8')
def main():
 for p in (SOURCE/'final_test/prepared').iterdir():
  if p.is_file():copy(p,'final_test/prepared/'+p.name)
 for p in (SOURCE/'final_test/reference').iterdir():
  if p.is_file():copy(p,'final_test/reference/'+p.name)
 copy(SOURCE/'final_test/run/reference_replay.csv','final_test/reference/reference_replay.csv')
 copy(REPO/'experiments/spectraretro_inputs/small/prepared/stage1_known_test_predictions.jsonl','final_test/prepared/stage1_known_test_predictions.jsonl')
 for p in (SOURCE/'work/fusion_reranker').rglob('*'):
  if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts:copy(p,'work/fusion_reranker/'+p.relative_to(SOURCE/'work/fusion_reranker').as_posix())
 copy(SOURCE/'eval_package/decode.py','decode.py')
 # Retain the exact native Windows decoder binary used in the original study.
 for p in (SOURCE/'runtime/decoder_dependencies').rglob('*'):
  if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts:copy(p,'runtime/decoder_dependencies/'+p.relative_to(SOURCE/'runtime/decoder_dependencies').as_posix())
 # The native decoder root changes by one level when placed in this bundle.
 decode=(ROOT/'decode.py').read_text();decode=decode.replace("Path(__file__).resolve().parents[1] / 'runtime/decoder_dependencies'","Path(__file__).resolve().parent / 'runtime/decoder_dependencies'");(ROOT/'decode.py').write_text(decode)
 sampler=SOURCE/'final_test/test_sample.py'
 before=sampler.read_text();start=before.index('def load_model(arm):');end=before.index('\ndef save_npz',start)
 new='''def load_model(arm):
    from composition_model import construct
    c=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if c['arm']!=arm or not c['compatibility_enabled'] or c['observed_graph_enabled']:
        raise ValueError('Checkpoint method mismatch')
    mode=arm.split('_seed')[0]
    if bool(c['class_input'])!=(mode=='known'):raise ValueError('Class protocol mismatch')
    model=construct(mode,**c['configuration'])
    model.load_state_dict(c['state_dict'],strict=True);model.edm.T=100
    return model.eval().cuda()
'''
 changes=[(before[start:end],new),('import portable_common as common','import eval_runtime as common'),('from portable_common import *','from eval_runtime import *'),("safe=inference_inputs(batch);sizes=shared['sizes'][ids]","safe=inference_inputs(batch);sizes=shared['sizes'][ids]\n                if a.arm.startswith('known_'):\n                    safe['reaction_class']=torch.nn.functional.one_hot(torch.tensor([int(rows[i]['class_index']) for i in ids],device='cuda'),6).float()")]
 adapt('test_sample.py',sampler,changes)
 adapt('validate_test.py',SOURCE/'final_test/validate_raw.py',[
  ('import portable_common as common','import eval_runtime as common'),('class_input=False','class_input=args.arm.startswith(\'known_\')')])
 adapt('score_test.py',SOURCE/'final_test/test_score.py',[
  ('import portable_common as common','import eval_runtime as common'),('from portable_common import *','from eval_runtime import *'),
  ("ranker=FrozenFusionRanker('unknown','fusion');metrics={};individual={}","mode=a.arm.split('_seed')[0];ranker=FrozenFusionRanker(mode,'fusion');metrics={};individual={}"),
  ("(common.BUNDLE/'final_test/prepared/stage1_test_predictions.jsonl')","(common.BUNDLE/('final_test/prepared/stage1_known_test_predictions.jsonl' if a.arm.startswith('known_') else 'final_test/prepared/stage1_test_predictions.jsonl'))"),
  ("request={'reaction_id':key,'product':products.pop(),'fragments':[]}","request={'reaction_id':key,'product':products.pop(),'fragments':[]}\n            if mode=='known':request['class_id']=report_class")])
 print(json.dumps({'prepared_full_test':4192,'reactions':2400,'draws':300,'ranking_source_unchanged':True,'known_size_policy':'same frozen learned class-free size bank for both modes; explicit class in known diffusion/Stage1/ranking'}))
if __name__=='__main__':main()
