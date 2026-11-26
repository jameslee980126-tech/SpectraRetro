"""Relative-path evaluation binding, shared sampling math and native CPU scoring."""
import hashlib,json,os,sys
from pathlib import Path
BUNDLE=Path(__file__).resolve().parent
ROOT=BUNDLE/'test_runs'
WORK=BUNDLE/'work';DATA=BUNDLE/'final_test/prepared';OLD=BUNDLE/'old'
ARMS=['unknown_seed1','unknown_seed2','known_seed1','known_seed2']
def digest(path):
 p=Path(path)
 if not p.is_absolute():p=BUNDLE/p
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8');os.replace(q,p)
def configure(bundle_root,run_root):
 global BUNDLE,ROOT,WORK,DATA,OLD
 BUNDLE=Path(bundle_root).resolve();ROOT=Path(run_root).resolve();WORK=BUNDLE/'work';DATA=BUNDLE/'final_test/prepared';OLD=BUNDLE/'old'
 assert ROOT==BUNDLE or BUNDLE in ROOT.parents
 os.environ['RETRO_STAGE2_BUNDLE_ROOT']=str(BUNDLE);os.environ['RETRO_STAGE2_OLD_ROOT']=str(OLD);os.environ['RETRO_STAGE2_CYCLE_ROOT']=str(BUNDLE/'cycle')
 for p in [BUNDLE,WORK,OLD,OLD/'stage2']:sys.path.insert(0,str(p))
def seed(value):
 import torch
 torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def setup():
 import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.78)
