"""Two frozen, product-only root cases; reuse previous inference verbatim."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent / 'paper_redesign/recursive_probe'
sys.path.insert(0, str(PROBE))
import run_recursive as engine

ROOTS = [
    dict(id='9773', smiles='COc1cc(-c2ccc(C(=O)C(OC)c3ccc(N4CCOCC4)cc3)o2)cc(OC)c1OC(=O)c1ccccc1'),
    dict(id='9207', smiles='O=C(c1cc(-c2ccc(F)cc2)c(-c2ccc(F)cc2)s1)C(F)(F)F'),
]
OUT = HERE / 'product_only_v1'


def main():
    previous = json.loads((PROBE/'product_only_v1/protocol.json').read_text())
    assert engine.sha(PROBE/'run_recursive.py') == previous['runner_sha256']
    assert engine.sha(engine.CHECKPOINT) == previous['checkpoints']['diffusion']
    assert engine.sha(engine.SIZE_CHECKPOINT) == previous['checkpoints']['size']
    assert engine.torch.__version__ == previous['torch']
    protocol = dict(previous, roots=ROOTS, max_depth=1,
        scope='Two product-only single-step validation cases; identical root inference to previous three cases; exploratory, not official evaluation',
        expansion='None: validate root step only', runner_sha256=engine.sha(Path(__file__)),
        shared_engine_sha256=engine.sha(PROBE/'run_recursive.py'),
        helper_sha256={p.name:engine.sha(p) for p in [PROBE/'score_product.py',PROBE/'score_next_round.py',PROBE/'decode_candidates.py',engine.EXP/'runtime_common.py']})
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        engine.write(OUT/'protocol.json',protocol)
    engine.setup(2)
    engine.seed(0)
    engine.RDLogger.DisableLog('rdApp.*')
    model=engine.create(3,engine.torch.device('cuda'))
    model.load_state_dict(engine.torch.load(engine.CHECKPOINT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
    model.eval()
    size_model=engine.create(3,engine.torch.device('cuda'),size=True)
    size_model.load_state_dict(engine.torch.load(engine.SIZE_CHECKPOINT,map_location='cpu',weights_only=False)['state_dict'],strict=True)
    size_model.eval()
    model.edm.T=100
    results=[]
    for root in ROOTS:
        node=dict(id=root['id']+'_root',case_id=root['id'],depth=0,ancestors=[],smiles=engine.canonical(root['smiles']))
        folder=OUT/node['id']
        engine.write(OUT/'status.json',dict(state='running',active_node=node['id'],completed_nodes=[r['id'] for r in results]))
        if (folder/'result.json').exists():
            result=json.loads((folder/'result.json').read_text())
        else:
            result=engine.generate_node(node,model,size_model,folder,30,100)
        results.append(result)
        engine.write(OUT/'results.json',dict(protocol=protocol,nodes=results))
    engine.write(OUT/'status.json',dict(state='completed',completed_nodes=[r['id'] for r in results]))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        engine.write(OUT/'failure.json',dict(error=repr(exc)))
        raise
