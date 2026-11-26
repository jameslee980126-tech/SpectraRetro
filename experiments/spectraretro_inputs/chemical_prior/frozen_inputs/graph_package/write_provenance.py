"""Seal ready graph code and CPU evidence without changing external files."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = HERE / 'provenance.json'
    if target.exists():
        raise FileExistsError('Package provenance already sealed')
    cpu = json.loads((HERE/'cpu_verification.json').read_text(encoding='utf8'))
    audit = json.loads((HERE/'observed_cache_audit.json').read_text(encoding='utf8'))
    assert cpu['passed'] and audit['passed']
    files = [p for p in HERE.iterdir() if p.is_file() and p.name != target.name]
    record = {'created_utc':datetime.now(timezone.utc).isoformat(), 'ready_for_remote_gpu_preflight':True,
        'no_gpu_execution_in_this_package_preparation':True, 'no_long_training_started':True,
        'original_source_edits':False, 'no_test_read':True,
        'derivation':'Graph adapter structure transplanted from frozen OLD/internal_model.py; class-free parent CYCLE/unknown_compatibility/unknown_model.py',
        'method_changes':['Two observed-synthon-only bond message passes and anchor pooled bounded scalar embedding residual, zero-initialized projection'],
        'shared_engineering_adaptations':['Same complete adapter state in every control/compatibility/graph/physics arm',
            'Known-prefix-only sampling input: crop to observed maximum, rebuild masks and counts, remove unknown target values and counts',
            'Environment-root imports, strict class/sidechannel input schema'],
        'unchanged':['Original EDM L2 loss','128 hidden dimensions','Eight EGNN blocks','26 element schema',
            'No full target graph input','No joint bond prediction','No added CE','No reranking code'],
        'exports':['UnknownObservedGraphDDPM','collate_graph','inference_inputs'],
        'source_and_evidence_sha256':{p.name:sha(p) for p in sorted(files)},
        'upstream_source_sha256':cpu['source_sha256'], 'observed_source_sha256':audit['source_hashes']}
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'sealed':str(target),'files':len(files)},indent=2))


if __name__ == '__main__':
    main()
