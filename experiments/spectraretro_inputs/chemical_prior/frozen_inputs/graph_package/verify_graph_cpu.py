"""CPU engineering checks on frozen real rows; no model training or accuracy claim."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from graph_model import OLD, CYCLE, HERE, UnknownObservedGraphDDPM, CLASS_FIELDS, MODEL_FIELDS
from graph_data import collate_graph, inference_inputs
sys.path.insert(0, str(OLD))
from stage2.train import default_config
from unknown_model import UnknownCompatibilityDDPM
from src import utils
from src.datasets import create_templates_for_linker_generation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def rejects(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return True
    raise AssertionError('Forbidden input was accepted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'cpu_verification.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Verification record already exists; use a new output path')
    begun = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rows = torch.load(OLD / 'newdata/eval_geometry.pt', map_location='cpu', weights_only=False)
    positive = next(r for r in rows if int(r['linker_mask'].sum()) > 0 and len(r['positions']) <= 14)
    zero = next(r for r in rows if int(r['linker_mask'].sum()) == 0 and 5 <= len(r['positions']) <= 14)
    examples = [positive, zero]
    batch = collate_graph(examples)
    assert set(batch) <= MODEL_FIELDS and not CLASS_FIELDS & set(batch)
    configuration = default_config('frozen_rows_only', 'cpu', 2)
    models = {}
    for arm in ['control', 'compatibility', 'graph']:
        torch.manual_seed(0)
        models[arm] = UnknownObservedGraphDDPM(observed_graph_enabled=arm == 'graph',
            compatibility_enabled=arm == 'compatibility', **configuration)
    hashes = {arm: state_hash(model) for arm, model in models.items()}
    assert len(set(hashes.values())) == 1
    original = UnknownCompatibilityDDPM(compatibility_enabled=False, **configuration)
    source_state = {k: v for k, v in models['control'].state_dict().items()
        if not k.startswith('edm.observed_graph_adapter.')}
    original.load_state_dict(source_state, strict=True)
    prior_batch = {k: v for k, v in batch.items() if not k.startswith('observed_')}
    torch.manual_seed(903)
    old_loss = original.forward(prior_batch, True)
    losses = {}
    for arm in ['control', 'graph']:
        torch.manual_seed(903)
        losses[arm] = models[arm].forward(batch, True)
        assert all(torch.equal(x, y) if torch.is_tensor(x) else x == y
            for x, y in zip(losses[arm], old_loss))
    adapter = models['graph'].edm.observed_graph_adapter
    losses['graph'][4].backward()
    first_projection_grad = float(adapter.projection.weight.grad.norm())
    assert first_projection_grad > 0
    first_base_grad = sum(float(p.grad.norm()) for p in models['graph'].edm.dynamics.base.parameters() if p.grad is not None)
    assert first_base_grad > 0
    optimizer = torch.optim.AdamW(models['graph'].edm.parameters(), lr=2e-4, weight_decay=1e-12, amsgrad=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(904)
    second_loss = models['graph'].forward(batch, True)[4]
    second_loss.backward()
    second_encoder_grad = float(adapter.input.weight.grad.norm())
    assert second_encoder_grad > 0 and torch.isfinite(second_loss)
    graph_effect = None
    with torch.no_grad():
        torch.manual_seed(905)
        graph_loss = models['graph'].forward(batch, True)[4]
        models['control'].load_state_dict(models['graph'].state_dict(), strict=True)
        torch.manual_seed(905)
        control_loss = models['control'].forward(batch, True)[4]
        graph_effect = float(abs(graph_loss - control_loss))
        assert graph_effect > 0
    guard_checks = {}
    for field in sorted(CLASS_FIELDS):
        guard_checks['class_rejected_' + field] = rejects(lambda field=field: models['control'].forward(
            dict(batch, **{field: torch.zeros(2)}), True))
    guard_checks['train_target_sidechannel_rejected'] = rejects(lambda: models['graph'].forward(
        dict(batch, full_bond_index=torch.zeros(2, 1, dtype=torch.long)), True))
    guard_checks['combined_adapters_rejected'] = rejects(lambda: UnknownObservedGraphDDPM(
        compatibility_enabled=True, observed_graph_enabled=True, **configuration))
    guard_checks['implicit_reference_size_rejected'] = rejects(lambda: models['graph'].sample_chain(batch))
    bad_features = batch['observed_features'].clone()
    n_known = int(batch['fragment_mask'][0].sum())
    bad_features[0, n_known, 0] = 1
    guard_checks['unknown_graph_features_rejected'] = rejects(lambda: models['graph'].forward(
        dict(batch, observed_features=bad_features), True))
    bad_adjacency = batch['observed_adjacency'].clone()
    bad_adjacency[0, 0, n_known] = bad_adjacency[0, n_known, 0] = True
    guard_checks['unknown_graph_edges_rejected'] = rejects(lambda: models['graph'].forward(
        dict(batch, observed_adjacency=bad_adjacency), True))
    assert all(guard_checks.values())

    clean = inference_inputs(batch)
    poison = copy.deepcopy(batch)
    for field, value in [('positions', float('nan')), ('one_hot', float('nan')), ('charges', float('inf'))]:
        poison[field] = torch.where(poison['fragment_mask'].bool(), poison[field],
            torch.full_like(poison[field], value))
    # Change reference missing lengths and every target-only count/mask without
    # changing the observed prefix or graph. The safe sampler inputs are equal.
    for field in ['positions', 'one_hot', 'charges', 'anchors', 'fragment_mask', 'linker_mask', 'observed_features']:
        v = poison[field]
        poison[field] = torch.cat([v, torch.zeros(v.shape[0], 7, v.shape[-1], dtype=v.dtype)], 1)
    adj = poison['observed_adjacency']
    expanded = torch.zeros(adj.shape[0], adj.shape[1] + 7, adj.shape[2] + 7, dtype=adj.dtype)
    expanded[:, :adj.shape[1], :adj.shape[2]] = adj
    poison['observed_adjacency'] = expanded
    poison['num_atoms'] = [999, 888]
    poison['linker_mask'] = 1 - poison['fragment_mask']
    poison['atom_mask'] = torch.ones_like(poison['fragment_mask'])
    poison['edge_mask'] = torch.ones(1)
    poison['full_bond_index'] = torch.ones(2, 50)
    poison['reaction_class'] = torch.ones(2, 6)
    poisoned_clean = inference_inputs(poison)
    assert same(clean, poisoned_clean)
    size_bank = torch.tensor([2, 0], dtype=torch.long)
    template = create_templates_for_linker_generation(clean, size_bank)
    for i, count in enumerate(clean['num_atoms']):
        assert int(template['atom_mask'][i].sum()) == count + int(size_bank[i])
        for field in ['positions', 'one_hot', 'charges', 'anchors', 'fragment_mask']:
            assert torch.equal(template[field][i, :count], batch[field][i, :count])
    models['graph'].edm.T = 100
    models['graph'].eval()
    callback_input_verified = []
    def predicted_sizes(data):
        callback_input_verified.append(same(data, clean))
        assert same(data, clean)
        return size_bank
    with torch.no_grad():
        torch.manual_seed(991)
        chain, mask = models['graph'].sample_chain(clean, sample_fn=predicted_sizes, keep_frames=1)
        torch.manual_seed(991)
        poisoned_chain, poisoned_mask = models['graph'].sample_chain(poisoned_clean,
            sample_fn=predicted_sizes, keep_frames=1)
    assert torch.isfinite(chain).all() and torch.equal(chain, poisoned_chain) and torch.equal(mask, poisoned_mask)
    centered = utils.remove_partial_mean_with_mask(batch['positions'], batch['atom_mask'], batch['fragment_mask'])
    errors = []
    for i, count in enumerate(clean['num_atoms']):
        errors.append(float((chain[0, i, :count, :3] - centered[i, :count]).abs().max()))
        assert errors[-1] == 0
        assert torch.equal(chain[0, i, :count, 3:].argmax(-1), batch['one_hot'][i, :count].argmax(-1))
    observed_availability = {'rows': len(rows), 'graph_available': sum(bool(r['observed_graph_available']) for r in rows)}
    for row in rows:
        nf = int(row['fragment_mask'].sum())
        assert not row['observed_features'][nf:].any()
        assert not row['observed_adjacency'][nf:].any() and not row['observed_adjacency'][:, nf:].any()
        assert torch.equal(row['observed_adjacency'], row['observed_adjacency'].T)
        if row['observed_graph_available']:
            assert torch.equal(row['observed_features'][:nf, :26], row['one_hot'][:nf])
    result = dict(passed=True, scope='CPU engineering only; two original-loss updates on temporary model, no saved trained weights',
        no_gpu=True, no_test_read=True, equal_initialization=hashes,
        original_loss_exact_at_zero_residual=True, first_projection_gradient_norm=first_projection_grad,
        first_base_gradient_sum=first_base_grad, second_graph_encoder_gradient_norm=second_encoder_grad,
        graph_toggle_changes_original_l2_after_one_update=graph_effect, guards=guard_checks,
        sampler_steps=100, explicit_sampled_sizes=size_bank.tolist(), callback_saw_known_only_inputs=all(callback_input_verified),
        unknown_target_value_and_size_poison_inference_inputs_exact=True,
        unknown_target_poison_generation_exact=True, known_coordinate_error=errors,
        known_elements_exact=True, sampled_counts=mask.sum(1).view(-1).tolist(),
        all_eval_observed_target_masks_checked=observed_availability, example_uuids=[int(r['uuid']) for r in examples],
        configuration=configuration, elapsed_seconds=time.monotonic()-begun,
        source_sha256={str(p):digest(p) for p in [HERE/'graph_model.py',HERE/'graph_data.py',Path(__file__),
            OLD/'internal_model.py', OLD/'internal_data.py', OLD/'prepare_geometry_data.py',
            CYCLE/'unknown_compatibility/unknown_model.py', OLD/'newdata/eval_geometry.pt',OLD/'newdata/schema.json']})
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: result[k] for k in ['passed','first_projection_gradient_norm',
        'second_graph_encoder_gradient_norm','sampler_steps','known_coordinate_error','elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
