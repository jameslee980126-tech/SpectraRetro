#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"expected one RetroXpert source marker, found {source.count(old)}")
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--warmup-validations", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--move-graphs-to-input-device", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args, runner_args = parser.parse_known_args()

    source = args.runner.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "        if epoch % 5 == 0:\n",
        "        if epoch % 1 == 0:\n",
    )
    view_expression = ".view(-1, 1)"
    if source.count(view_expression) != 4:
        raise RuntimeError(
            f"expected four RetroXpert adjacency views, found {source.count(view_expression)}"
        )
    source = source.replace(view_expression, ".reshape(-1, 1)")
    if args.move_graphs_to_input_device:
        graph_batch = "g_dgl = dgl.batch(x_graph)"
        if source.count(graph_batch) != 2:
            raise RuntimeError(
                f"expected two RetroXpert graph batches, found {source.count(graph_batch)}"
            )
        source = source.replace(graph_batch, graph_batch + ".to(x_atom.device)")

    state_marker = "    # Record epoch start time\n"
    state_block = f'''    _early_stop_event_log = r"{args.event_log}"
    _early_stop_warmup = {args.warmup_validations}
    _early_stop_min_delta = {args.min_delta!r}
    _early_stop_patience = {args.patience}
    _early_stop_best_metric = None
    _early_stop_significant_best = None
    _early_stop_stalled = 0
    _early_stop_validation_count = 0
    _early_stop_checkpoint = 'checkpoints/{{}}_checkpoint.pt'.format(args.exp_name)
    with open(_early_stop_event_log, 'a', encoding='utf-8') as _event_handle:
        _event_handle.write(json.dumps({{
            'event': 'early_stop_monitor_started',
            'metric': 'validation_bond_disconnection_accuracy',
            'validation_interval_epochs': 1,
            'warmup_validations': _early_stop_warmup,
            'min_delta': _early_stop_min_delta,
            'patience': _early_stop_patience,
            'graph_device': 'input_tensor_device',
            'model_source_changed': False,
        }}, sort_keys=True) + '\\n')

'''
    source = replace_once(source, state_marker, state_block + state_marker)

    validation_marker = "            csv_logger.writerow(row)\n"
    validation_block = '''            _early_stop_validation_count += 1
            _early_stop_metric = float(valid_acc)
            if _early_stop_best_metric is None or _early_stop_metric > _early_stop_best_metric:
                _early_stop_best_metric = _early_stop_metric
                torch.save(GAT_model.state_dict(), _early_stop_checkpoint)
            if _early_stop_significant_best is None:
                _early_stop_gain = _early_stop_metric
                _early_stop_improved = True
            else:
                _early_stop_gain = _early_stop_metric - _early_stop_significant_best
                _early_stop_improved = _early_stop_gain >= _early_stop_min_delta
            if _early_stop_improved:
                _early_stop_significant_best = _early_stop_metric
            _early_stop_eligible = _early_stop_validation_count > _early_stop_warmup
            if _early_stop_eligible:
                _early_stop_stalled = 0 if _early_stop_improved else _early_stop_stalled + 1
            else:
                _early_stop_stalled = 0
            _early_stop_payload = {
                'event': 'validation_checked',
                'epoch': epoch,
                'metric': 'validation_bond_disconnection_accuracy',
                'best_metric': _early_stop_best_metric,
                'significant_best': _early_stop_significant_best,
                'gain': _early_stop_gain,
                'improved': _early_stop_improved,
                'early_stop_eligible': _early_stop_eligible,
                'stalled': _early_stop_stalled,
            }
            with open(_early_stop_event_log, 'a', encoding='utf-8') as _event_handle:
                _event_handle.write(json.dumps(_early_stop_payload, sort_keys=True) + '\\n')
            if _early_stop_eligible and _early_stop_stalled >= _early_stop_patience:
                with open(_early_stop_event_log, 'a', encoding='utf-8') as _event_handle:
                    _event_handle.write(json.dumps({
                        'event': 'early_stop',
                        'epoch': epoch,
                        'best_metric': _early_stop_best_metric,
                        'selected_checkpoint': _early_stop_checkpoint,
                    }, sort_keys=True) + '\\n')
                print('Early stopping after Epoch {} with best validation accuracy {:.6f}'.format(
                    epoch, _early_stop_best_metric))
                break
'''
    source = replace_once(
        source,
        validation_marker,
        validation_marker + validation_block,
    )

    final_save = """    csv_logger.close()
    torch.save(GAT_model.state_dict(),
               'checkpoints/{}_checkpoint.pt'.format(args.exp_name))
"""
    final_block = """    csv_logger.close()
    if _early_stop_best_metric is None:
        torch.save(GAT_model.state_dict(), _early_stop_checkpoint)
"""
    source = replace_once(source, final_save, final_block)
    code = compile(source, str(args.runner), "exec")
    if args.check_only:
        print(json.dumps({"runner": str(args.runner), "status": "compatible"}))
        return

    os.chdir(args.runner.parent)
    sys.path.insert(0, str(args.runner.parent))
    sys.argv = [str(args.runner), *runner_args]
    namespace = {
        "__file__": str(args.runner),
        "__name__": "__main__",
        "json": json,
    }
    exec(code, namespace)


if __name__ == "__main__":
    main()
