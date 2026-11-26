"""Train and evaluate unknown-class FSpec reaction-center identification."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from spectraretro.reaction_center import training as common
from spectraretro.reaction_center.features import BOND_COLUMNS, NODE_COLUMNS, SCHEMA
from spectraretro.reaction_center.unknown_class.prior_model import build_task
import numpy as np
import torch
from rdkit import rdBase
from torchdrug import data


def source_manifest():
    return common.source_manifest(Path(__file__).resolve().parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("baseline", "prior"))
    for name in ("train", "valid", "test"):
        parser.add_argument("--" + name + "-csv", type=Path,
                            default=common.REPOSITORY / "datasets/full" / (name + ".csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-batch-policy", choices=("first", "largest"), default="largest")
    parser.add_argument("--expected-common-sha256")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.width, args.layers, args.batch_size) <= 0 or args.epochs > 100:
        parser.error("Require positive settings and at most 100 epochs")
    if len({q.resolve() for q in (args.train_csv, args.valid_csv, args.test_csv)}) != 3:
        parser.error("Train, valid and test must be distinct files")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(args.device)
    sources = source_manifest()
    common.seed_all(args.seed)
    train, train_rows, train_index, train_report = common.load_split(args.train_csv, sources)
    valid, valid_rows, valid_index, valid_report = common.load_split(args.valid_csv, sources)
    common.seed_all(args.seed)
    task = build_task(train, valid, prior=args.arm == "prior",
                      width=args.width, num_layers=args.layers).to(device)
    common_sha = common.common_state_hash(task)
    saved_contract = None
    if args.resume:
        saved_contract = json.loads((args.output / "contract.json").read_text())
    paired_sha = saved_contract["common_initial_sha256"] if saved_contract else common_sha
    if args.expected_common_sha256 and paired_sha != args.expected_common_sha256:
        raise RuntimeError("Paired common parameter initialization mismatch")
    optimizer = torch.optim.Adam(task.parameters(), lr=args.lr, weight_decay=0.)
    contract = dict(
        arm=args.arm, known_class=False, reaction_class_usage="reporting metadata only",
        backbone="FSpecFaithfulReactionCenter", dual_branch=False, seed=args.seed,
        epochs=args.epochs, patience=args.patience, width=args.width, layers=args.layers,
        batch_size=args.batch_size, lr=args.lr, dropout=0.,
        candidate_domain="canonical bonds plus atoms",
        selection="max valid top-1; earliest tie", common_initial_sha256=common_sha,
        descriptor_schema=SCHEMA, node_columns=NODE_COLUMNS, bond_columns=BOND_COLUMNS,
        test_loaded_for_selection=False, sources=sources,
        data={name: common.sha(getattr(args, name + "_csv")) for name in ("train", "valid", "test")},
        indices={name: common.sha(getattr(args, name + "_csv").with_name(name + "_index.csv"))
                 for name in ("train", "valid", "test")
                 if getattr(args, name + "_csv").with_name(name + "_index.csv").exists()},
        train=train_report, valid=valid_report,
        parameters=sum(p.numel() for p in task.parameters()), device=str(device),
        torch=torch.__version__, numpy=np.__version__, rdkit=rdBase.rdkitVersion)
    contract = json.loads(json.dumps(contract))
    if saved_contract is not None:
        current_sources = dict(contract["sources"])
        saved_sources = dict(saved_contract["sources"])
        migration_runner = Path(__file__).resolve().relative_to(common.REPOSITORY).as_posix()
        current_runner_sha = current_sources.pop(migration_runner, None)
        saved_runner_sha = saved_sources.pop(migration_runner, None)
        if current_sources != saved_sources:
            raise RuntimeError("Cross-platform resume changed non-migration source files")
        current_semantic = dict(contract)
        saved_semantic = dict(saved_contract)
        for value in (current_semantic, saved_semantic):
            value.pop("sources", None)
            value.pop("common_initial_sha256", None)
        if current_semantic != saved_semantic:
            raise RuntimeError("Cross-platform resume changed data, configuration, or runtime contract")
        common.save_json(args.output / "resume_migration.json", dict(
            status="validated", reason="cross-platform continuation from Windows to Linux",
            checkpoint_common_initial_sha256=saved_contract["common_initial_sha256"],
            discarded_fresh_linux_initial_sha256=common_sha,
            expected_common_sha256=args.expected_common_sha256,
            migration_runner_saved_sha256=saved_runner_sha,
            migration_runner_current_sha256=current_runner_sha,
            invariant_non_migration_sources=True,
            invariant_data_configuration_runtime=True,
            model_optimizer_and_rng_loaded_from_checkpoint=True))
        # Preserve the original experimental contract in all continued checkpoints;
        # the migration-specific runner identity is recorded separately above.
        contract = saved_contract
    if args.preflight_only:
        indices = list(range(len(train)))
        if args.preflight_batch_policy == "largest":
            indices.sort(key=lambda i: int(train.data[i][1].num_node), reverse=True)
        selected = indices[:args.batch_size]
        batch = common.move(next(iter(data.DataLoader([train[i] for i in selected],
                                                      len(selected), shuffle=False))), device)
        loss, _ = task(batch)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                           for p in task.parameters()):
            raise FloatingPointError("Preflight nonfinite loss/gradient")
        common.save_json(args.output / "preflight.json", dict(status="passed", loss=float(loss.detach()),
                         contract=contract, optimizer_steps=0, selected_train_row_indices=selected))
        return
    completed, best, best_epoch, stale = 0, -float("inf"), 0, 0
    if args.resume:
        if saved_contract != contract:
            raise RuntimeError("Resume contract differs")
        state = common.torch_load(args.output / ("last.pth" if (args.output / "last.pth").exists() else "best.pth"))
        if state["contract"] != contract:
            raise RuntimeError("Checkpoint contract differs")
        task.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        completed, best, best_epoch, stale = (state[k] for k in ("epoch", "best", "best_epoch", "stale"))
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        common.save_json(args.output / "contract.json", contract)
    common.save_json(args.output / "status.json", dict(status="training", pid=os.getpid(), epoch=completed))
    for epoch in range(completed + 1, args.epochs + 1):
        if stale >= args.patience:
            break
        task.train(); started = time.perf_counter(); seen = 0; loss_sum = 0.
        for batch_index, batch in enumerate(data.DataLoader(train, args.batch_size, shuffle=True, num_workers=0), 1):
            batch = common.move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = task(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in task.parameters()):
                raise FloatingPointError("Nonfinite training gradient")
            optimizer.step()
            count = len(batch["reaction"]); seen += count; loss_sum += float(loss.detach()) * count
            if batch_index == 1 or batch_index % 100 == 0:
                progress = dict(status="training", pid=os.getpid(), epoch=epoch, batch=batch_index,
                                train_rows_seen=seen, train_rows_total=len(train),
                                train_loss_running=loss_sum / seen,
                                best=best if np.isfinite(best) else None, best_epoch=best_epoch,
                                epoch_elapsed_seconds=time.perf_counter() - started)
                common.save_json(args.output / "status.json", progress)
                print(json.dumps(progress), flush=True)
        metrics = common.evaluate(task, valid, valid_rows, valid_index, args.batch_size, device)
        improved = metrics["overall"]["top-1 accuracy"] > best
        if improved: best, best_epoch, stale = metrics["overall"]["top-1 accuracy"], epoch, 0
        else: stale += 1
        state = common.checkpoint(task, optimizer, epoch, best, best_epoch, stale, contract)
        if improved: common.save_checkpoint(args.output / "best.pth", state)
        common.save_checkpoint(args.output / "last.pth", state)
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_rows_seen=seen,
                   valid=metrics, best=best, best_epoch=best_epoch, stale=stale,
                   epoch_seconds=time.perf_counter() - started)
        with (args.output / "metrics.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(row) + "\n")
        common.save_json(args.output / "status.json", dict(status="training", pid=os.getpid(), **row))
        print(json.dumps(row), flush=True)
    common.save_json(args.output / "selection_freeze.json", dict(
        status="frozen", best_epoch=best_epoch, valid_top1=best,
        best_sha256=common.sha(args.output / "best.pth"), test_loaded=False))
    task.load_state_dict(common.torch_load(args.output / "best.pth")["model"], strict=True)
    valid_metrics = common.evaluate(task, valid, valid_rows, valid_index, args.batch_size, device,
                                    args.output / "valid_predictions.jsonl")
    test, test_rows, test_index, test_report = common.load_split(args.test_csv, sources)
    test_metrics = common.evaluate(task, test, test_rows, test_index, args.batch_size, device,
                                   args.output / "test_predictions.jsonl")
    result = dict(status="completed", arm=args.arm, known_class=False,
                  best_epoch=best_epoch, valid=valid_metrics, test=test_metrics,
                  test_descriptor_coverage=test_report, common_initial_sha256=common_sha,
                  interpretation="single-seed exploratory high/medium/low similarity pilot")
    common.save_json(args.output / "result.json", result)
    common.save_json(args.output / "status.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
