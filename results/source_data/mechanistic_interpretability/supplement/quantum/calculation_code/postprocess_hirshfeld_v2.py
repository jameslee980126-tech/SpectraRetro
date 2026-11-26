#!/usr/bin/env python3
"""Native PySCF Hirshfeld populations from completed fixed-geometry DFT files.

Writes a separate output directory; never modifies or reruns molecular SCFs.
Version 2 normalizes an empty ECP array after PySCF molecule deserialization.
PySCF 2.15's native Hirshfeld routine generates neutral spherical atomic HF
reference densities in the molecular orbital basis. It does run those small
atomic reference calculations internally. Molecular density matrices stay fixed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-mb", type=int, default=8000)
    ap.add_argument("--levels", default="4,5")
    ap.add_argument("--charge-sum-tol", type=float, default=1e-4)
    ap.add_argument("--grid-charge-tol", type=float, default=1e-4)
    ap.add_argument("--grid-fukui-tol", type=float, default=1e-4)
    args = ap.parse_args()
    levels = [int(v) for v in args.levels.split(",")]
    if len(levels) < 2 or levels != sorted(set(levels)) or min(levels) < 0 or max(levels) > 9:
        raise ValueError("Provide at least two increasing distinct grid levels in [0,9]")
    if args.threads < 1 or args.memory_mb < 512:
        raise ValueError("Invalid resource allocation")
    for value in (args.charge_sum_tol, args.grid_charge_tol, args.grid_fukui_tol):
        if not 0 < value <= 1e-3:
            raise ValueError("Require a positive numerical tolerance at most 1e-3")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    run = Path(args.run_dir).resolve()
    out = Path(args.outdir).resolve()
    if out == run or run in out.parents:
        raise ValueError("Postprocessing output must be outside the immutable calculation directory")
    out.mkdir(parents=True, exist_ok=False)
    scratch = out / "scratch"
    scratch.mkdir()
    os.environ["TMPDIR"] = str(scratch)
    os.environ["PYSCF_TMPDIR"] = str(scratch)
    started = time.monotonic()
    status = {"state": "running", "started_at": utc(), "pid": os.getpid(),
              "run_dir": str(run), "script_sha256": sha(__file__)}
    write(out / "status.json", status)
    try:
        import numpy as np
        import pyscf
        from pyscf import dft, gto, lib
        from pyscf.scf import hirshfeld
        lib.num_threads(args.threads)
        if not hasattr(hirshfeld, "hirshfeld_charges"):
            raise RuntimeError("Installed PySCF lacks native Hirshfeld charges")
        source_result = json.loads((run / "result.json").read_text())
        source_manifest = json.loads((run / "artifact_manifest.json").read_text())
        if source_result.get("status") != "complete" or source_result.get("passed") is not True:
            raise RuntimeError("Source molecular calculation is not complete/passed")
        required = ["result.json", "protocol.json", "atoms.json"]
        for state in ("neutral", "cation"):
            required += [f"{state}/molecule.json", f"{state}/wavefunction.npz", f"{state}/result.json"]
        bound_files = {}
        for rel in required:
            actual = sha(run / rel)
            if actual != source_manifest[rel]["sha256"]:
                raise RuntimeError(f"Source artifact changed: {rel}")
            bound_files[rel] = actual
        bound_files["artifact_manifest.json"] = sha(run / "artifact_manifest.json")
        write(out / "source_binding.json", bound_files)
        module_path = Path(inspect.getfile(hirshfeld)).resolve()
        protocol = {
            "analysis": "Native PySCF Hirshfeld charges on saved molecular density matrices",
            "molecular_scf_recomputed": False,
            "script_revision": "v2-empty-ecp-deserialization-repair",
            "molecule_load_normalizations": [],
            "native_api": "pyscf.scf.hirshfeld.hirshfeld_charges(mol, dm, grids, verbose)",
            "reference": "Neutral, spherically averaged atomic HF densities in the molecular orbital basis; native hf.init_guess_by_atom",
            "reference_is_dft": False,
            "native_source_url": "https://raw.githubusercontent.com/pyscf/pyscf/v2.15.0/pyscf/scf/hirshfeld.py",
            "pyscf_version": pyscf.__version__, "numpy_version": np.__version__,
            "native_module_path": str(module_path), "native_module_sha256": sha(module_path),
            "grid_levels": levels, "pruning": "nwchem_prune", "charge_sum_tolerance": args.charge_sum_tol,
            "grid_charge_tolerance": args.grid_charge_tol, "grid_fukui_tolerance": args.grid_fukui_tol,
            "same_grid_for_neutral_and_cation": True,
            "f_minus_definition": "q(cation +1 doublet)-q(neutral singlet)",
            "selected_grid": levels[-1], "atom_indices": "0-based original XYZ order",
            "threads": args.threads, "memory_target_mb": args.memory_mb,
        }
        write(out / "protocol.json", protocol)
        mols, densities, state_metadata = {}, {}, {}
        with (out / "native_analysis.log").open("w", encoding="utf-8") as log:
            for state, charge, spin in (("neutral", 0, 0), ("cation", 1, 1)):
                meta = json.loads((run / state / "result.json").read_text())
                if not meta.get("passed") or not meta.get("converged"):
                    raise RuntimeError(f"Unaccepted source state: {state}")
                mol = gto.loads((run / state / "molecule.json").read_text())
                mol.stdout = log
                mol.verbose = 4
                mol.max_memory = args.memory_mb
                # gto.loads converts serialized [] to shape (0,), whereas the
                # native atomic-reference routine indexes ECP rows as [:, 0].
                # Restore only the empty native table shape; do not rebuild
                # the molecule or reopen its original calculation output.
                ecp_before = list(mol._ecpbas.shape)
                if mol._ecpbas.size == 0:
                    mol._ecpbas = mol._ecpbas.reshape(0, gto.mole.BAS_SLOTS)
                elif mol._ecpbas.ndim != 2 or mol._ecpbas.shape[1] != gto.mole.BAS_SLOTS:
                    raise RuntimeError(f"Invalid nonempty ECP table: {state}")
                if list(mol._ecpbas.shape) != ecp_before:
                    protocol["molecule_load_normalizations"].append({
                        "state": state, "field": "_ecpbas", "shape_before": ecp_before,
                        "shape_after": list(mol._ecpbas.shape), "element_count": 0,
                        "reason": "Restore empty native ECP table after gto.loads"})
                    write(out / "protocol.json", protocol)
                if mol.charge != charge or mol.spin != spin:
                    raise RuntimeError(f"Incorrect source charge/spin: {state}")
                with np.load(run / state / "wavefunction.npz", allow_pickle=False) as data:
                    dm = np.array(data["total_density"])
                    spin_dm = np.array(data["density"])
                    stored_overlap = np.array(data["overlap"])
                    stored_coords = np.array(data["coords_bohr"])
                    stored_z = np.array(data["nuclear_charges"])
                expected_dm = spin_dm if spin_dm.ndim == 2 else spin_dm.sum(axis=0)
                if not np.isfinite(dm).all() or not np.allclose(dm, expected_dm, atol=1e-12, rtol=0):
                    raise RuntimeError(f"Source density mismatch: {state}")
                if not np.allclose(stored_coords, mol.atom_coords(), atol=1e-12, rtol=0) or not np.array_equal(stored_z, mol.atom_charges()):
                    raise RuntimeError(f"Source geometry/element mismatch: {state}")
                overlap = mol.intor_symmetric("int1e_ovlp")
                if not np.allclose(overlap, stored_overlap, atol=1e-10, rtol=0):
                    raise RuntimeError(f"Source AO basis/overlap mismatch: {state}")
                electrons = float(np.einsum("ij,ji->", dm, overlap).real)
                if abs(electrons - mol.nelectron) > 1e-6:
                    raise RuntimeError(f"AO electron count mismatch: {state}")
                mols[state], densities[state], state_metadata[state] = mol, dm, meta
            a, b = mols["neutral"], mols["cation"]
            if a.ao_labels() != b.ao_labels() or not np.array_equal(a._bas, b._bas):
                raise RuntimeError("Neutral/cation AO basis mismatch")
            if not np.array_equal(a.atom_charges(), b.atom_charges()) or not np.array_equal(a.atom_coords(), b.atom_coords()):
                raise RuntimeError("Neutral/cation geometry mismatch")
            if state_metadata["neutral"]["geometry_sha256"] != state_metadata["cation"]["geometry_sha256"]:
                raise RuntimeError("Neutral/cation geometry source hash mismatch")
            results = []
            ni = dft.numint.NumInt()
            for level in levels:
                status.update(stage=f"grid_{level}", updated_at=utc())
                write(out / "status.json", status)
                grid = dft.gen_grid.Grids(a)
                grid.level = level
                grid.prune = dft.gen_grid.nwchem_prune
                grid.build(with_non0tab=True)
                grid_record = {"grid_level": level, "grid_points": len(grid.weights), "states": {}}
                for state in ("neutral", "cation"):
                    mol, dm = mols[state], densities[state]
                    q = np.asarray(hirshfeld.hirshfeld_charges(mol, dm, grids=grid, verbose=4))
                    rho = ni.get_rho(mol, dm, grid, max_memory=args.memory_mb)
                    electrons = float(np.dot(rho, grid.weights))
                    charge_sum = float(q.sum())
                    checks = {
                        "finite_charges_and_density": bool(np.isfinite(q).all() and np.isfinite(rho).all()),
                        "charge_conservation": abs(charge_sum - mol.charge) <= args.charge_sum_tol,
                        "numerical_electron_conservation": abs(electrons - mol.nelectron) <= args.charge_sum_tol,
                        "partition_matches_total_density": abs(float(mol.atom_charges().sum()) - charge_sum - electrons) <= args.charge_sum_tol,
                    }
                    grid_record["states"][state] = {
                        "charges": q.tolist(), "sum_charges": charge_sum,
                        "integrated_electrons_grid": electrons, "expected_electrons": mol.nelectron,
                        "checks": checks, "passed": all(checks.values()),
                    }
                fminus = np.asarray(grid_record["states"]["cation"]["charges"]) - np.asarray(grid_record["states"]["neutral"]["charges"])
                grid_record["f_minus"] = fminus.tolist()
                grid_record["f_minus_sum"] = float(fminus.sum())
                grid_record["f_minus_sum_passed"] = abs(float(fminus.sum()) - 1) <= 2 * args.charge_sum_tol
                results.append(grid_record)
                write(out / f"grid_{level}.json", grid_record)
                log.flush()
            comparisons = []
            for previous, current in zip(results[:-1], results[1:]):
                state_deltas = {state: float(np.max(np.abs(np.asarray(current["states"][state]["charges"]) - np.asarray(previous["states"][state]["charges"])))) for state in ("neutral", "cation")}
                fdelta = float(np.max(np.abs(np.asarray(current["f_minus"]) - np.asarray(previous["f_minus"]))))
                comparisons.append({"coarser_grid": previous["grid_level"], "finer_grid": current["grid_level"],
                                    "max_charge_change_e": state_deltas, "max_f_minus_change": fdelta,
                                    "passed": max(state_deltas.values()) <= args.grid_charge_tol and fdelta <= args.grid_fukui_tol})
            selected = results[-1]
            checks = {"source_molecular_results_verified": True,
                      "all_grids_conserve_electrons": all(r["states"][s]["passed"] and r["f_minus_sum_passed"] for r in results for s in ("neutral", "cation")),
                      "grid_refinement_passed": all(r["passed"] for r in comparisons)}
            result = {"status": "complete" if all(checks.values()) else "needs_attention", "passed": all(checks.values()),
                      "checks": checks, "selected_grid_level": levels[-1],
                      "neutral_hirshfeld_charges": selected["states"]["neutral"]["charges"],
                      "cation_hirshfeld_charges": selected["states"]["cation"]["charges"],
                      "f_minus": selected["f_minus"], "f_minus_sum": selected["f_minus_sum"],
                      "grid_comparisons": comparisons,
                      "geometry_sha256": state_metadata["neutral"]["geometry_sha256"],
                      "atom_elements": [a.atom_pure_symbol(i) for i in range(a.natm)],
                      "source_run": str(run), "completed_at": utc(), "elapsed_seconds": time.monotonic()-started}
        # Validate input immutability once more after the entire analysis.
        if any(sha(run / rel) != value for rel, value in bound_files.items()):
            raise RuntimeError("Source changed during population analysis")
        write(out / "result.json", result)
        status.update(state=result["status"], completed_at=utc(), elapsed_seconds=time.monotonic()-started)
        write(out / "status.json", status)
        manifest = {str(p.relative_to(out)): {"sha256": sha(p), "size_bytes": p.stat().st_size}
                    for p in out.rglob("*") if p.is_file() and "scratch" not in p.relative_to(out).parts
                    and p.name not in {"artifact_manifest.json", "launcher.log"}}
        write(out / "artifact_manifest.json", manifest)
        return 0 if result["passed"] else 2
    except BaseException as exc:
        status.update(state="failed", failed_at=utc(), error=f"{type(exc).__name__}: {exc}")
        write(out / "status.json", status)
        (out / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    sys.exit(main())
