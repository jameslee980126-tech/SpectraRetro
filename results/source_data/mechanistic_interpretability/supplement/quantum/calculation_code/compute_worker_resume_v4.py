#!/usr/bin/env python3
"""Fixed-geometry neutral/cation PBE0 single points; strict stability diagnostics.

The input atom order and geometry are unchanged. Native PySCF Mulliken and
meta-Lowdin populations are saved separately. This script does not optimize
geometries or implement Hirshfeld partitioning. A queue must enforce wall time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024**2)


def parse_xyz(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n = int(lines[0].strip())
    records = [line.split() for line in lines[2:] if line.strip()]
    if len(records) != n or any(len(record) < 4 for record in records):
        raise ValueError("XYZ atom count/columns inconsistent")
    atoms = [(r[0], tuple(float(x) for x in r[1:4])) for r in records]
    return atoms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory-mb", type=int, default=16000)
    ap.add_argument("--resume-from", type=Path, help="Reuse an archived converged neutral SCF, without rerunning its kernel")
    ap.add_argument("--reuse-neutral-stability", type=Path, help="Reuse a validated standalone strict diagnostic for the archived neutral SCF")
    ap.add_argument("--strict-guess", choices=["native_dense", "native_plus_lowdiag"], default="native_dense")
    ap.add_argument("--skip-stability", action="store_true", help="Explicitly skip internal SCF stability; diagnostics report not tested")
    args = ap.parse_args()
    if args.threads < 1 or args.memory_mb < 256:
        raise ValueError("Invalid resource allocation")
    if args.reuse_neutral_stability and (not args.resume_from or args.skip_stability or args.strict_guess != "native_plus_lowdiag"):
        raise ValueError("Neutral stability cache requires --resume-from and --strict-guess native_plus_lowdiag, with stability enabled")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # Refuse accidental overwrite or scientific re-execution in an existing run.
    if (outdir / "status.json").exists() or (outdir / "neutral").exists():
        raise FileExistsError(f"Run directory already contains a calculation: {outdir}")
    scratch = outdir / "scratch"
    scratch.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(scratch)
    os.environ["PYSCF_TMPDIR"] = str(scratch)
    t0 = time.monotonic()
    status = {"state": "running", "started_at": utc(), "pid": os.getpid(),
              "xyz": str(Path(args.xyz).resolve()), "xyz_sha256": sha(args.xyz),
              "worker_sha256": sha(__file__), "threads": args.threads,
              "memory_target_mb": args.memory_mb, "stage": "initializing"}
    atomic_json(outdir / "status.json", status)
    try:
        import numpy as np
        import pyscf
        from pyscf import dft, gto, lib, scf
        import strict_stability_lib_v3 as strict_stability_lib
        import resume_scf_lib_v2 as resume_scf_lib
        import stability_cache_lib
        lib.num_threads(args.threads)
        atoms = parse_xyz(args.xyz)
        if not np.isfinite(np.asarray([a[1] for a in atoms])).all():
            raise ValueError("Non-finite coordinates")
        protocol = {
            "functional": "PBE0", "orbital_basis": "def2-tzvp",
            "auxiliary_basis": "def2-universal-jkfit", "density_fitting": "Coulomb and exchange",
            "grid_level": 4, "grid_pruning": "PySCF default nwchem_prune",
            "scf_conv_tol_hartree": 1e-9, "scf_conv_tol_grad": 1e-6,
            "scf_max_cycle": 200, "max_newton_retries": 1,
            "geometry": "Fixed input XYZ; no geometry optimization",
            "solvent": "none (gas phase)", "dispersion": "none",
            "populations": ["Mulliken", "meta-Lowdin (ANO pre-orthogonalization)"],
            "f_minus_definition": "q(cation +1 doublet)-q(neutral singlet), same coordinates",
            "atom_indices": "0-based, identical to input XYZ row order",
            "internal_stability_requested": not args.skip_stability,
            "stability_diagnostic": "Native orbital Hessian with convergence-checked Davidson, subspace 60",
            "stability_library_sha256": sha(strict_stability_lib.__file__),
            "stability_native_equivalent_method": {
                "hessian": "PySCF native gen_g_hop_rhf/gen_g_hop_uhf, with_symmetry=True",
                "nroots": 3, "davidson_tolerance": 1e-4, "residual_tolerance": 0.01,
                "negative_eigenvalue_threshold": -1e-5, "max_cycle": 50,
                "max_space_argument": 60, "native_max_space_argument": 12,
                "initial_guess": args.strict_guess,
                "molecular_scf_parameters_unchanged": True,
            },
            "worker_revision": "resume-neutral-scf-strict-stability-v4",
            "neutral_stability_cache": str(args.reuse_neutral_stability.resolve()) if args.reuse_neutral_stability else None,
            "stability_cache_helper_sha256": sha(stability_cache_lib.__file__),
            "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
            "resume_helper_sha256": sha(resume_scf_lib.__file__),
            "neutral_scf_reuse_requested": bool(args.resume_from),
            "external_stability": "not tested",
            "pyscf_version": pyscf.__version__, "numpy_version": np.__version__,
            "libxc_version": getattr(dft.libxc, "__version__", None),
            "python_version": sys.version, "platform": platform.platform(),
            "input_sha256": status["xyz_sha256"], "worker_sha256": status["worker_sha256"],
        }
        atomic_json(outdir / "protocol.json", protocol)
        atomic_json(outdir / "atoms.json", [{"index": i, "element": symbol, "xyz_angstrom": xyz}
                    for i, (symbol, xyz) in enumerate(atoms)])
        states = {}
        neutral_orbitals = None
        for label, charge, spin in (("neutral", 0, 0), ("cation", 1, 1)):
            stage_t0 = time.monotonic()
            stage = outdir / label
            stage.mkdir()
            status.update(stage=label, updated_at=utc(), elapsed_seconds=time.monotonic()-t0)
            atomic_json(outdir / "status.json", status)
            mol = gto.M(atom=atoms, unit="Angstrom", basis=protocol["orbital_basis"],
                        charge=charge, spin=spin, symmetry=False, cart=False,
                        max_memory=args.memory_mb, verbose=4, output=str(stage / "scf.log"))
            mf = (dft.RKS(mol) if spin == 0 else dft.UKS(mol)).density_fit(auxbasis=protocol["auxiliary_basis"])
            mf.xc = protocol["functional"]
            mf.grids.level = protocol["grid_level"]
            mf.conv_tol = protocol["scf_conv_tol_hartree"]
            mf.conv_tol_grad = protocol["scf_conv_tol_grad"]
            mf.max_cycle = protocol["scf_max_cycle"]
            mf.max_memory = args.memory_mb
            mf.chkfile = str(stage / "attempt_1.chk")
            history_path = stage / "iterations.jsonl"
            attempt = 1

            def callback(env):
                record = {"at": utc(), "attempt": attempt, "elapsed_seconds": time.monotonic()-stage_t0}
                for key in ("cycle", "e_tot", "norm_gorb", "norm_ddm"):
                    val = env.get(key)
                    if val is not None and np.isscalar(val) and np.isfinite(val):
                        record[key] = int(val) if key == "cycle" else float(val)
                with open(history_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            mf.callback = callback
            dm0 = None
            if neutral_orbitals is not None:
                # Remove one beta electron from the neutral HOMO; both charge
                # states retain exactly the same atom order and AO basis.
                coeff, neutral_occ, neutral_energy = neutral_orbitals
                occ_a = (neutral_occ > 0).astype(float)
                occ_b = occ_a.copy()
                homo = np.where(occ_a > 0)[0][np.argmax(neutral_energy[occ_a > 0])]
                occ_b[homo] = 0.0
                dm0 = np.asarray([(coeff * occ_a) @ coeff.T, (coeff * occ_b) @ coeff.T])
            reused_scf = label == "neutral" and args.resume_from is not None
            if reused_scf:
                source_attempts, resume_proof = resume_scf_lib.restore_neutral_scf(
                    mf, args.resume_from, stage, status["xyz_sha256"])
                atomic_json(stage / "resume_provenance.json", resume_proof)
                attempts = [{"attempt": 0, "strategy": "Reuse converged archived neutral molecular SCF",
                             "converged": True, "energy_hartree": float(mf.e_tot),
                             "checkpoint": None, "molecular_kernel_called": False,
                             "source_attempts": source_attempts}]
            else:
                mf.kernel(dm0=dm0)
                attempts = [{"attempt": 1, "strategy": "DIIS", "converged": bool(mf.converged),
                             "energy_hartree": float(mf.e_tot), "checkpoint": "attempt_1.chk"}]
            if not mf.converged:
                attempt = 2
                dm_restart = mf.make_rdm1()
                mf = mf.newton()
                mf.chkfile = str(stage / "attempt_2_newton.chk")
                mf.max_cycle = 100
                mf.callback = callback
                mf.kernel(dm0=dm_restart)
                attempts.append({"attempt": 2, "strategy": "Newton continuation",
                                 "converged": bool(mf.converged), "energy_hartree": float(mf.e_tot),
                                 "checkpoint": "attempt_2_newton.chk"})
            atomic_json(stage / "attempts.json", attempts)
            dm = np.asarray(mf.make_rdm1())
            dm_total = dm if dm.ndim == 2 else dm.sum(axis=0)
            overlap = mf.get_ovlp()
            electron_count = float(np.einsum("ij,ji->", dm_total, overlap).real)
            _, q_mulliken = scf.hf.mulliken_pop(mol, dm_total, s=overlap, verbose=0)
            _, q_meta = scf.hf.mulliken_meta(mol, dm_total, s=overlap, pre_orth_method="ANO", verbose=0)
            np.savez_compressed(stage / "wavefunction.npz", density=dm, total_density=dm_total,
                                overlap=overlap, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ,
                                mo_energy_hartree=mf.mo_energy, coords_bohr=mol.atom_coords(),
                                coords_angstrom=mol.atom_coords(unit="Angstrom"), nuclear_charges=mol.atom_charges(),
                                mulliken_charges=q_mulliken, meta_lowdin_charges=q_meta)
            (stage / "molecule.json").write_text(mol.dumps(), encoding="utf-8")
            orbital_channels = []
            energies = np.atleast_2d(mf.mo_energy)
            occupations = np.atleast_2d(mf.mo_occ)
            for channel, (energy, occ) in enumerate(zip(energies, occupations)):
                homo_e = float(np.max(energy[occ > 1e-8]))
                lumo_e = float(np.min(energy[occ < 1e-8]))
                orbital_channels.append({"channel": "restricted" if spin == 0 else ("alpha" if channel == 0 else "beta"),
                                         "homo_hartree": homo_e, "lumo_hartree": lumo_e,
                                         "gap_ev": (lumo_e-homo_e)*27.211386245988})
            spin_square, spin_multiplicity = (0.0, 1.0) if spin == 0 else tuple(float(v) for v in mf.spin_square())
            spin_atoms = np.zeros(mol.natm)
            if spin:
                spin_ao = np.einsum("ij,ji->i", dm[0]-dm[1], overlap).real
                for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
                    spin_atoms[ia] = spin_ao[p0:p1].sum()
            stability = {"tested": False, "internal_stable": None, "external_tested": False,
                         "numerically_converged": None, "implementation": "strict_internal_stability"}
            if mf.converged and not args.skip_stability:
                status.update(stage=label+"_internal_stability", updated_at=utc())
                atomic_json(outdir / "status.json", status)
                cached_stability = label == "neutral" and args.reuse_neutral_stability is not None
                if cached_stability:
                    diagnostic, cache_proof = stability_cache_lib.reuse_neutral_stability(
                        mf, args.reuse_neutral_stability, args.resume_from, stage / "stability", resume_proof)
                    atomic_json(stage / "stability_reuse_provenance.json", cache_proof)
                    stability.update(implementation="reused_verified_standalone_cache",
                                     reuse_provenance="stability_reuse_provenance.json",
                                     neutral_hessian_recomputed=False,
                                     cache_helper_sha256=sha(stability_cache_lib.__file__))
                else:
                    diagnostic = strict_stability_lib.strict_internal_stability(
                        mf, stage / "stability", max_space=60, nroots=3, tol=1e-4, max_cycle=50,
                        initial_guess=args.strict_guess)
                    stability.update(library_sha256=sha(strict_stability_lib.__file__))
                stability.update(tested=True, internal_stable=diagnostic["internal_stable"],
                                 numerically_converged=diagnostic["numerically_converged"],
                                 passed=diagnostic["passed"], eigenvalues=diagnostic["eigenvalues"],
                                 davidson_converged=diagnostic["davidson_converged"],
                                 independent_residual_norms=diagnostic["independent_residual_norms"],
                                 residual_tolerance=diagnostic["residual_tolerance"],
                                 diagnostic_result="stability/result.json",
                                 diagnostic_result_sha256=sha(stage / "stability/result.json"))
            checks = {"scf_converged": bool(mf.converged),
                      "finite_energy_and_density": bool(np.isfinite(mf.e_tot) and np.isfinite(dm).all()),
                      "electron_count": abs(electron_count-mol.nelectron) < 1e-6,
                      "mulliken_charge_sum": abs(float(q_mulliken.sum())-charge) < 1e-6,
                      "meta_lowdin_charge_sum": abs(float(q_meta.sum())-charge) < 1e-6,
                      "spin_electron_count": abs(float(spin_atoms.sum())-spin) < 1e-6,
                      "internal_stability": (stability["internal_stable"] is True if stability["tested"] else True),
                      "stability_numerical_convergence": (stability["numerically_converged"] is True if stability["tested"] else True)}
            warnings = []
            if spin and abs(spin_square-0.75) > 0.10:
                warnings.append("UKS <S^2> differs from doublet reference 0.75 by >0.10; review spin contamination")
            if not stability["tested"]:
                warnings.append("Internal SCF stability was not tested")
            record = {"state": label, "charge": charge, "pyscf_spin_nalpha_minus_nbeta": spin,
                      "molecular_scf_recomputed": not reused_scf,
                      "resume_provenance": "resume_provenance.json" if reused_scf else None,
                      "multiplicity_input": spin+1, "nelectron": mol.nelectron,
                      "integrated_electrons_ao": electron_count, "nalpha_nbeta": list(mol.nelec),
                      "energy_hartree": float(mf.e_tot), "nao": mol.nao_nr(),
                      "natoms": mol.natm, "grid_points": len(mf.grids.weights) if mf.grids.weights is not None else None,
                      "grid_points_note": "Grid not reconstructed during archived SCF/stability reuse" if mf.grids.weights is None else None,
                      "converged": bool(mf.converged), "attempts": attempts,
                      "mulliken_charges": q_mulliken.tolist(), "meta_lowdin_charges": q_meta.tolist(),
                      "mulliken_charge_sum": float(q_mulliken.sum()), "meta_lowdin_charge_sum": float(q_meta.sum()),
                      "spin_square": spin_square, "spin_effective_multiplicity": spin_multiplicity,
                      "mulliken_atomic_spin_populations": spin_atoms.tolist(), "orbitals": orbital_channels,
                      "stability": stability, "checks": checks, "passed": all(checks.values()), "warnings": warnings,
                      "elapsed_seconds": time.monotonic()-stage_t0, "peak_process_rss_mb": peak_rss_mb(),
                      "geometry_sha256": status["xyz_sha256"], "completed_at": utc()}
            atomic_json(stage / "result.json", record)
            mol.stdout.flush()
            mol.stdout.close()
            states[label] = record
            if not record["passed"]:
                raise RuntimeError(f"{label} quality checks failed: {checks}")
            if spin == 0:
                neutral_orbitals = (np.asarray(mf.mo_coeff).copy(), np.asarray(mf.mo_occ).copy(),
                                    np.asarray(mf.mo_energy).copy())
        fminus = {}
        for key in ("mulliken_charges", "meta_lowdin_charges"):
            values = np.asarray(states["cation"][key])-np.asarray(states["neutral"][key])
            if abs(float(values.sum())-1.0) > 1e-6:
                raise RuntimeError("Condensed Fukui sum inconsistent with removal of one electron")
            fminus[key.replace("_charges", "")] = {"values": values.tolist(), "sum": float(values.sum())}
        combined = {"status": "complete", "passed": True, "geometry_sha256": status["xyz_sha256"], "atom_count": len(atoms),
                    "fixed_geometry": True, "electronic_singlepoints": 2,
                    "vertical_ionization_energy_ev": (states["cation"]["energy_hartree"]-states["neutral"]["energy_hartree"])*27.211386245988,
                    "f_minus": fminus, "states": states, "elapsed_seconds": time.monotonic()-t0,
                    "peak_process_rss_mb": peak_rss_mb(), "completed_at": utc()}
        atomic_json(outdir / "result.json", combined)
        artifacts = {str(p.relative_to(outdir)): {"size_bytes": p.stat().st_size, "sha256": sha(p)}
                     for p in outdir.rglob("*") if p.is_file() and "scratch" not in p.relative_to(outdir).parts
                     and str(p.relative_to(outdir)) not in {"status.json", "artifact_manifest.json", "launcher.log", "execution_receipt.json"}}
        atomic_json(outdir / "artifact_manifest.json", artifacts)
        status.update(state="complete", stage="complete", completed_at=utc(),
                      elapsed_seconds=time.monotonic()-t0, peak_process_rss_mb=peak_rss_mb())
        atomic_json(outdir / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", failed_at=utc(), error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic()-t0)
        atomic_json(outdir / "status.json", status)
        (outdir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
