"""Convergence-checked native orbital-Hessian analysis of a live converged SCF.

This function never calls a molecular SCF kernel. It uses the same PySCF
internal-stability Hessian, initial vector, preconditioner, roots and tolerance
as the standalone saved-density diagnostic; only the Davidson subspace differs
from the original native default, and convergence evidence is retained.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write(path, data):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_internal_stability(mf, outdir, max_space=60, nroots=3, tol=1e-4, max_cycle=50,
                              initial_guess='native_dense'):
    """Return the final numerical-convergence and stability record, with files.

    A nonconverged eigensolver produces ``internal_stable=None`` and
    ``passed=False``. Genuine converged negative curvature also returns
    ``passed=False``. Exceptions preserve a failed diagnostic status and raise.
    """
    import numpy as np
    import pyscf
    from pyscf import lib
    from pyscf.scf import stability
    from pyscf.soscf import newton_ah

    if max_space < 12 or nroots != 3 or tol != 1e-4 or max_cycle != 50:
        raise ValueError('Preserve three native roots, tol=1e-4, max_cycle=50 and max_space>=12')
    if initial_guess not in ('native_dense', 'native_plus_lowdiag'):
        raise ValueError('Unknown stability initial-guess strategy')
    if not mf.converged:
        raise ValueError('Strict stability requires an already converged molecular SCF')
    if (mf.mol.charge, mf.mol.spin) not in ((0, 0), (1, 1)):
        raise ValueError('Unexpected charge/spin for this electronic-validation protocol')
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(state='running', stage='initialize', pid=os.getpid(), started_at=_utc(),
                 library_sha256=_sha(__file__))
    _write(out / 'status.json', state)

    def fingerprint(array):
        a = np.asarray(array)
        h = hashlib.sha256()
        h.update(a.dtype.str.encode())
        h.update(repr(a.shape).encode())
        h.update(a.tobytes(order='C'))
        return h.hexdigest()

    def source_snapshot():
        return dict(energy_hartree=float(mf.e_tot), scf_converged=bool(mf.converged),
                    mo_coeff_sha256=fingerprint(mf.mo_coeff), mo_occ_sha256=fingerprint(mf.mo_occ),
                    mo_energy_sha256=fingerprint(mf.mo_energy), density_sha256=fingerprint(mf.make_rdm1()),
                    coordinates_bohr_sha256=fingerprint(mf.mol.atom_coords()),
                    nuclear_charges_sha256=fingerprint(mf.mol.atom_charges()))

    try:
        before = source_snapshot()
        _write(out / 'source_live_mf.json', before)
        spin = mf.mol.spin
        native = stability.rhf_internal if spin == 0 else stability.uhf_internal
        factory = newton_ah.gen_g_hop_rhf if spin == 0 else newton_ah.gen_g_hop_uhf
        residual_tol = tol**0.5
        standalone = Path(__file__).with_name('strict_stability.py')
        protocol = dict(
            implementation='strict_internal_stability live-mf v3 explicit guess', pyscf_version=pyscf.__version__,
            molecular_scf_recomputed=False, functional=mf.xc, orbital_basis=mf.mol.basis,
            auxiliary_basis=mf.with_df.auxbasis, grid_level=mf.grids.level,
            charge=mf.mol.charge, spin=spin, with_symmetry=True,
            nroots=nroots, davidson_tolerance=tol, residual_tolerance=residual_tol,
            stability_negative_threshold=-1e-5, max_cycle=max_cycle, max_space_argument=max_space,
            effective_max_space=max_space + (nroots - 1) * 4,
            native_max_space_argument=12, native_effective_max_space=20, davidson_memory_mb=4000,
            initial_guess=initial_guess,
            initial_guess_definition=('native x0[g!=0]=1/hdiag[g!=0]' if initial_guess == 'native_dense'
                                      else 'native dense x0 plus unit vectors at three lowest hdiag entries'),
            preconditioner='native diagonal minus Ritz eigenvalue; abs denominator <1e-8 replaced by +1e-8',
            hessian_scaling='native hdiag*=2; hessian_x=hop(x).real*2',
            hessian_factory=factory.__module__ + '.' + factory.__name__,
            native_stability_source_sha256=hashlib.sha256(inspect.getsource(native).encode()).hexdigest(),
            native_hessian_factory_source_sha256=hashlib.sha256(inspect.getsource(factory).encode()).hexdigest(),
            davidson1_source_sha256=hashlib.sha256(inspect.getsource(lib.davidson1).encode()).hexdigest(),
            library_sha256=_sha(__file__),
            standalone_reference_script_sha256=_sha(standalone) if standalone.exists() else None,
            threads=lib.num_threads(), memory_mb=mf.max_memory)
        _write(out / 'protocol.json', protocol)
        with (out / 'strict_stability.log').open('w', encoding='utf-8') as log:
            state.update(stage='build_native_hessian', updated_at=_utc())
            _write(out / 'status.json', state)
            g, hop, hdiag = factory(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=True)
            hdiag *= 2
            hessian_calls = 0
            last = {}
            seed_count = 4 if initial_guess == 'native_plus_lowdiag' else 1
            initial_trial_rayleigh = []

            def precond(dx, eigenvalue, x0):
                denominator = hdiag - eigenvalue
                denominator[abs(denominator) < 1e-8] = 1e-8
                return dx / denominator

            def hessian_x(x):
                nonlocal hessian_calls
                value = hop(x).real * 2
                hessian_calls += 1
                if hessian_calls <= seed_count:
                    initial_trial_rayleigh.append(float(np.dot(x, value) / np.dot(x, x)))
                return value

            native_x0 = np.zeros_like(g)
            native_x0[g != 0] = 1. / hdiag[g != 0]
            if not np.isfinite(native_x0).all() or np.linalg.norm(native_x0) == 0:
                raise ValueError('Native stability guess is degenerate')
            if not np.isfinite(hdiag).all():
                raise ValueError('Nonfinite preconditioner diagonal')
            seed_indices = np.array([], dtype=int)
            x0 = native_x0
            if initial_guess == 'native_plus_lowdiag':
                seed_indices = np.argsort(hdiag, kind='stable')[:nroots]
                unit_seeds = np.zeros((len(seed_indices), len(g)))
                unit_seeds[np.arange(len(seed_indices)), seed_indices] = 1.
                x0 = [native_x0] + [x for x in unit_seeds]
            protocol.update(seed_indices_0based=seed_indices.tolist(),
                            preconditioner_diagonal_at_unit_seeds=hdiag[seed_indices].tolist(),
                            native_dense_seed_norm=float(np.linalg.norm(native_x0)),
                            initial_seed_count=seed_count, native_dense_seed_retained=True,
                            rayleigh_definition='x dot Hx / x dot x for first orthonormalized Davidson trial vectors')
            _write(out / 'protocol.json', protocol)

            def callback(env):
                row = dict(cycle=int(env['icyc']), subspace=int(env['space']),
                           elapsed_seconds=time.monotonic() - started, hessian_calls=hessian_calls,
                           eigenvalues=np.asarray(env['e']).tolist(),
                           residual_norms=np.asarray(env['dx_norm']).tolist(),
                           converged=np.asarray(env['conv'], dtype=bool).tolist())
                last.clear()
                last.update(row)
                with (out / 'iterations.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                state.update(stage='davidson', updated_at=_utc(), **row)
                _write(out / 'status.json', state)

            state.update(stage='davidson', updated_at=_utc())
            _write(out / 'status.json', state)
            converged, eigenvalues, eigenvectors = lib.davidson1(
                lambda xs: [hessian_x(x) for x in xs], x0, precond,
                tol=tol, max_cycle=max_cycle, max_space=max_space, max_memory=4000, nroots=nroots,
                verbose=lib.logger.Logger(log, 5), callback=callback)
            state.update(stage='independent_eigen_residuals', updated_at=_utc())
            _write(out / 'status.json', state)
            residuals = [float(np.linalg.norm(hessian_x(v) - e * v))
                         for e, v in zip(eigenvalues, eigenvectors)]
            np.savez_compressed(out / 'eigenpairs.npz', eigenvalues=np.asarray(eigenvalues),
                                eigenvectors=np.asarray(eigenvectors), residual_norms=np.asarray(residuals),
                                converged=np.asarray(converged, dtype=bool), hdiag=hdiag, initial_gradient=g)
            unchanged = source_snapshot() == before
            checks = dict(source_scf_converged=True, all_three_roots_returned=len(eigenvalues) == nroots,
                          davidson_all_roots_converged=bool(np.all(converged)),
                          independent_residuals_passed=bool(all(r < residual_tol for r in residuals)),
                          finite_eigenvalues_and_residuals=bool(np.isfinite(eigenvalues).all() and np.isfinite(residuals).all()),
                          live_source_molecular_state_unchanged=unchanged)
            numerical = all(checks.values())
            stable = bool(np.min(eigenvalues) >= -1e-5) if numerical else None
            result = dict(status='complete' if numerical and stable else 'needs_attention',
                          passed=bool(numerical and stable), numerically_converged=numerical,
                          internal_stable=stable, checks=checks, eigenvalues=np.asarray(eigenvalues).tolist(),
                          davidson_converged=np.asarray(converged, dtype=bool).tolist(),
                          independent_residual_norms=residuals, residual_tolerance=residual_tol,
                          hessian_calls=hessian_calls, elapsed_seconds=time.monotonic() - started,
                          initial_guess=initial_guess, seed_indices_0based=seed_indices.tolist(),
                          initial_orthonormal_trial_rayleigh_quotients=initial_trial_rayleigh,
                          last_nonterminal_iteration=last, source_scf_energy_hartree=before['energy_hartree'],
                          molecular_scf_recomputed=False, library_sha256=protocol['library_sha256'], completed_at=_utc())
            log.flush()
        _write(out / 'result.json', result)
        state.update(state=result['status'], stage='complete', completed_at=_utc(), elapsed_seconds=result['elapsed_seconds'])
        _write(out / 'status.json', state)
        manifest = {str(p.relative_to(out)): dict(sha256=_sha(p), size_bytes=p.stat().st_size)
                    for p in out.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}
        _write(out / 'artifact_manifest.json', manifest)
        return result
    except BaseException as exc:
        state.update(state='failed', failed_at=_utc(), error=f'{type(exc).__name__}: {exc}')
        _write(out / 'status.json', state)
        (out / 'failure_traceback.txt').write_text(traceback.format_exc())
        raise
