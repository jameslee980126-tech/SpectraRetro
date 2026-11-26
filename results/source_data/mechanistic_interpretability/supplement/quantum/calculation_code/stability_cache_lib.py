"""Reuse a fully validated standalone neutral Hessian diagnostic without recomputation."""
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath


STANDALONE_SHA256 = 'fd821c6d8d6350b6a2d39604f64efa6aca86273b35a11af9241bd84c2408d6aa'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reuse_neutral_stability(mf, cache_directory, source_structure, destination, resume_proof):
    import numpy as np
    import pyscf
    from pyscf.scf import stability
    from pyscf.soscf import newton_ah

    cache, source, dest = (Path(p).resolve() for p in (cache_directory, source_structure, destination))
    if dest.exists() or dest == cache or cache in dest.parents or source in dest.parents:
        raise ValueError('Cached stability destination must be new and outside either source')
    if (mf.mol.charge, mf.mol.spin) != (0, 0) or not mf.converged:
        raise ValueError('Stability cache reuse requires a converged neutral singlet')
    if resume_proof.get('checks_passed') is not True or Path(resume_proof['source_structure']).resolve() != source:
        raise ValueError('Validated neutral SCF restoration is required before cache reuse')
    manifest_bytes = (cache / 'artifact_manifest.json').read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    required = {'result.json', 'protocol.json', 'source_binding.json', 'eigenpairs.npz',
                'status.json', 'source_scf_log_snapshot.txt', 'strict_stability.log', 'iterations.jsonl'}
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError('Stability cache manifest lacks required evidence')
    cached_bytes = {}
    for name, binding in manifest.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name or name == 'artifact_manifest.json':
            raise ValueError('Unsafe cache manifest relative path')
        path = cache / name
        if path.is_symlink() or not path.resolve().is_relative_to(cache):
            raise ValueError('Stability cache artifact escapes source directory')
        data = path.read_bytes()
        if len(data) != binding['size_bytes'] or hashlib.sha256(data).hexdigest() != binding['sha256']:
            raise ValueError(f'Stability cache artifact checksum failed: {name}')
        cached_bytes[name] = data
    result, protocol, binding, status = (json.loads(cached_bytes[name]) for name in
                                       ('result.json', 'protocol.json', 'source_binding.json', 'status.json'))
    if result.get('status') != 'complete' or status.get('state') != 'complete':
        raise ValueError('Stability cache did not complete successfully')
    if status.get('script_sha256') != STANDALONE_SHA256:
        raise ValueError('Stability cache producer differs from validated standalone implementation')
    if any(result.get(k) is not True for k in ('passed', 'numerically_converged', 'internal_stable')):
        raise ValueError('Stability cache has not passed numerical and physical stability gates')
    required_checks = ('source_density_verified', 'all_three_roots_returned', 'davidson_all_roots_converged',
                       'independent_residuals_passed', 'finite_eigenvalues_and_residuals', 'immutable_source_files_unchanged')
    if any(result.get('checks', {}).get(k) is not True for k in required_checks):
        raise ValueError('Stability cache has incomplete checks')
    if Path(result['state_dir']).resolve() != source / 'neutral' or result.get('molecular_scf_recomputed') is not False:
        raise ValueError('Stability cache is not bound to the requested original neutral state')
    expected = dict(pyscf_version=pyscf.__version__, molecular_scf_recomputed=False,
                    functional='PBE0', orbital_basis='def2-tzvp', auxiliary_basis='def2-universal-jkfit',
                    grid_level=4, charge=0, spin=0, nroots=3, davidson_tolerance=1e-4,
                    residual_tolerance=.01, stability_negative_threshold=-1e-5, max_cycle=50,
                    max_space_argument=60, effective_max_space=68, native_max_space_argument=12,
                    native_effective_max_space=20, davidson_memory_mb=4000, with_symmetry=True,
                    initial_guess='native dense x0 plus unit vectors at three lowest hdiag entries',
                    preconditioner='native diagonal minus Ritz eigenvalue; abs denominator <1e-8 replaced by +1e-8',
                    hessian_scaling='native hdiag*=2; hessian_x=hop(x).real*2',
                    initial_seed_count=4, native_dense_seed_retained=True,
                    hessian_factory=newton_ah.gen_g_hop_rhf.__module__+'.'+newton_ah.gen_g_hop_rhf.__name__,
                    native_stability_source_sha256=hashlib.sha256(inspect.getsource(stability.rhf_internal).encode()).hexdigest(),
                    native_hessian_factory_source_sha256=hashlib.sha256(inspect.getsource(newton_ah.gen_g_hop_rhf).encode()).hexdigest())
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f'Stability cache method mismatch: {key}')
    if mf.xc != 'PBE0' or mf.mol.basis != 'def2-tzvp' or mf.with_df.auxbasis != 'def2-universal-jkfit' or mf.grids.level != 4:
        raise ValueError('Live restored mean field differs from cached Hessian method')
    expected_paths = {str(source / p) for p in ('protocol.json', 'neutral/molecule.json', 'neutral/wavefunction.npz')}
    if set(binding.get('immutable_files', {})) != expected_paths:
        raise ValueError('Stability cache immutable source paths differ')
    for path, digest in binding['immutable_files'].items():
        rel = str(Path(path).relative_to(source))
        if sha(path) != digest or resume_proof['source_raw_sha256'].get(rel) != digest:
            raise ValueError(f'Stability cache source checksum mismatch: {rel}')
    if binding.get('source_log_snapshot_sha256') != manifest['source_scf_log_snapshot.txt']['sha256']:
        raise ValueError('Stability cache SCF log snapshot binding differs')
    for energy in (binding.get('source_scf_energy_hartree'), result.get('source_scf_energy_hartree')):
        if energy is None or not np.isfinite(energy) or abs(energy - mf.e_tot) > 1e-8:
            raise ValueError('Stability cache source SCF energy differs from restored exact energy')
    eigenvalues = np.asarray(result['eigenvalues'], dtype=float)
    residuals = np.asarray(result['independent_residual_norms'], dtype=float)
    if (eigenvalues.shape != (3,) or residuals.shape != (3,) or
            result.get('davidson_converged') != [True, True, True] or
            not np.isfinite(eigenvalues).all() or not np.isfinite(residuals).all() or
            np.min(eigenvalues) < -1e-5 or np.min(residuals) < 0 or np.max(residuals) >= .01 or
            result.get('residual_tolerance') != .01):
        raise ValueError('Cached eigenvalues, convergence flags or independent residuals failed')
    # Arrays are read from the already SHA-verified snapshot, never from a moving file.
    import io
    with np.load(io.BytesIO(cached_bytes['eigenpairs.npz']), allow_pickle=False) as archive:
        pairs = {k: np.array(archive[k]) for k in ('eigenvalues', 'eigenvectors', 'residual_norms', 'converged', 'hdiag', 'initial_gradient')}
    dim = int(np.count_nonzero(mf.mo_occ > 0) * np.count_nonzero(mf.mo_occ == 0))
    if pairs['eigenvectors'].shape != (3, dim) or pairs['hdiag'].shape != (dim,) or pairs['initial_gradient'].shape != (dim,):
        raise ValueError('Cached Hessian array dimensions differ from restored orbital space')
    if not all(np.isfinite(a).all() for a in pairs.values()):
        raise ValueError('Cached Hessian arrays contain nonfinite values')
    if (not np.array_equal(pairs['eigenvalues'], eigenvalues) or not np.array_equal(pairs['residual_norms'], residuals)
            or not np.array_equal(pairs['converged'], np.ones(3, dtype=bool))):
        raise ValueError('Cached eigenpair arrays disagree with final diagnostic result')
    if not np.allclose(pairs['eigenvectors'] @ pairs['eigenvectors'].T, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError('Cached eigenvectors are not orthonormal')
    seeds = np.argsort(pairs['hdiag'], kind='stable')[:3].tolist()
    if protocol.get('seed_indices_0based') != seeds or result.get('seed_indices_0based') != seeds:
        raise ValueError('Cached low-diagonal seed selection differs')
    if not np.array_equal(protocol['preconditioner_diagonal_at_unit_seeds'], pairs['hdiag'][seeds]):
        raise ValueError('Cached seed diagonal values differ')
    with np.load(source / 'neutral/wavefunction.npz', allow_pickle=False) as archive:
        for key, live in (('mo_coeff', mf.mo_coeff), ('mo_occ', mf.mo_occ), ('mo_energy_hartree', mf.mo_energy)):
            if not np.array_equal(archive[key], live):
                raise ValueError(f'Restored live orbitals differ from cached source: {key}')
        if not np.allclose(archive['density'], mf.make_rdm1(), atol=1e-10, rtol=0):
            raise ValueError('Restored live density differs from cached source')
    # Copy only the authenticated immutable snapshot, preserving every artifact byte.
    dest.mkdir(parents=True, exist_ok=False)
    for name, data in cached_bytes.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if sha(path) != manifest[name]['sha256']:
            raise IOError(f'Copied stability cache artifact checksum failed: {name}')
    (dest / 'artifact_manifest.json').write_bytes(manifest_bytes)
    if (sha(cache / 'artifact_manifest.json') != manifest_sha or
            any(sha(cache / name) != item['sha256'] for name, item in manifest.items()) or
            any(sha(path) != digest for path, digest in binding['immutable_files'].items())):
        raise ValueError('Stability cache or immutable molecular source changed during reuse')
    proof = dict(cache_directory=str(cache), destination=str(dest), source_structure=str(source),
                 cached_artifacts=len(manifest), cache_artifact_manifest_sha256=manifest_sha,
                 cache_result_sha256=manifest['result.json']['sha256'],
                 cache_protocol_sha256=manifest['protocol.json']['sha256'],
                 source_immutable_sha256=binding['immutable_files'],
                 cached_standalone_sha256=STANDALONE_SHA256, helper_sha256=sha(__file__),
                 cached_method=expected, source_energy_full_precision_hartree=float(mf.e_tot),
                 molecular_scf_recomputed=False, neutral_hessian_recomputed=False,
                 independent_residuals='Validated recorded independent residuals; not recomputed during cache reuse',
                 bytes_preserved=True, checks_passed=True)
    return result, proof
