"""Restore a converged neutral molecular SCF, without invoking any SCF kernel."""
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_neutral_scf(mf, source_structure, destination_state, xyz_sha256):
    import numpy as np
    import pyscf
    from pyscf import gto

    source = Path(source_structure).resolve()
    destination = Path(destination_state).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Resume output must be outside the original structure directory')
    state = source / 'neutral'
    required = ['protocol.json', 'neutral/molecule.json', 'neutral/wavefunction.npz', 'neutral/attempts.json']
    bindings = {name: sha(source / name) for name in required}
    protocol = json.loads((source / 'protocol.json').read_text())
    expected = dict(functional='PBE0', orbital_basis='def2-tzvp', auxiliary_basis='def2-universal-jkfit',
                    grid_level=4, scf_conv_tol_hartree=1e-9, scf_conv_tol_grad=1e-6,
                    scf_max_cycle=200, max_newton_retries=1, solvent='none (gas phase)', dispersion='none')
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f'Source SCF method mismatch: {key}')
    if protocol.get('input_sha256') != xyz_sha256 or protocol.get('pyscf_version') != pyscf.__version__:
        raise ValueError('Source XYZ hash or PySCF version differs')
    if mf.xc != expected['functional'] or mf.mol.basis != expected['orbital_basis']:
        raise ValueError('Destination SCF functional/basis differs')
    if mf.with_df.auxbasis != expected['auxiliary_basis'] or mf.grids.level != expected['grid_level']:
        raise ValueError('Destination density-fitting/grid protocol differs')
    if mf.conv_tol != expected['scf_conv_tol_hartree'] or mf.conv_tol_grad != expected['scf_conv_tol_grad']:
        raise ValueError('Destination SCF tolerances differ')
    if (mf.mol.charge, mf.mol.spin) != (0, 0):
        raise ValueError('Only neutral singlet SCF reuse is supported')
    source_mol = gto.loads((state / 'molecule.json').read_text())
    if source_mol._ecpbas.size == 0:
        source_mol._ecpbas = np.zeros((0, gto.mole.BAS_SLOTS), dtype=np.int32)
    if (source_mol.charge, source_mol.spin) != (0, 0):
        raise ValueError('Source is not a neutral singlet')
    if not np.array_equal(source_mol.atom_charges(), mf.mol.atom_charges()):
        raise ValueError('Source/destination element sequence differs')
    if not np.allclose(source_mol.atom_coords(), mf.mol.atom_coords(), atol=1e-12, rtol=0):
        raise ValueError('Source/destination coordinates differ')
    if source_mol.cart != mf.mol.cart or source_mol.ao_labels() != mf.mol.ao_labels():
        raise ValueError('Source/destination AO convention or labels differ')
    if source_mol.nbas != mf.mol.nbas:
        raise ValueError('Source/destination shell count differs')
    # _bas contains pointers into _env. Element-dependent storage order can
    # change between Python processes without changing a single basis function.
    # Verify the physical shell definitions and later the numerical overlap.
    shell_checks = []
    for shell in range(source_mol.nbas):
        fields = ('bas_atom', 'bas_angular', 'bas_nprim', 'bas_nctr', 'bas_kappa')
        metadata = {field: int(getattr(source_mol, field)(shell)) for field in fields}
        if any(getattr(mf.mol, field)(shell) != value for field, value in metadata.items()):
            raise ValueError(f'Source/destination shell metadata differs: {shell}')
        exp_a, exp_b = source_mol.bas_exp(shell), mf.mol.bas_exp(shell)
        coef_a, coef_b = source_mol.bas_ctr_coeff(shell), mf.mol.bas_ctr_coeff(shell)
        if exp_a.shape != exp_b.shape or not np.allclose(exp_a, exp_b, atol=1e-14, rtol=0):
            raise ValueError(f'Source/destination Gaussian exponents differ: {shell}')
        if coef_a.shape != coef_b.shape or not np.allclose(coef_a, coef_b, atol=1e-14, rtol=0):
            raise ValueError(f'Source/destination contraction coefficients differ: {shell}')
        shell_checks.append(dict(shell=shell, metadata=metadata,
                                 max_exponent_difference=float(np.max(np.abs(exp_a-exp_b))),
                                 max_coefficient_difference=float(np.max(np.abs(coef_a-coef_b)))))
    raw_basis_tables_equal = bool(np.array_equal(source_mol._bas, mf.mol._bas))
    if not np.array_equal(source_mol._ecpbas, mf.mol._ecpbas):
        raise ValueError('Source/destination ECP specification differs')
    with np.load(state / 'wavefunction.npz', allow_pickle=False) as archive:
        values = {k: np.array(archive[k]) for k in ['density', 'total_density', 'overlap', 'mo_coeff', 'mo_occ',
                  'mo_energy_hartree', 'coords_bohr', 'nuclear_charges', 'mulliken_charges', 'meta_lowdin_charges']}
    if not all(np.isfinite(a).all() for a in values.values()):
        raise ValueError('Source wavefunction contains nonfinite values')
    if not np.array_equal(values['nuclear_charges'], mf.mol.atom_charges()) or not np.allclose(
            values['coords_bohr'], mf.mol.atom_coords(), atol=1e-12, rtol=0):
        raise ValueError('Saved wavefunction atom sequence or coordinates differ')
    nao = mf.mol.nao_nr()
    if values['density'].shape != (nao, nao) or values['mo_coeff'].shape != (nao, nao):
        raise ValueError('Source restricted density or MO coefficient dimensions differ')
    if values['mo_occ'].shape != (nao,) or values['mo_energy_hartree'].shape != (nao,):
        raise ValueError('Source occupation or orbital energy dimensions differ')
    if not np.all(np.isin(values['mo_occ'], [0., 2.])):
        raise ValueError('Source is not an integer-occupation restricted singlet')
    if not np.allclose(values['density'], values['total_density'], atol=1e-12, rtol=0):
        raise ValueError('Source total density differs from restricted density')
    overlap = mf.mol.intor_symmetric('int1e_ovlp')
    if not np.allclose(overlap, values['overlap'], atol=1e-10, rtol=0):
        raise ValueError('Source/destination AO overlap differs')
    source_overlap = source_mol.intor_symmetric('int1e_ovlp')
    if not np.allclose(source_overlap, overlap, atol=1e-10, rtol=0):
        raise ValueError('Serialized source basis does not reproduce saved overlap')
    electron_count = float(np.einsum('ij,ji->', values['density'], overlap).real)
    if abs(electron_count - mf.mol.nelectron) > 1e-6 or abs(float(values['mo_occ'].sum()) - mf.mol.nelectron) > 1e-6:
        raise ValueError('Source density or occupations have incorrect electron count')
    orbital_metric = values['mo_coeff'].T.conj() @ overlap @ values['mo_coeff']
    if not np.allclose(orbital_metric, np.eye(nao), atol=1e-8, rtol=0):
        raise ValueError('Source orbitals are not orthonormal in this AO basis')
    attempts = json.loads((state / 'attempts.json').read_text())
    if not attempts or attempts[-1].get('converged') is not True:
        raise ValueError('Source attempts do not establish molecular SCF convergence')
    energy = float(attempts[-1]['energy_hartree'])
    if not np.isfinite(energy):
        raise ValueError('Nonfinite source SCF energy')
    # The original stability computation may append to this log. Bind the exact
    # copied snapshot, while requiring the wavefunction/protocol files immutable.
    log_bytes = (state / 'scf.log').read_bytes()
    matches = re.findall(r'converged SCF energy\s*=\s*([-+0-9.eE]+)', log_bytes.decode(errors='replace'))
    if not matches or abs(float(matches[-1]) - energy) > 1e-8:
        raise ValueError('Source log and attempts disagree on converged SCF energy')
    (destination / 'original_scf.log').write_bytes(log_bytes)
    (destination / 'original_attempts.json').write_bytes((state / 'attempts.json').read_bytes())
    if (state / 'iterations.jsonl').exists():
        iteration_bytes = (state / 'iterations.jsonl').read_bytes()
        (destination / 'original_iterations.jsonl').write_bytes(iteration_bytes)
    optional = state / 'result.json'
    if optional.exists():
        original_result = json.loads(optional.read_text())
        if not original_result.get('converged') or abs(original_result['energy_hartree'] - energy) > 1e-10:
            raise ValueError('Existing source state result disagrees with converged SCF')
    mf.mo_coeff = values['mo_coeff']
    mf.mo_occ = values['mo_occ']
    mf.mo_energy = values['mo_energy_hartree']
    mf.e_tot = energy
    mf.converged = True
    restored_density = np.asarray(mf.make_rdm1())
    if not np.allclose(restored_density, values['density'], atol=1e-10, rtol=0):
        raise ValueError('Source orbitals do not reconstruct saved density')
    # Retain the exact archived density, rather than only the recomputed form,
    # for the caller's independent population comparison.
    from pyscf import scf
    _, q_mulliken = scf.hf.mulliken_pop(mf.mol, restored_density, s=overlap, verbose=0)
    _, q_meta = scf.hf.mulliken_meta(mf.mol, restored_density, s=overlap, pre_orth_method='ANO', verbose=0)
    deltas = dict(mulliken=float(np.max(np.abs(q_mulliken - values['mulliken_charges']))),
                  meta_lowdin=float(np.max(np.abs(q_meta - values['meta_lowdin_charges']))))
    if max(deltas.values()) > 1e-8:
        raise ValueError('Restored-density populations disagree with source')
    if any(sha(source / path) != digest for path, digest in bindings.items()):
        raise ValueError('Immutable source changed during SCF restoration')
    mf.stdout.write('\nRESUMED CONVERGED NEUTRAL SCF: molecular kernel not called.\n')
    mf.stdout.write(f'Source structure: {source}\nExact archived energy = {energy!r}\n')
    mf.stdout.flush()
    proof = dict(source_structure=str(source), source_state='neutral', destination_state=str(destination),
                 molecular_scf_recomputed=False, source_scf_converged=True,
                 source_energy_hartree=energy, source_raw_sha256=bindings,
                 source_scf_log_snapshot_sha256=hashlib.sha256(log_bytes).hexdigest(),
                 source_log_may_continue_appending=True, source_attempts=attempts,
                 xyz_sha256=xyz_sha256, source_protocol=expected, pyscf_version=pyscf.__version__,
                 verified_electrons=electron_count, verified_population_max_abs_change=deltas,
                 basis_validation='Physical per-shell metadata, exponents, coefficients, AO labels, cart convention and overlap',
                 raw_basis_tables_equal=raw_basis_tables_equal,
                 raw_basis_pointer_columns_source=source_mol._bas[:, [gto.mole.PTR_EXP, gto.mole.PTR_COEFF]].tolist(),
                 raw_basis_pointer_columns_destination=mf.mol._bas[:, [gto.mole.PTR_EXP, gto.mole.PTR_COEFF]].tolist(),
                 shell_validation=shell_checks,
                 helper_sha256=sha(__file__), checks_passed=True)
    return attempts, proof
