# Electronic calculations and supplementary data



- geometries/: 17 unchanged GFN2-xTB-optimized Cartesian inputs, in angstroms.
- structure_manifest.json: chemical identities, mapped SMILES, atom mapping, bond records, target atoms and geometry SHA256 values.
- dft/: completed fixed-geometry PBE0/def2-TZVP neutral/cation state results, SCF logs, iterations and internal-stability diagnostics.
- hirshfeld/: original level-4 and level-5 integrations and the selected level-5 charges.
- dft_all_atom_populations.csv: neutral/cation charges and f_minus for every atom, under Hirshfeld, meta-Lowdin and Mulliken partitions; f_minus = q_cation - q_neutral.
- xtb_all_atom_q_fminus.csv: original GFN2-xTB all-atom neutral charges and f_minus; no DFT partition is assigned to these values.
- The input manifest retains corrected terminal orbital values for the 12 parent/target structures; five alternative-site values are null. No orbital-energy table is used in this supplement, and earlier top-level orbital fields in the original xTB extraction JSON are not used.
- dft_34_state_summary.csv and structure_inventory.csv: full-precision electronic-state and structural inventories.
- [calculation_code/](calculation_code/) contains the calculation and postprocessing scripts. The [electronic-state index](electronic_states.json) links the original result file and its checksum for each of the 34 states.
- original_*_protocol.json and software_requirements.lock: original method settings and software records.

State multiplicities are 1 for neutral charge 0 and 2 for vertical charge +1. The two states of a structure use identical coordinates. All IDs refer to archived test-case identifiers. Alternative N numbers are atom-map numbers. Atomic index columns explicitly distinguish zero-based indices and one-based atom numbers; hydrogen atom-map values may be zero.

The manuscript uses bond-direction-averaged model logits. Population partitions are applied to the same electronic states and are not independent molecular repetitions. All electronic and alternative-site contrasts are retained.
