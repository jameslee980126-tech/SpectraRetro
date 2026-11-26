# Mechanistic interpretability source data

Data accompanying main Figure 4 and Supporting Information Sections S4.6 and S7.

Main Figure 4 is organized as a (chemical-context associations), b(i)--(iii) (spatial intervention), and c(i)--(v) (electronic analysis). Panel c(i) identifies the representative structure, c(ii) shows GFN2-xTB electrostatic potentials, c(iii)/(iv) compare charge and electron-removal-response changes from GFN2-xTB and PBE0/def2-TZVP Hirshfeld populations, and c(v) reports the bond-direction-averaged model response. `main_figure_six_pairs.csv` retains both model readouts for the six primary pairs.

The `supplement/` directory supplies the full-precision source tables, molecule identities, 17 archived geometries, 34 electronic-state calculations, atom populations, calculation logs and method records supporting Figures S4–S6 and Tables S22--S27. `supplement/data_index.csv` and `.json` map files to displays; `supplement/manifest.json` inventories their byte counts and SHA-256 hashes.

The spatial analysis contains 100 selected orientations nested within 38 fragments and 22 retained-scaffold graph clusters. Reported intervals use the independently replayed `analysis_summary.json`. DFT uses the archived GFN2-xTB-optimized geometries. Source-level numerical precision and both original and bond-direction-averaged model readouts are preserved with their labels.
