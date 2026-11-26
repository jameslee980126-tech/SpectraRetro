"""Bounded chemistry tools for CHRetroAgent, separate from learned model weights.

Scientific scope is carried in tool results: graph checks are not reaction
feasibility predictions, model analogs are not literature evidence, and a stock
entry is not a current supplier quote. Network results retain their sources.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from itertools import islice, permutations
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors, rdChemReactions
from rdkit.Chem import rdFingerprintGenerator


def _now():
    return datetime.now(timezone.utc).isoformat()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _mol(smiles):
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 4000:
        raise ValueError("Provide a nonempty SMILES of at most 4000 characters")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or not mol.GetNumAtoms() or mol.GetNumAtoms() > 400:
        raise ValueError("Invalid SMILES or molecule exceeds the 400-atom tool limit")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("A concrete molecule is required; wildcard atoms are unsupported")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    # Atom indices must match the molecule drawn from the returned SMILES.
    return canonical, Chem.MolFromSmiles(canonical)


def _bounded_int(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    return value


GROUP_SMARTS = {
    "alcohol": "[OX2H][CX4]", "phenol": "[OX2H]c", "carboxylic_acid": "[CX3](=O)[OX2H]",
    "aldehyde": "[CX3H1](=O)[#6]", "ketone": "[#6][CX3](=O)[#6]",
    "ester": "[CX3](=O)[OX2][#6]", "amide": "[CX3](=O)[NX3]",
    "amine": "[NX3;!$(N[C,S,P]=O);!$(N=*)]", "aromatic_nitrogen": "[n]",
    "thiol": "[SX2H]", "sulfide": "[#6][SX2][#6]", "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]", "nitrile": "[CX2]#[NX1]", "nitro": "[N+](=O)[O-]",
    "aryl_halide": "c[F,Cl,Br,I]", "alkyl_halide": "[CX4][F,Cl,Br,I]",
    "boronic_acid": "[#5]([OX2H])[OX2H]", "peroxide": "[OX2][OX2]",
    "acetal": "[CX4]([OX2][#6])([OX2][#6])", "epoxide": "[O;r3]1[C;r3][C;r3]1",
}
GROUPS = {name: Chem.MolFromSmarts(pattern) for name, pattern in GROUP_SMARTS.items()}


def _groups(mol):
    return [{"name": name, "count": len(matches), "atom_indices": [list(x) for x in matches]}
            for name, query in GROUPS.items() if (matches := mol.GetSubstructMatches(query))]


SMILES = {"type": "string", "minLength": 1, "maxLength": 4000, "description": "Molecule SMILES; stereochemistry is preserved."}
PRECURSORS = {"type": "array", "minItems": 1, "maxItems": 20, "items": SMILES}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 20}
QUERY = {"type": "string", "minLength": 1, "maxLength": 500}


def _spec(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}}


TOOL_SPECS = [
    _spec("standardize_molecule", "Validate and canonicalize a molecule while preserving stereo, salts and charge.", {"smiles": SMILES}, ["smiles"]),
    _spec("molecular_properties", "Calculate RDKit descriptors and identify functional groups.", {"smiles": SMILES}, ["smiles"]),
    _spec("analyze_ch_sites", "Inventory carbon atoms bearing H with canonical atom indices. Structural analysis, not learned selectivity.", {"smiles": SMILES}, ["smiles"]),
    _spec("validate_precursor_set", "Check precursor structures, duplicates, self-loops and element coverage; does not establish reaction feasibility.", {"product": SMILES, "precursors": PRECURSORS}, ["product", "precursors"]),
    _spec("retrieve_reaction_analogs", "Retrieve similar products from the SpectraRetro catalogue and their model candidates; these are model analogs, not experimental records.", {"smiles": SMILES, "limit": LIMIT, "min_similarity": {"type": "number", "minimum": 0, "maximum": 1}}, ["smiles"]),
    _spec("evaluate_ch_selectivity", "Review structural C-H site factors and competing sites, without fabricated selectivity probabilities.", {"smiles": SMILES, "atom_index": {"type": "integer", "minimum": 0, "maximum": 399}}, ["smiles"]),
    _spec("recommend_conditions", "Retrieve source-backed imported reaction conditions by product similarity; missing evidence is reported explicitly.", {"smiles": SMILES, "limit": LIMIT}, ["smiles"]),
    _spec("check_functional_compatibility", "Report conditional functional-group review flags for selected reaction regimes.", {"smiles": SMILES, "regimes": {"type": "array", "maxItems": 6, "items": {"type": "string", "enum": ["acid", "base", "oxidation", "reduction", "transition_metal", "radical"]}}}, ["smiles"]),
    _spec("assess_complexity", "Calculate graph complexity descriptors; no synthetic feasibility or yield claim.", {"smiles": SMILES}, ["smiles"]),
    _spec("check_inventory", "Check configured terminal membership and separately inspect imported dated supplier quotes.", {"smiles": SMILES}, ["smiles"]),
    _spec("search_literature", "Search Europe PMC and return actual article metadata with source URLs; abstracts are not experimental verification.", {"query": QUERY, "limit": LIMIT}, ["query"]),
    _spec("search_patents", "Search Europe PMC's EPO biological patent records and provide a broader Google Patents query link. Coverage is limited.", {"query": QUERY, "limit": LIMIT}, ["query"]),
    _spec("lookup_compound", "Retrieve PubChem properties and available GHS hazard annotations for a SMILES.", {"smiles": SMILES, "include_hazards": {"type": "boolean"}}, ["smiles"]),
    _spec("forward_validate", "Apply a supplied reaction SMARTS or a configured external forward predictor, then compare products to the target. Template matches are not learned feasibility.", {"product": SMILES, "precursors": PRECURSORS, "reaction_smarts": {"type": "string", "minLength": 3, "maxLength": 3000}}, ["product", "precursors"]),
    _spec("expand_with_spectraretro", "Call SpectraRetro's one-step model adapter and preserve prediction provenance and Stage1 gate metadata.", {"product": SMILES, "top_k": {"type": "integer", "minimum": 1, "maximum": 300}}, ["product"]),
]


class ChemistryToolkit:
    def __init__(self, model=None, stock=(), inventory_path=None, evidence_path=None):
        self.model = model
        self.stock = frozenset(_mol(s)[0] for s in stock)
        self.inventory_path = Path(inventory_path) if inventory_path else None
        self.evidence_path = Path(evidence_path) if evidence_path else None
        self._lock = threading.RLock()
        self._quotes = []
        self._evidence = []
        self._cache = OrderedDict()
        self._analogs = None
        self._fingerprints = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
        self.forward_url = os.getenv("CHRETRO_FORWARD_URL", "").strip()
        self.forward_key = os.getenv("CHRETRO_FORWARD_API_KEY", "").strip()
        if self.inventory_path and self.inventory_path.is_file():
            self.set_inventory(json.loads(self.inventory_path.read_text(encoding="utf-8")), persist=False)
        if self.evidence_path and self.evidence_path.is_file():
            self.set_condition_evidence(json.loads(self.evidence_path.read_text(encoding="utf-8")), persist=False)

    def schemas(self):
        return deepcopy(TOOL_SPECS)

    def catalog(self):
        evidence_ready = bool(self.condition_evidence())
        rows = []
        for spec in TOOL_SPECS:
            item = spec["function"]
            name = item["name"]
            status = "ready"
            mode = "local"
            if name in ("search_literature", "search_patents", "lookup_compound"):
                mode = "public_api"
            elif name == "forward_validate":
                mode = "external_provider" if self.forward_url else "reaction_template"
                status = "ready" if self.forward_url else "requires_template"
            elif name == "recommend_conditions":
                status = "ready" if evidence_ready else "requires_evidence"
                mode = "curated_evidence"
            elif name in ("expand_with_spectraretro", "retrieve_reaction_analogs") and self.model is None:
                status = "not_configured"
            rows.append({"name": name, "description": item["description"], "status": status, "mode": mode})
        return rows

    def execute(self, name, args):
        spec = next((x["function"] for x in TOOL_SPECS if x["function"]["name"] == name), None)
        if spec is None:
            return {"status": "unavailable", "error": "Unknown chemistry tool"}
        if not isinstance(args, dict):
            return {"status": "invalid_input", "error": "Tool arguments must be an object"}
        schema = spec["parameters"]
        if set(args) - set(schema["properties"]) or any(x not in args for x in schema["required"]):
            return {"status": "invalid_input", "error": "Missing or unsupported tool arguments"}
        try:
            return getattr(self, name)(**args)
        except (ValueError, TypeError, KeyError) as exc:
            return {"status": "invalid_input", "error": str(exc)[:400]}
        except (URLError, TimeoutError, OSError) as exc:
            return {"status": "provider_unavailable", "error": f"External service request failed ({type(exc).__name__})"}
        except Exception as exc:
            return {"status": "error", "error": f"Tool could not complete ({type(exc).__name__})"}

    def standardize_molecule(self, smiles):
        canonical, mol = _mol(smiles)
        return {"status": "ok", "smiles": canonical, "canonical_smiles": canonical,
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "fragment_count": len(Chem.GetMolFrags(mol)),
                "formal_charge": Chem.GetFormalCharge(mol),
                "stereocenters": [{"atom_index": i, "assignment": tag} for i, tag in Chem.FindMolChiralCenters(mol, includeUnassigned=True)],
                "method": "RDKit sanitized, atom maps removed, isomeric canonical SMILES",
                "preserved": ["stereochemistry", "isotopes", "formal charge", "disconnected components"]}

    def molecular_properties(self, smiles):
        canonical, mol = _mol(smiles)
        return {"status": "ok", "smiles": canonical, "formula": rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": round(Descriptors.MolWt(mol), 3),
                "exact_mass": round(Descriptors.ExactMolWt(mol), 5),
                "logp": round(Descriptors.MolLogP(mol), 3),
                "tpsa": round(Descriptors.TPSA(mol), 3),
                "h_bond_donors": Descriptors.NumHDonors(mol), "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "ring_count": Descriptors.RingCount(mol), "heavy_atom_count": mol.GetNumHeavyAtoms(),
                "functional_groups": _groups(mol), "method": "RDKit calculated descriptors",
                "units": {"molecular_weight": "g/mol", "exact_mass": "Da", "tpsa": "angstrom^2"}}

    def analyze_ch_sites(self, smiles):
        canonical, mol = _mol(smiles)
        hetero = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)]
        distances = Chem.GetDistanceMatrix(mol)
        sites = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6 or atom.GetTotalNumHs(includeNeighbors=True) < 1:
                continue
            idx = atom.GetIdx()
            neighbors = list(atom.GetNeighbors())
            hybrid = str(atom.GetHybridization())
            heavy = sum(n.GetAtomicNum() > 1 for n in neighbors)
            carbonyl = any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetAtomicNum() == 8 for b in atom.GetBonds())
            aromatic_neighbor = any(n.GetIsAromatic() and n.GetAtomicNum() == 6 for n in neighbors)
            allylic = hybrid == "SP3" and any(any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(n).GetAtomicNum() == 6 for b in n.GetBonds()) for n in neighbors if n.GetAtomicNum() == 6)
            if atom.GetIsAromatic():
                label = "Aryl C(sp2)-H"
            elif carbonyl:
                label = "Formyl C(sp2)-H"
            elif hybrid == "SP":
                label = "Alkynyl C(sp)-H"
            elif hybrid == "SP2":
                label = "Vinylic / other C(sp2)-H"
            elif hybrid == "SP3" and aromatic_neighbor:
                label = "Benzylic C(sp3)-H"
            elif allylic:
                label = "Allylic C(sp3)-H"
            else:
                label = {1: "Primary aliphatic C(sp3)-H", 2: "Secondary aliphatic C(sp3)-H", 3: "Tertiary aliphatic C(sp3)-H"}.get(heavy, "Aliphatic C-H")
            reachable = [int(distances[idx, j]) for j in hetero if distances[idx, j] < mol.GetNumAtoms()]
            sites.append({"atom_index": idx, "hydrogen_count": int(atom.GetTotalNumHs(includeNeighbors=True)),
                          "site_class": label, "aromatic": atom.GetIsAromatic(), "in_ring": atom.IsInRing(),
                          "hybridization": hybrid, "heavy_degree": heavy,
                          "nearest_hetero_bonds": min(reachable) if reachable else None,
                          "direct_hetero_neighbors": [n.GetSymbol() for n in neighbors if n.GetAtomicNum() not in (1, 6)],
                          "benzylic": hybrid == "SP3" and aromatic_neighbor, "allylic": allylic})
        return {"status": "ok", "smiles": canonical, "sites": sites, "site_count": len(sites),
                "aromatic_site_count": sum(x["aromatic"] for x in sites),
                "sp3_site_count": sum(x["hybridization"] == "SP3" for x in sites),
                "atom_index_basis": "zero-based RDKit indices of returned canonical SMILES",
                "method": "RDKit structural C-H inventory", "scope": "Structural site inventory and mechanistic review",
                "factors": ["Directing-group identity and geometry", "Local electronics and charge", "Steric accessibility", "Bond strength and radical stability", "Catalyst, ligand, solvent and temperature"],
                "selectivity_probability": None}

    def validate_precursor_set(self, product, precursors):
        target, target_mol = _mol(product)
        if not isinstance(precursors, list) or not 1 <= len(precursors) <= 20:
            raise ValueError("Provide 1 to 20 precursor SMILES")
        canonical = []
        atoms = Counter()
        for precursor in precursors:
            smi, mol = _mol(precursor)
            canonical.extend(smi.split("."))
            atoms.update(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        counts = Counter(canonical)
        target_atoms = Counter(a.GetSymbol() for a in target_mol.GetAtoms() if a.GetAtomicNum() > 1)
        deficits = dict(target_atoms - atoms)
        duplicate = sorted(s for s, count in counts.items() if count > 1)
        self_loop = target in canonical
        return {"status": "ok", "product": target, "precursors": canonical,
                "structurally_valid": True, "search_eligible": not self_loop,
                "self_loop": self_loop, "duplicate_components": duplicate,
                "element_deficits": deficits, "feasibility": "not_assessed",
                "notes": (["Product occurs unchanged in precursor set"] if self_loop else []) +
                         (["Element deficits require unlisted reactants/reagents or a corrected prediction"] if deficits else []) +
                         (["Repeated components may represent stoichiometry; do not silently drop them"] if duplicate else []),
                "method": "RDKit graph validation and elemental coverage; no atom mapping or mechanistic proof"}

    def _analog_index(self):
        with self._lock:
            if self._analogs is not None:
                return self._analogs
            model = self.model
            if hasattr(model, "primary"):
                model = model.primary
            products = getattr(model, "available_products", None)
            if products is None:
                products = getattr(model, "data", {}).keys()
            items = []
            for smi in sorted(products or [])[:20000]:
                try:
                    canonical, mol = _mol(smi)
                    items.append((canonical, self._fingerprints.GetFingerprint(mol)))
                except ValueError:
                    continue
            self._analogs = items
            return items

    def retrieve_reaction_analogs(self, smiles, limit=5, min_similarity=0.0):
        canonical, mol = _mol(smiles)
        _bounded_int(limit, 1, 20, "limit")
        if type(min_similarity) not in (int, float) or not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1")
        if self.model is None:
            return {"status": "not_configured", "matches": [], "source": "model_candidate"}
        index = self._analog_index()
        scores = DataStructs.BulkTanimotoSimilarity(self._fingerprints.GetFingerprint(mol), [x[1] for x in index])
        ranked = sorted(zip(scores, index), key=lambda x: (-x[0], x[1][0]))
        matches = []
        for score, (smi, _) in ranked:
            if score < min_similarity or len(matches) >= limit:
                break
            candidates = self.model.predict(smi, 3)
            matches.append({"product": smi, "similarity": round(score, 5), "evidence_type": "model_candidate",
                            "candidates": candidates, "experimental_verification": False})
        return {"status": "ok", "smiles": canonical, "matches": matches, "indexed_products": len(index),
                "source": "SpectraRetro model catalogue", "evidence_type": "model_candidate",
                "method": "Morgan radius 2, 2048 bits, chirality enabled; Tanimoto product similarity",
                "scope": "Product-structure similarity, not reaction-mechanism or experimental-condition similarity"}

    def evaluate_ch_selectivity(self, smiles, atom_index=None):
        result = self.analyze_ch_sites(smiles)
        sites = result["sites"]
        if atom_index is not None:
            _bounded_int(atom_index, 0, 399, "atom_index")
            if atom_index not in {x["atom_index"] for x in sites}:
                raise ValueError("Selected canonical atom is not a carbon bearing hydrogen")
        reviews = []
        for site in sites:
            factors = []
            if site["nearest_hetero_bonds"] is not None and site["nearest_hetero_bonds"] <= 3:
                factors.append("Nearby heteroatom: review its coordinating ability and directing geometry")
            if site["benzylic"] or site["allylic"]:
                factors.append("Benzylic/allylic environment: review radical stabilization for the chosen mechanism")
            if site["heavy_degree"] >= 3:
                factors.append("High local substitution: review steric accessibility")
            if site["site_class"].startswith("Formyl"):
                factors.append("Formyl hydrogen: review acyl-radical and competing carbonyl chemistry")
            factors.append("Regioselectivity depends on the catalyst and reaction class")
            reviews.append({**site, "selected": atom_index == site["atom_index"], "review_factors": factors})
        return {"status": "ok", "smiles": result["smiles"], "sites": reviews,
                "competing_site_count": max(0, len(sites) - (1 if atom_index is not None else 0)),
                "method": "Structural heuristic review", "selectivity_probability": None,
                "evidence_needed": ["Reaction-class matched precedents", "Catalyst/ligand identity", "Observed regioisomer distribution"]}

    def recommend_conditions(self, smiles, limit=5):
        canonical, mol = _mol(smiles)
        _bounded_int(limit, 1, 20, "limit")
        records = self.condition_evidence()
        if not records:
            return {"status": "not_configured", "smiles": canonical, "recommendations": [],
                    "reason": "No source-backed experimental condition records have been imported",
                    "next_tool": "search_literature", "suggested_query": "C-H activation " + " ".join(x["name"] for x in _groups(mol)[:3]),
                    "required_record_fields": ["product", "conditions", "source_url or doi"]}
        fp = self._fingerprints.GetFingerprint(mol)
        found = []
        for record in records:
            if not isinstance(record, dict) or not record.get("conditions") or not (record.get("source_url") or record.get("doi")):
                continue
            try:
                smi, ref = _mol(record.get("product"))
            except ValueError:
                continue
            similarity = DataStructs.TanimotoSimilarity(fp, self._fingerprints.GetFingerprint(ref))
            # Do not recommend conditions from a weakly related structure.
            if similarity < 0.5:
                continue
            found.append({"product": smi, "similarity": round(similarity, 5),
                          "conditions": record["conditions"], "source_url": record.get("source_url") or "https://doi.org/" + record["doi"],
                          "doi": record.get("doi"), "evidence_type": "user_imported_condition_record",
                          "evidence_status": "user_supplied_not_independently_verified",
                          "applicability": "exact_product_requires_reaction_check" if canonical == smi else "analog_requires_review"})
        found.sort(key=lambda x: -x["similarity"])
        return {"status": "ok" if found else "no_evidence", "smiles": canonical, "recommendations": found[:limit],
                "method": "Source-backed condition retrieval, product similarity >= 0.5",
                "scope": "Source-attributed user records; source content and reaction applicability require review"}

    def condition_evidence(self):
        with self._lock:
            return deepcopy(self._evidence)

    def set_condition_evidence(self, rows, persist=True):
        if not isinstance(rows, list) or len(rows) > 20000:
            raise ValueError("Condition evidence must be a list with at most 20000 records")
        validated = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Every condition record must be an object")
            product, _ = _mol(row.get("product"))
            conditions = row.get("conditions")
            if not isinstance(conditions, (str, dict)) or not conditions:
                raise ValueError("Each condition record requires a nonempty conditions object or text")
            if isinstance(conditions, str) and not conditions.strip():
                raise ValueError("Condition text must not be blank")
            if isinstance(conditions, dict) and (len(conditions) > 30 or any(not isinstance(k, str) or not isinstance(v, (str, int, float, bool, type(None))) for k, v in conditions.items())):
                raise ValueError("Conditions must be a flat object with at most 30 named values")
            if len(json.dumps(conditions, allow_nan=False)) > 10000:
                raise ValueError("Condition record exceeds the 10000-character limit")
            doi = row.get("doi") or ""
            url = row.get("source_url") or ""
            if not isinstance(doi, str) or not isinstance(url, str):
                raise ValueError("Evidence DOI and source_url must be text")
            doi = doi.strip().removeprefix("https://doi.org/")
            url = url.strip()
            if doi and (len(doi) > 300 or not re.fullmatch(r"10\.\d{4,9}/\S+", doi)):
                raise ValueError("Evidence DOI has an invalid format")
            parsed = urlsplit(url)
            if url and (len(url) > 2000 or parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password):
                raise ValueError("Evidence source_url must be a valid public HTTP(S) citation URL")
            if not url and not doi:
                raise ValueError("Every condition record requires a source_url or DOI")
            entry = {"product": product, "conditions": deepcopy(conditions), "source_url": url or "https://doi.org/" + doi,
                     "doi": doi or None, "evidence_type": "user_imported_condition_record",
                     "evidence_status": "user_supplied_not_independently_verified", "imported_at": row.get("imported_at") or _now()}
            for field in ("title", "notes", "reaction_class"):
                value = row.get(field, "")
                if not isinstance(value, str) or len(value) > 2000:
                    raise ValueError(f"{field} must be text of at most 2000 characters")
                entry[field] = value
            validated.append(entry)
        with self._lock:
            # Keep different protocols from the same paper as separate records.
            def key(row):
                return row["product"], row["source_url"], json.dumps(row["conditions"], sort_keys=True)
            merged = {key(row): row for row in self._evidence}
            merged.update({key(row): row for row in validated})
            result = list(merged.values())
            if len(result) > 20000:
                raise ValueError("Combined condition evidence exceeds 20000 records")
            if persist and self.evidence_path:
                self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
                temp = self.evidence_path.with_suffix(self.evidence_path.suffix + ".tmp")
                temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temp, self.evidence_path)
            self._evidence = result
        return {"status": "ok", "imported_rows": len(validated), "total_rows": len(result),
                "evidence_status": "user_supplied_not_independently_verified"}

    def check_functional_compatibility(self, smiles, regimes=None):
        canonical, mol = _mol(smiles)
        choices = ["acid", "base", "oxidation", "reduction", "transition_metal", "radical"]
        if regimes is None:
            regimes = choices
        if not isinstance(regimes, list) or len(regimes) > 6 or any(x not in choices for x in regimes):
            raise ValueError("Choose supported reaction regimes")
        groups = _groups(mol)
        present = {x["name"] for x in groups}
        rules = [
            ("acid", {"acetal", "epoxide"}, "Acid may cleave acetals or open epoxides; inspect strength and exposure"),
            ("base", {"ester", "aldehyde", "ketone"}, "Strong base may promote hydrolysis or enolization; inspect conditions"),
            ("oxidation", {"alcohol", "aldehyde", "thiol", "sulfide", "alkene"}, "These groups may compete for oxidation under some oxidants"),
            ("reduction", {"aldehyde", "ketone", "nitro", "alkene", "alkyne", "aryl_halide"}, "These groups may be reduced or hydrogenolyzed depending on reagent and catalyst"),
            ("transition_metal", {"amine", "aromatic_nitrogen", "thiol", "sulfide"}, "Coordinating groups may direct the reaction or modify catalyst activity"),
            ("radical", {"alkene", "alkyne", "thiol", "peroxide"}, "Review radical addition, chain transfer and initiator interactions"),
        ]
        flags = [{"regime": regime, "functional_groups": sorted(names & present), "review": note}
                 for regime, names, note in rules if regime in regimes and names & present]
        return {"status": "ok", "smiles": canonical, "functional_groups": groups, "review_flags": flags,
                "method": "SMARTS and conditional expert-review rules", "compatibility_verdict": "not_determined",
                "scope": "Flags are condition-dependent; absence of a flag is not proof of compatibility"}

    def assess_complexity(self, smiles):
        canonical, mol = _mol(smiles)
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return {"status": "ok", "smiles": canonical, "heavy_atoms": mol.GetNumHeavyAtoms(),
                "ring_count": Descriptors.RingCount(mol), "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "stereocenters": len(chiral), "unassigned_stereocenters": sum(tag == "?" for _, tag in chiral),
                "bertz_complexity": round(Descriptors.BertzCT(mol), 3),
                "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
                "method": "RDKit graph descriptors", "synthesis_feasibility": "not_assessed"}

    def inventory(self):
        with self._lock:
            return deepcopy(self._quotes)

    def set_inventory(self, rows, persist=True):
        if not isinstance(rows, list) or len(rows) > 10000:
            raise ValueError("Inventory import must contain at most 10000 rows")
        validated = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Every inventory row must be an object")
            smi, _ = _mol(row.get("smiles"))
            price = row.get("price")
            if price not in (None, ""):
                if type(price) not in (int, float) or not math.isfinite(price) or price < 0:
                    raise ValueError("Quote prices must be finite nonnegative numbers")
            else:
                price = None
            available = row.get("available")
            if available is not None and type(available) is not bool:
                raise ValueError("Quote availability must be boolean or null")
            currency = str(row.get("currency", "")).upper().strip()
            unit = str(row.get("unit", "")).strip()
            if price is not None and (not re.fullmatch(r"[A-Z]{3}", currency) or not unit):
                raise ValueError("Prices require a three-letter currency and a package/unit description")
            entry = {"smiles": smi, "price": price, "available": available,
                     "currency": currency, "unit": unit, "evidence_type": "user_imported_quote"}
            for field in ("supplier", "source", "updated_at"):
                value = row.get(field, "")
                if not isinstance(value, str) or len(value) > 1000:
                    raise ValueError(f"{field} must be text of at most 1000 characters")
                entry[field] = value
            entry["imported_at"] = row.get("imported_at") or _now()
            validated.append(entry)
        with self._lock:
            merged = {(r["smiles"], r["supplier"], r["unit"], r["currency"]): r for r in self._quotes}
            merged.update({(r["smiles"], r["supplier"], r["unit"], r["currency"]): r for r in validated})
            result = list(merged.values())
            if len(result) > 10000:
                raise ValueError("Combined inventory exceeds 10000 quote rows")
            if persist and self.inventory_path:
                self.inventory_path.parent.mkdir(parents=True, exist_ok=True)
                temp = self.inventory_path.with_suffix(self.inventory_path.suffix + ".tmp")
                temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temp, self.inventory_path)
            self._quotes = result
        return {"status": "ok", "imported_rows": len(validated), "total_rows": len(result),
                "terminal_stock_modified": False}

    def check_inventory(self, smiles):
        canonical, _ = _mol(smiles)
        quotes = [x for x in self.inventory() if x["smiles"] == canonical]
        return {"status": "ok", "smiles": canonical, "in_terminal_inventory": canonical in self.stock,
                "quotes": quotes, "commercial_availability": "reported_available" if any(q["available"] is True for q in quotes) else "not_verified",
                "source": "Configured terminal set and separate user-imported quotes",
                "scope": "Terminal membership does not imply current supplier availability; quotes are not live"}

    def _fetch(self, url, payload=None, headers=None):
        key = (url, json.dumps(payload, sort_keys=True) if payload is not None else "")
        with self._lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < 600:
                return deepcopy(cached[1])
        request_headers = {"Accept": "application/json", "User-Agent": "CHRetroAgent/1.0 (research chemistry tools)"}
        request_headers.update(headers or {})
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        with build_opener(_NoRedirect()).open(Request(url, data=body, headers=request_headers), timeout=12) as response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("External response exceeded the 2 MB limit")
            result = json.loads(raw.decode("utf-8"))
        with self._lock:
            self._cache[key] = (time.monotonic(), deepcopy(result))
            self._cache.move_to_end(key)
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
        return result

    def _publication_search(self, query, limit, patents):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("Provide a search query of 1 to 500 characters")
        _bounded_int(limit, 1, 20, "limit")
        actual = f"({query.strip()}) AND SRC:PAT" if patents else f"({query.strip()}) NOT SRC:PAT"
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode({"query": actual, "format": "json", "pageSize": limit, "resultType": "lite"})
        data = self._fetch(url)
        records = []
        for row in data.get("resultList", {}).get("result", [])[:limit]:
            doi = row.get("doi")
            source = str(row.get("source", ""))
            identifier = str(row.get("id", ""))
            records.append({"title": row.get("title"), "authors": row.get("authorString"),
                            "year": row.get("pubYear"), "journal": row.get("journalTitle"), "doi": doi,
                            "identifier": identifier, "record_source": source,
                            "url": "https://doi.org/" + doi if doi else "https://europepmc.org/article/" + quote(source, safe="") + "/" + quote(identifier, safe=""),
                            "evidence_type": "retrieved_bibliographic_record"})
        result = {"status": "ok", "query": query, "total_hits": data.get("hitCount", 0), "results": records,
                  "source": "Europe PMC / EPO biological patent records" if patents else "Europe PMC",
                  "retrieved_at": _now(), "scope": "Bibliographic retrieval; full experimental procedures have not been extracted"}
        if patents:
            result.update({"coverage": "EPO biological patent records indexed by Europe PMC; not a comprehensive patent or freedom-to-operate search",
                           "broader_search_url": "https://patents.google.com/?" + urlencode({"q": query})})
        return result

    def search_literature(self, query, limit=5):
        return self._publication_search(query, limit, False)

    def search_patents(self, query, limit=5):
        return self._publication_search(query, limit, True)

    def lookup_compound(self, smiles, include_hazards=True):
        canonical, _ = _mol(smiles)
        if type(include_hazards) is not bool:
            raise ValueError("include_hazards must be boolean")
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/" + quote(canonical, safe="") + "/property/MolecularFormula,MolecularWeight,IUPACName,InChIKey/JSON"
        try:
            data = self._fetch(url)
        except HTTPError as exc:
            if exc.code == 404:
                return {"status": "not_found", "smiles": canonical, "source": "PubChem", "hazards_status": "not_retrieved"}
            raise
        records = data.get("PropertyTable", {}).get("Properties", [])
        if not records:
            return {"status": "not_found", "smiles": canonical, "source": "PubChem"}
        record = records[0]
        cid = record.get("CID")
        result = {"status": "ok", "smiles": canonical, "source": "PubChem PUG REST / PUG View", "cid": cid,
                  "properties": record, "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                  "retrieved_at": _now(), "hazard_annotations": [], "hazards_status": "not_requested"}
        if include_hazards and isinstance(cid, int):
            try:
                hazard = self._fetch(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=GHS%20Classification")
                annotations = []
                def walk(node):
                    if isinstance(node, dict):
                        if "StringWithMarkup" in node:
                            annotations.extend(x["String"][:1200] for x in node["StringWithMarkup"] if isinstance(x, dict) and isinstance(x.get("String"), str))
                        for key, value in node.items():
                            if key != "StringWithMarkup":
                                walk(value)
                    elif isinstance(node, list):
                        for value in node:
                            walk(value)
                walk(hazard.get("Record", {}).get("Section", []))
                result["hazard_annotations"] = list(dict.fromkeys(annotations))[:25]
                result["hazards_status"] = "retrieved" if annotations else "no_annotations"
                result["hazard_references"] = [{"source": x.get("SourceName"), "url": x.get("URL")} for x in hazard.get("Record", {}).get("Reference", [])[:20]]
            except (HTTPError, URLError, TimeoutError, OSError):
                result["hazards_status"] = "unavailable"
        result["hazards_scope"] = "Database annotations; absence of records does not establish safety"
        return result

    def forward_validate(self, product, precursors, reaction_smarts=None):
        target, _ = _mol(product)
        structural = self.validate_precursor_set(target, precursors)
        canonical = structural["precursors"]
        if reaction_smarts is None:
            if not self.forward_url:
                return {"status": "not_configured", "product": target, "structural_check": structural,
                        "target_recovered": None, "reason": "Supply a reaction SMARTS or configure CHRETRO_FORWARD_URL"}
            parsed = urlsplit(self.forward_url)
            if (parsed.scheme not in ("http", "https") or not parsed.netloc
                    or parsed.username or parsed.password or parsed.fragment
                    or (parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"))):
                raise ValueError("Forward provider requires HTTPS or loopback HTTP, without URL credentials or fragments")
            headers = {"Authorization": "Bearer " + self.forward_key} if self.forward_key else {}
            data = self._fetch(self.forward_url, {"precursors": canonical}, headers)
            raw = data.get("predicted_products")
            if not isinstance(raw, list) or len(raw) > 100 or not all(isinstance(s, str) for s in raw):
                raise ValueError("Forward provider must return predicted_products as a list of at most 100 SMILES")
            predictions = [_mol(s)[0] for s in raw]
            return {"status": "ok", "product": target, "predicted_products": predictions,
                    "target_recovered": target in predictions, "method": "Configured external forward predictor",
                    "evidence_type": "external_model_prediction", "provider_model": str(data.get("model", "unspecified"))[:200],
                    "scope": "Forward prediction agreement is not experimental verification"}
        if not isinstance(reaction_smarts, str) or not 3 <= len(reaction_smarts) <= 3000:
            raise ValueError("Provide a reaction SMARTS of at most 3000 characters")
        if len(canonical) > 6:
            raise ValueError("Template validation supports at most six precursor components")
        with rdBase.BlockLogs():
            reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        if reaction is None or reaction.GetNumReactantTemplates() != len(canonical) or reaction.GetNumProductTemplates() < 1:
            raise ValueError("Reaction SMARTS reactant count must match the precursor component count")
        molecules = [Chem.MolFromSmiles(s) for s in canonical]
        predictions = set()
        attempted = 0
        truncated = math.factorial(len(molecules)) > 120
        for ordering in islice(permutations(molecules), 120):
            attempted += 1
            with rdBase.BlockLogs():
                product_sets = reaction.RunReactants(ordering, maxProducts=32)
            truncated = truncated or len(product_sets) >= 32
            for product_set in product_sets:
                try:
                    parts = []
                    for mol in product_set:
                        Chem.SanitizeMol(mol)
                        parts.append(_mol(Chem.MolToSmiles(mol, isomericSmiles=True))[0])
                    predictions.update(parts)
                    if len(parts) > 1:
                        predictions.add(_mol(".".join(parts))[0])
                except (ValueError, RuntimeError):
                    continue
        return {"status": "ok", "product": target, "predicted_products": sorted(predictions)[:100],
                "target_recovered": target in predictions, "method": "RDKit user-supplied reaction SMARTS execution",
                "evidence_type": "template_consistency", "permutations_tested": attempted,
                "enumeration_truncated": truncated or len(predictions) > 100,
                "scope": "Template consistency only; no learned forward score or reaction feasibility claim"}

    def expand_with_spectraretro(self, product, top_k=100):
        canonical, _ = _mol(product)
        _bounded_int(top_k, 1, 300, "top_k")
        if self.model is None:
            return {"status": "not_configured", "product": canonical, "candidates": []}
        candidates = self.model.predict(canonical, top_k)
        result = {"status": "ok" if candidates else "no_candidates", "product": canonical,
                  "candidates": candidates, "candidate_count": len(candidates), "requested_top_k": top_k,
                  "source": "SpectraRetro model adapter", "evidence_type": "model_prediction"}
        model = self.model.primary if hasattr(self.model, "primary") else self.model
        if hasattr(model, "gate_status"):
            result["stage1_gate"] = model.gate_status(canonical)
        return result
