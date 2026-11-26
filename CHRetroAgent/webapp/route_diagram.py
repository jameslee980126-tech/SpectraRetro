"""Standalone molecular route SVG and explicitly scoped structural checks."""
from __future__ import annotations

import base64
import copy
from collections import defaultdict
import xml.etree.ElementTree as ET

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from agentic_retrosynthesis.planner import canonical_smiles


NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)


def chemistry_review(step):
    """A forward SMARTS consistency check is not experimental feasibility."""
    mols = [Chem.MolFromSmiles(s) for s in step["precursors"]]
    valid = bool(Chem.MolFromSmiles(step["product"])) and all(m is not None for m in mols)
    result = {"structures_valid": valid, "forward_template_match": None,
              "experimental_feasibility_verified": False,
              "conditions_verified": False, "template_name": None}
    if not valid or len(mols) != 2:
        return result
    reaction = rdChemReactions.ReactionFromSmarts(
        "[C:1](=[O:2])[O;H1].[N;H1:3]>>[C:1](=[O:2])[N:3]")
    for reactants in (tuple(mols), tuple(reversed(mols))):
        for products in reaction.RunReactants(reactants):
            for product in products:
                try:
                    Chem.SanitizeMol(product)
                    if canonical_smiles(Chem.MolToSmiles(product)) == canonical_smiles(step["product"]):
                        result.update(forward_template_match=True,
                                      template_name="carboxylic acid + secondary amine amidation")
                        return result
                except (ValueError, RuntimeError):
                    continue
    return result


def route_graph(result, route_index):
    routes = result.get("routes", [])
    if type(route_index) is not int or not 0 <= route_index < len(routes):
        raise ValueError("Route index is outside the available routes")
    route = routes[route_index]
    steps = {s["product"]: (i, s) for i, s in enumerate(route["steps"], 1)}
    if len(steps) != len(route["steps"]):
        raise ValueError("Ambiguous duplicate expanded molecule in route")
    nodes, reactions, levels, memo = [], [], defaultdict(list), {}
    terminal = set(route["starting_materials"])

    def visit(smiles, depth, ancestors):
        if smiles in ancestors:
            raise ValueError("Cyclic route cannot be drawn as a complete route")
        if smiles in memo:
            return memo[smiles]
        node = {"id": "m" + str(len(nodes)), "smiles": smiles, "depth": depth,
                "terminal": smiles in terminal, "target": depth == 0,
                "step_number": steps.get(smiles, (None,))[0]}
        nodes.append(node)
        memo[smiles] = node["id"]
        levels[depth].append(node)
        if smiles in steps:
            index, step = steps[smiles]
            # Expanded branches first; never remove any companion precursor.
            children = sorted(step["precursors"], key=lambda s: (s not in steps, step["precursors"].index(s)))
            child_ids = [visit(s, depth + 1, ancestors | {smiles}) for s in children]
            reactions.append({"index": index, "parent": node["id"], "children": child_ids,
                              "step": step, "review": chemistry_review(step)})
        elif smiles not in terminal:
            raise ValueError("Unresolved leaf in supposedly complete route")
        return node["id"]

    visit(result["target"], 0, set())
    if len(reactions) != len(route["steps"]):
        raise ValueError("Disconnected reaction in complete route")
    # Longest root distance keeps shared precursors below every parent.
    lookup = {n["id"]: n for n in nodes}
    for _ in range(len(nodes)):
        changed = False
        for reaction in reactions:
            depth = lookup[reaction["parent"]]["depth"] + 1
            for child in reaction["children"]:
                if lookup[child]["depth"] < depth:
                    lookup[child]["depth"] = depth
                    changed = True
        if not changed:
            break
    levels.clear()
    for node in nodes:
        levels[node["depth"]].append(node)
    width = max(700, max(len(row) for row in levels.values()) * 280 + 64)
    for depth, row in levels.items():
        row.sort(key=lambda n: (n["terminal"], nodes.index(n)))
        left = (width - len(row) * 280) / 2
        for index, node in enumerate(row):
            node.update(x=left + index * 280 + 12, y=86 + depth * 300)
    return {"width": width, "height": 86 + len(levels) * 300, "nodes": nodes,
            "reactions": reactions, "route": route}


def compact_layout(graph):
    """Paper-style precursor groups; follow the main chain across rows, then branches."""
    graph = copy.deepcopy(graph)
    lookup = {n["id"]: n for n in graph["nodes"]}
    by_parent = {r["parent"]: r for r in graph["reactions"]}
    groups, occupied, assigned, used_terminal = [], set(), {}, set()
    columns, pitch_x, pitch_y = (4 if len(by_parent) >= 7 else 3), 340, 228

    def new_group(row, col):
        while (row, col) in occupied:
            col += 1
            if col == columns:
                row, col = row + 1, 0
        group = {"id": "g" + str(len(groups)), "row": row, "column": col,
                 "x": 42 + col * pitch_x, "y": 72 + row * pitch_y,
                 "w": 220, "h": 194, "nodes": []}
        groups.append(group);occupied.add((row, col))
        return group

    def next_slot(group):
        col = group["column"] + 1
        return (group["row"], col) if col < columns else (group["row"] + 1, 0)

    def add_terminal(child_id, group, reaction):
        node = lookup[child_id]
        if child_id in used_terminal:
            node = copy.deepcopy(node)
            node["id"] = child_id + "_use_" + str(reaction["index"])
            graph["nodes"].append(node);lookup[node["id"]] = node
            reaction["children"] = [node["id"] if c == child_id else c for c in reaction["children"]]
        used_terminal.add(child_id)
        group["nodes"].append(node["id"]);assigned[node["id"]] = group

    def expand(node_id, group):
        assigned[node_id] = group;group["nodes"].append(node_id)
        reaction = by_parent.get(node_id)
        if not reaction:
            return
        expanded = [c for c in reaction["children"] if not lookup[c]["terminal"]]
        terminals = [c for c in reaction["children"] if lookup[c]["terminal"]]
        if not expanded:
            end = new_group(*next_slot(group))
            for child in terminals:
                add_terminal(child, end, reaction)
            return
        for branch, child in enumerate(expanded):
            if child in assigned:
                destination = assigned[child]
            else:
                slot = next_slot(group) if branch == 0 else (max(g["row"] for g in groups) + 1, 0)
                destination = new_group(*slot)
            # Partners belong to the resulting precursor set, after the arrow.
            if branch == 0:
                for terminal in terminals:
                    add_terminal(terminal, destination, reaction)
            if child not in assigned:
                expand(child, destination)

    target = next(n for n in graph["nodes"] if n["target"])
    if target["terminal"]:
        group = new_group(0, 0);group["nodes"].append(target["id"]);assigned[target["id"]] = group
    else:
        expand(target["id"], new_group(0, 0))
    for group in groups:
        group["logical_column"] = group["column"]
        group["flow"] = 1 if group["row"] % 2 == 0 else -1
        if group["flow"] == -1:
            group["column"] = columns - 1 - group["column"]
            group["x"] = 42 + group["column"] * pitch_x
        members = [lookup[i] for i in group["nodes"]]
        main = [n for n in members if not n["terminal"]]
        terminals = [n for n in members if n["terminal"]]
        if main:
            main[0].update(x=group["x"], y=group["y"] + 5, w=220, h=116)
            size = min(108, (220 - 8 * max(0, len(terminals) - 1)) / max(1, len(terminals)))
            for i, node in enumerate(terminals):
                node.update(x=group["x"] + (220 - len(terminals) * (size + 8) + 8) / 2 + i * (size + 8),
                            y=group["y"] + 128, w=size, h=65)
        else:
            step_h = min(92, 184 / max(1, len(terminals)))
            for i, node in enumerate(terminals):
                node.update(x=group["x"] + 20, y=group["y"] + 6 + i * step_h,
                            w=180, h=step_h - 4)
        for node in members:
            node.update(row=group["row"], column=group["column"], group_id=group["id"])
    graph.update(width=columns * pitch_x + 28,
                 height=72 + (max(g["row"] for g in groups) + 1) * pitch_y + 20,
                 groups=groups, layout="paper-reaction-scheme")
    return graph


def route_checks(result, route_index=0):
    graph = route_graph(result, route_index)
    route = graph["route"]
    reviews = [chemistry_review(s) for s in route["steps"]]
    explicit_matches = [s.get("metadata", {}).get("literature_match") for s in route["steps"]]
    has_literature_annotations = all("literature_match" in s.get("metadata", {}) for s in route["steps"])
    matched = sum(bool(m) for m in explicit_matches)
    return {"route_index": route_index, "checks": [
        {"name": "Molecular structures", "status": "passed" if all(r["structures_valid"] for r in reviews) else "failed",
         "detail": f"RDKit parsed every product and precursor in {len(reviews)} reactions."},
        {"name": "Route connectivity", "status": "passed",
         "detail": f"All {len(graph['reactions'])} reactions are connected to the target; every precursor branch is retained."},
        {"name": "Starting-material closure", "status": "passed",
         "detail": "Every leaf meets this report's configured starting-material boundary."},
        {"name": "Literature precursor sets", "status": ("passed" if matched == len(reviews) else "partial") if has_literature_annotations else "not_checked",
         "detail": f"{matched}/{len(reviews)} reactions match annotated literature precursor sets." if has_literature_annotations else "This report has no per-edge literature match annotation."},
        {"name": "Forward reaction template", "status": "consistent" if any(r["forward_template_match"] for r in reviews) else "not_checked",
         "detail": f"{sum(bool(r['forward_template_match']) for r in reviews)} reaction(s) recover the target under the configured amidation SMARTS." if any(r["forward_template_match"] for r in reviews) else "No applicable forward-template result is recorded; this is not a failed experiment."},
        {"name": "Experimental feasibility", "status": "not_verified", "detail": "Reaction conditions, yields and experimental feasibility have not been established by these checks."}
    ]}


def route_svg(result, route_index=0):
    graph = compact_layout(route_graph(result, route_index))
    width, height = graph["width"], graph["height"]
    svg = ET.Element("{" + NS + "}svg", {"viewBox": f"0 0 {width} {height}",
        "width": str(width), "height": str(height), "role": "img",
        "data-layout": graph["layout"], "data-molecule-count": str(len(graph["nodes"])),
        "data-reaction-count": str(len(graph["reactions"]))})

    def element(name, attrs=None, parent=svg, text=None):
        node = ET.SubElement(parent, "{" + NS + "}" + name,
                             {k: str(v) for k, v in (attrs or {}).items()})
        if text is not None:
            node.text = str(text)
        return node

    element("title", text=(result.get("case_name") or "Retrosynthesis") + " — complete connected route")
    element("desc", text="Paper-style molecular scheme. Follow the serpentine arrows: right, down, left, then down again. Precursors from each disconnection are grouped after its arrow. All branches are retained. Rank denotes recorded model candidate rank, not reaction probability. T denotes a configured terminal precursor set, not verified supplier inventory.")
    element("rect", {"width": width, "height": height, "fill": "white"})
    defs = element("defs")
    for name, color in [("arrow", "#999999"), ("extra", "#ce8c3d")]:
        marker = element("marker", {"id": name, "viewBox": "0 0 10 10", "refX": 9,
            "refY": 5, "markerWidth": 6, "markerHeight": 6, "orient": "auto"}, parent=defs)
        element("path", {"d": "M 0 0 L 10 5 L 0 10 z", "fill": color}, parent=marker)
    font = "Georgia, Times New Roman, serif"
    element("text", {"x": 30, "y": 30, "fill": "#111111", "font-size": 20,
                    "font-family": font, "font-weight": 700},
            text=(result.get("case_name") or "Retrosynthesis") + f" · {len(graph['reactions'])} steps")
    element("text", {"x": 30, "y": 49, "fill": "#777777", "font-size": 11,
                    "font-family": "Arial, sans-serif"}, text="Target → precursor sets · follow the arrows: right, down, left")
    lookup = {n["id"]: n for n in graph["nodes"]}
    groups = {g["id"]: g for g in graph["groups"]}
    for reaction in graph["reactions"]:
        parent = lookup[reaction["parent"]];source = groups[parent["group_id"]]
        metadata = reaction["step"].get("metadata", {});match = metadata.get("literature_match")
        extra = "literature_match" in metadata and match is None
        color = "#ce8c3d" if extra else "#999999"
        group = element("g", {"class": "route-reaction", "data-step": reaction["index"],
                              "data-extra-edge": str(extra).lower()})
        direction = source["flow"]
        sx = source["x"] + (226 if direction == 1 else -10)
        sy = source["y"] + 96
        destinations = defaultdict(list)
        for child in reaction["children"]:
            destinations[lookup[child]["group_id"]].append(child)
        for lane_index, (destination_id, children) in enumerate(destinations.items()):
            destination = groups[destination_id]
            tx = destination["x"] + (-10 if direction == 1 else 226)
            ty = destination["y"] + 96
            anchor = "middle"
            if destination["row"] == source["row"] and destination["column"] == source["column"] + direction:
                path = f"M {sx} {sy} L {tx} {ty}"
                lx, ly = (sx + tx) / 2, sy
            elif destination["row"] == source["row"] + 1 and destination["column"] == source["column"]:
                # Direct downward turn, followed by the opposite row direction.
                cx = source["x"] + 110
                start_y, end_y = source["y"] + 199, destination["y"] - 4
                path = f"M {cx} {start_y} L {cx} {end_y}"
                lx, ly, anchor = cx + 20, (start_y + end_y) / 2 + 3, "start"
            else:
                # Branch/shared-node edges travel only through the outer and row gutters.
                side = source["x"] + (244 if direction == 1 else -28)
                lane = source["y"] + 211 + lane_index * 5
                outer = width - 18 - lane_index * 5 if direction == 1 else 18 + lane_index * 5
                end_lane = destination["y"] - 17
                cx = destination["x"] + 110
                path = f"M {sx} {sy} Q {side} {sy} {side} {sy+18} L {side} {lane-12} Q {side} {lane} {side+direction*12} {lane} L {outer} {lane} L {outer} {end_lane} L {cx} {end_lane} L {cx} {destination['y']-4}"
                lx, ly = side + direction * 25, sy + 61
                anchor = "start" if direction == 1 else "end"
            for child_id in children:
                # One arrow denotes the entire precursor set; retain each true edge in SVG metadata.
                element("path", {"class": "precursor-edge", "data-from": parent["id"], "data-to": child_id,
                    "d": path, "stroke": color, "stroke-width": 1.5, "fill": "none",
                    "stroke-dasharray": "5 4" if extra else "none",
                    "marker-end": "url(#extra)" if extra else "url(#arrow)"}, parent=group)
            labels = element("g", {"class": "edge-label", "data-direction": direction}, parent=group)
            element("text", {"x": lx, "y": ly - 11, "text-anchor": anchor, "fill": "#e11b22",
                             "font-size": 14, "font-family": font}, parent=labels, text=f"Step {reaction['index']}")
            rank = reaction["step"].get("model_rank")
            if rank is not None:
                element("text", {"x": lx, "y": ly + 18, "text-anchor": anchor, "fill": "#111111",
                                 "font-size": 12, "font-family": font}, parent=labels, text=f"Rank {rank}")
            if extra:
                element("text", {"x": lx, "y": ly + 25, "text-anchor": anchor, "fill": "#aa6a17",
                                 "font-size": 9, "font-family": font}, parent=labels, text="Extra candidate")
    for node in graph["nodes"]:
        x, y, w, h = node["x"], node["y"], node["w"], node["h"]
        group = element("g", {"class": "route-molecule", "id": node["id"], "data-smiles": node["smiles"],
                              "data-role": "target" if node["target"] else "terminal" if node["terminal"] else "intermediate"})
        molecule = Chem.MolFromSmiles(node["smiles"])
        element("title", parent=group, text=node["smiles"] + " | " + rdMolDescriptors.CalcMolFormula(molecule))
        drawer = rdMolDraw2D.MolDraw2DSVG(int(w), int(h))
        opts = drawer.drawOptions();opts.clearBackground = False;opts.bondLineWidth = 1.25
        opts.minFontSize = 9;opts.maxFontSize = 13;opts.padding = .06
        drawer.DrawMolecule(molecule);drawer.FinishDrawing()
        encoded = base64.b64encode(drawer.GetDrawingText().encode()).decode()
        element("image", {"x": x, "y": y, "width": w, "height": h,
                          "href": "data:image/svg+xml;base64," + encoded}, parent=group)
    for group in graph["groups"]:
        if all(lookup[n]["terminal"] for n in group["nodes"]):
            tag = element("g", {"class": "terminal-set"})
            element("title", parent=tag, text="T: all precursors in this set meet the configured terminal boundary; supplier availability is not verified.")
            element("rect", {"x": group["x"] + 211, "y": group["y"] + 81,
                             "width": 25, "height": 30, "rx": 3, "fill": "#bce5e8"}, parent=tag)
            element("text", {"x": group["x"] + 223.5, "y": group["y"] + 101,
                             "text-anchor": "middle", "font-family": font, "font-size": 15}, parent=tag, text="T")
    return ET.tostring(svg, encoding="utf-8", xml_declaration=True)
