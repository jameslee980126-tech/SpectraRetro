"""Read current public supplier offers; never substitute stored inventory quotes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from agentic_retrosynthesis.planner import canonical_smiles

TZ = timezone(timedelta(hours=8))
ALLOWED = {"cymitquimica.com", "pubchem.ncbi.nlm.nih.gov"}
# Identity and product links only. Prices are always read over HTTP at query time.
KNOWN = [
    ("COC(OC)OC", "149-73-5", "Trimethyl Orthoformate", "3B-O0068/149-73-5/trimethyl-orthoformate/"),
    ("C1COCCN1", "110-91-8", "Morpholine", "3B-M0465/110-91-8/morpholine/"),
    ("Nc1ccc(C(=O)O)cc1O", "2374-03-0", "4-Amino-3-hydroxybenzoic Acid", "3B-A1194/2374-03-0/4-amino-3-hydroxybenzoic-acid/"),
    ("Clc1cc(Cl)cc(I)c1", "3032-81-3", "1,3-Dichloro-5-iodobenzene", "3B-D1866/3032-81-3/13-dichloro-5-iodobenzene/"),
    ("O=C(Cl)C(=O)Cl", "79-37-8", "Oxalyl Chloride", "3B-O0082/79-37-8/oxalyl-chloride/"),
]


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


class SameHostRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != urlsplit(req.full_url).hostname:
            raise ValueError("Supplier redirected outside its approved host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def read_public(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED:
        raise ValueError("Unapproved supplier endpoint")
    request = Request(url, headers={"User-Agent": "CHRetroAgent/1.0 (public reagent offers)",
                                   "Cache-Control": "no-cache", "Accept": "text/html,application/json"})
    opener = build_opener(SameHostRedirect())
    try:
        response = opener.open(request, timeout=10)
    except HTTPError as exc:
        # Python 3.10 does not automatically follow permanent 308 redirects.
        if exc.code != 308:
            raise
        target = urljoin(url, exc.headers.get("Location", ""))
        if urlsplit(target).hostname != parsed.hostname or urlsplit(target).scheme != "https":
            raise ValueError("Supplier redirected outside its approved host")
        response = opener.open(Request(target, headers=request.headers), timeout=10)
    with response:
        raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Supplier response exceeded limit")
        return raw.decode("utf-8"), response.url


def parse_offers(document, cas, source, retrieved_at):
    """Accept exact-CAS structured Product offers with an explicit package and currency."""
    offers, identity = [], {}
    for raw in re.findall(r'<script\b[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', document, re.S | re.I):
        try:
            entry = json.loads(raw)
        except (ValueError, TypeError):
            continue
        entries = entry if isinstance(entry, list) else entry.get("@graph", [entry])
        for product in entries:
            if product.get("@type") != "Product" or str(product.get("productID")) != cas:
                continue
            identity = {"name": product.get("name"), "brand": (product.get("brand") or {}).get("name", "")}
            properties = {p.get("name"): p.get("value") for p in product.get("additionalProperty", [])}
            rows = product.get("offers", [])
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows:
                try:
                    price = float(row["price"])
                except (KeyError, TypeError, ValueError):
                    continue
                currency, sku = row.get("priceCurrency", ""), row.get("sku", "")
                pack = re.search(r'(\d+(?:\.\d+)?\s*(?:mg|kg|ml|g|l))$', sku, re.I)
                if not math.isfinite(price) or price <= 0 or not re.fullmatch(r'[A-Z]{3}', currency) or not pack:
                    continue
                expires = row.get("priceValidUntil")
                if expires and expires[:10] < retrieved_at[:10]:
                    continue
                offers.append({"supplier": "CymitQuimica", "brand": identity["brand"],
                               "price": price, "currency": currency, "package": pack.group(1),
                               "sku": sku, "purity": properties.get("Purity"),
                               "availability": str(row.get("availability", "")).rsplit("/", 1)[-1] or "Not stated",
                               "region": "Europe", "source_url": source, "retrieved_at": retrieved_at,
                               "price_valid_until": expires, "quote_type": "public_supplier_listing",
                               "delivery_to_china_verified": False})
    return identity, offers


class LiveSupplierQuotes:
    def __init__(self, data_dir):
        self.directory = Path(data_dir) / "supplier_queries"
        self.identities = {canonical_smiles(s): {"cas": c, "name": n, "product": p} for s, c, n, p in KNOWN}
        self.cache, self.lock = {}, threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="supplier")
        self.slots = threading.BoundedSemaphore(2)

    def identify(self, smiles):
        with self.lock:
            known = self.identities.get(smiles)
        if known:
            return dict(known)
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/" + quote(smiles, safe="") + "/synonyms/JSON"
        data = json.loads(read_public(url)[0])["InformationList"]["Information"][0]
        synonyms = data.get("Synonym", [])
        candidates = [c for c in synonyms if re.fullmatch(r'\d{2,7}-\d{2}-\d', c)][:2]
        for cas in candidates:
            u = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + cas + "/property/CanonicalSMILES/JSON"
            properties = json.loads(read_public(u)[0])["PropertyTable"]["Properties"]
            if any(canonical_smiles(p.get("ConnectivitySMILES") or p.get("CanonicalSMILES") or p.get("SMILES") or "") == smiles for p in properties):
                identity = {"cas": cas, "name": synonyms[0] if synonyms else smiles, "cid": data.get("CID")}
                with self.lock:
                    self.identities[smiles] = identity
                return dict(identity)
        return {"name": synonyms[0] if synonyms else smiles, "cid": data.get("CID"), "cas": None}

    def material(self, smiles):
        retrieved = now()
        mol = Chem.MolFromSmiles(smiles)
        result = {"smiles": smiles, "name": None, "cas": None, "offers": [], "status": "needs_inquiry",
                  "formula": rdMolDescriptors.CalcMolFormula(mol), "molecular_weight": round(Descriptors.MolWt(mol), 2),
                  "queried_at": retrieved, "sources": []}
        try:
            identity = self.identify(smiles)
            result.update({k: v for k, v in identity.items() if k != "product"})
            cas = identity.get("cas")
            if not cas:
                result["status"] = "identity_unresolved"
                return result
            if identity.get("product"):
                urls = ["https://cymitquimica.com/products/" + identity["product"]]
            else:
                search_url = "https://cymitquimica.com/search/" + cas + "/"
                search, _ = read_public(search_url)
                urls = list(dict.fromkeys("https://cymitquimica.com" + html.unescape(u)
                    for u in re.findall(r'href="(/products/[^"<>]+)"', search)
                    if "/" + cas + "/" in u))[:2]
            for url in urls:
                try:
                    document, actual_url = read_public(url)
                    stamp = now()
                    _, offers = parse_offers(document, cas, actual_url, stamp)
                    result["offers"].extend(offers)
                    result["sources"].append({"url": actual_url, "retrieved_at": stamp,
                        "sha256": hashlib.sha256(document.encode()).hexdigest(), "http_status": 200})
                except Exception as exc:
                    result["sources"].append({"url": url, "status": "unavailable", "error_type": type(exc).__name__})
            result["status"] = "quoted" if result["offers"] else "needs_inquiry"
        except Exception as exc:
            result.update(status="query_unavailable", error_type=type(exc).__name__)
        return result

    def route_quotes(self, result, index, refresh=False):
        routes = result.get("routes", [])
        if type(index) is not int or not 0 <= index < len(routes):
            raise ValueError("Route index outside available routes")
        materials = sorted({canonical_smiles(s) for s in routes[index].get("starting_materials", [])})
        if len(materials) > 24:
            raise ValueError("Supplier query limited to 24 starting materials per route")
        key = tuple(materials)
        today = now()[:10]
        with self.lock:
            cached = self.cache.get(key)
            if not refresh and cached and cached[0] == today and time.monotonic() - cached[1] < 300:
                return {**cached[2], "route_index": index, "cache_reused": True}
        if not self.slots.acquire(blocking=False):
            raise ValueError("Supplier query capacity is busy; try again shortly")
        try:
            futures = [self.pool.submit(self.material, s) for s in materials]
            rows = [f.result() for f in futures]
            report = {"status": "complete", "route_index": index, "queried_at": now(), "query_date": today,
                      "materials": rows, "cache_reused": False, "cache_seconds": 300,
                      "supplier": "CymitQuimica public catalogue", "currency_conversion": False,
                      "quoted_materials": sum(bool(r["offers"]) for r in rows),
                      "note": "Public package prices queried today. Shipping, taxes and delivery to your location require supplier confirmation."}
            self.directory.mkdir(parents=True, exist_ok=True)
            token = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
            (self.directory / (token + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.lock:
                self.cache[key] = (today, time.monotonic(), report)
                if len(self.cache) > 128:
                    self.cache.pop(next(iter(self.cache)))
            return report
        finally:
            self.slots.release()
