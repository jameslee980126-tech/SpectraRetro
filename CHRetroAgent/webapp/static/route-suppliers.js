const RouteSuppliers = (() => {
  let generation=0;
  const esc=value=>SR.esc(value);
  const money=offer=>new Intl.NumberFormat('en',{style:'currency',currency:offer.currency}).format(offer.price);
  const link=url=>{try{const u=new URL(url);return u.protocol==='https:'?u.href:null;}catch{return null;}};
  const date=value=>value?new Date(value).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}):'—';
  function offerRows(offers){return offers.map(o=>{const url=link(o.source_url);return `<tr><td><b>${esc(o.supplier)}</b><small>Brand: ${esc(o.brand||'—')} · ${esc(o.region)}</small></td><td class="supply-price"><b>${esc(money(o))}</b><small>${esc(o.currency)} / package</small></td><td>${esc(o.package)}<small>${esc(o.purity||'Purity not stated')}</small></td><td>${esc(o.sku)}<small>${esc(o.availability==='InStock'?'Supplier lists in stock':o.availability)}</small></td><td>${url?`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">Supplier page ↗</a>`:''}<small>Checked ${esc(date(o.retrieved_at))}</small>${o.price_valid_until?`<small>Listed valid until ${esc(o.price_valid_until)}</small>`:''}</td></tr>`}).join('');}
  async function mount(host,analysis,index,refresh=false){
    const token=++generation;
    host.innerHTML='<div class="split-title"><div><div class="kicker">Starting materials</div><h3>Suppliers & today’s package prices</h3></div></div><p class="field-note" role="status">Querying public supplier websites for this route’s starting materials…</p>';
    try{
      const result=await SR.api(`/api/route-suppliers?analysis_id=${encodeURIComponent(analysis)}&route_index=${index}&refresh=${refresh?'1':'0'}`);
      if(token!==generation||!host.isConnected)return;
      host.innerHTML=`<div class="split-title"><div><div class="kicker">Starting materials</div><h3>Suppliers & today’s package prices</h3><p class="field-note">${result.quoted_materials}/${result.materials.length} materials priced · queried ${esc(date(result.queried_at))}${result.cache_reused?' · reused within 5 minutes':''}</p></div><div class="toolbar"><button type="button" class="button secondary small supply-refresh">Refresh prices ↻</button><button type="button" class="button ghost small supply-export">CSV ↓</button></div></div><p class="field-note">${esc(result.note)}</p>${result.materials.map((m,i)=>`<article class="supply-material"><div class="supply-identity"><img src="${SR.mol(m.smiles)}" alt="Starting material ${i+1}"><div><b>${esc(m.name||`Starting material ${i+1}`)}</b><div class="field-note">CAS ${esc(m.cas||'not resolved')} · ${esc(m.formula)} · ${esc(m.molecular_weight)} g/mol</div><details><summary class="field-note">Structure</summary><div class="smiles">${esc(m.smiles)}</div></details></div></div>${m.offers.length?`<div class="supply-table"><table><thead><tr><th>Supplier / brand</th><th>Price</th><th>Package / purity</th><th>Product / stock</th><th>Source / query time</th></tr></thead><tbody>${offerRows(m.offers)}</tbody></table></div>`:`<p class="notice amber">${m.status==='query_unavailable'?'Supplier query unavailable; no current quote obtained.':m.status==='identity_unresolved'?'Exact catalogue identity needs confirmation.':'No public package price found; request a supplier quote.'}${m.cas?` <a href="https://cymitquimica.com/search/${encodeURIComponent(m.cas)}/" target="_blank" rel="noopener noreferrer">Search supplier ↗</a>`:''}</p>`}</article>`).join('')}`;
      host.querySelector('.supply-refresh').onclick=()=>mount(host,analysis,index,true);
      host.querySelector('.supply-export').onclick=()=>{
        const cell=v=>`"${String(v??'').replaceAll('"','""')}"`;
        const rows=[['material','cas','smiles','supplier','brand','price','currency','package','purity','sku','availability','source','queried_at'],...result.materials.flatMap(m=>m.offers.length?m.offers.map(o=>[m.name,m.cas,m.smiles,o.supplier,o.brand,o.price,o.currency,o.package,o.purity,o.sku,o.availability,o.source_url,o.retrieved_at]):[[m.name,m.cas,m.smiles,'','','','','','','',m.status,'',m.queried_at]])];
        SR.download(`route-${index+1}-supplier-quotes.csv`,rows.map(r=>r.map(cell).join(',')).join('\n'),'text/csv');
      };
    }catch(error){if(token===generation&&host.isConnected){host.innerHTML=`<h3>Supplier query unavailable</h3><p class="field-note">${esc(error.message)}</p><button type="button" class="button secondary small supply-retry">Retry query</button>`;host.querySelector('.supply-retry').onclick=()=>mount(host,analysis,index,true);}}
  }
  return {mount};
})();
