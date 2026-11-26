SR.init('cases');
const descriptions = {
  lapatinib: 'Five-step route to lapatinib free base.',
  nemiralisib: 'Nemiralisib / GSK2269557 · main and alternative precursor branches.',
  osimertinib_3r: 'Active derivative 3r. This target is not the Osimertinib API.',
  tafamidis: 'Tafamidis free acid · retains the extra structurally closed candidate.'
};
SR.api('/api/fixed-core/cases').then(data => {
  SR.$('#fixed-core-grid').innerHTML = data.cases.map(item => `<article class="case-card" data-case="${SR.esc(item.case_id)}"><img src="${SR.mol(item.product)}" alt="Molecular structure of ${SR.esc(item.title)}"><div class="eyebrow-row"><span class="tag">Latest case weights</span></div><h3>${SR.esc(item.title)}</h3><p>${SR.esc(descriptions[item.case_id])}</p><div class="smiles">${SR.esc(item.product)}</div><div class="toolbar" style="margin-top:20px"><button class="button lime case-plan" data-case-id="${SR.esc(item.case_id)}">Search this route ↗</button><button class="button secondary small" data-ask-agent="${SR.esc(`Use plan_fixed_core_case for ${item.case_id}. Save the actual route search and show the report link. Explain documented and extra unverified routes.`)}">Ask Agent ✳</button></div><p class="field-note case-status" role="status"></p></article>`).join('');
  SR.$('#case-weights').textContent = `Size SHA256: ${data.cases[0].weights_sha256.size}\nEDM SHA256: ${data.cases[0].weights_sha256.edm}`;
  SR.$$('.case-plan').forEach(button => button.onclick = async () => {
    const status = button.closest('.case-card').querySelector('.case-status');
    button.disabled = true; status.textContent = 'Searching latest-weight generated candidates…';
    try {
      const result = await SR.api('/api/fixed-core/plan', {method:'POST', body:JSON.stringify({case_id:button.dataset.caseId})});
      location.href = `/routes.html?id=${encodeURIComponent(result.analysis_id)}`;
    } catch (error) { status.textContent = error.message; button.disabled = false; }
  });
}).catch(error => { SR.$('#fixed-core-grid').textContent = error.message; });
