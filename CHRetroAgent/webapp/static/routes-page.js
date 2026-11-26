SR.init('routes');
const root = SR.$('#route-root');
const id = new URLSearchParams(location.search).get('id');
const PAGE_SIZE = 15;
let current = null, selectedRoute = 0, candidatePage = 0;
const number = (value, digits = 3) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
const sourceLabel = source => ({
  spectraretro_stage1_gated_3d_snapshot: 'SpectraRetro',
  spectraretro_3d_snapshot: 'SpectraRetro',
  llm_generated_unverified: 'LLM proposal · unverified',
  latest_case_fixed_core_generated_bank: 'SpectraRetro'
}[source] || source || 'Source not recorded');
const sourceTag = source => `<span class="tag ${String(source).includes('llm') ? 'rose' : ''}" title="${SR.esc(sourceLabel(source))}">${SR.esc(sourceLabel(source))}</span>`;

function molecule(smiles, label, route, stepIndex) {
  const expandedAt = (route.steps || []).findIndex((step, index) => index !== stepIndex && step.product === smiles);
  const terminal = (route.starting_materials || []).includes(smiles);
  return `<div class="molecule-card"><div class="board-label">${SR.esc(label)}</div><img src="${SR.mol(smiles)}" alt="Molecular structure of ${SR.esc(smiles)}"><div class="smiles">${SR.esc(smiles)}</div><div class="eyebrow-row" style="margin-bottom:0">${terminal ? '<span class="tag">Terminal material</span>' : expandedAt >= 0 ? `<a class="tag" href="#reaction-step-${expandedAt + 1}">Continues at step ${expandedAt + 1} ↗</a>` : ''}</div></div>`;
}

function evidence(step) {
  const observation = step.metadata?.tool_evidence;
  const structures = observation?.precursors || [];
  return `<details class="reaction-evidence" style="margin-top:8px"><summary>Tool observations ${observation ? `· ${structures.length} precursor${structures.length === 1 ? '' : 's'}` : '· not recorded for this result'}</summary>${observation ? `<div class="eyebrow-row"><span class="tag">Structure ${SR.esc(observation.structure_status || 'not recorded')}</span><span class="tag amber">${Number(observation.unresolved_precursors || 0)} unresolved at expansion</span></div>${structures.map((p, i) => `<div class="side-item" style="margin-bottom:10px"><div class="split-title"><b>Precursor ${i + 1}</b><span class="tag ${p.in_terminal_inventory ? '' : 'amber'}">${p.in_terminal_inventory ? 'In terminal inventory' : 'Requires expansion'}</span></div><div class="smiles" style="margin-top:8px">${SR.esc(p.smiles)}</div><div class="tool-summary"><div class="result-field"><small>Molecular weight</small><b>${number(p.molecular_weight, 2)}</b> <small>g/mol</small></div><div class="result-field"><small>Heavy atoms</small><b>${SR.esc(p.heavy_atoms ?? '—')}</b></div><div class="result-field"><small>C–H sites</small><b>${SR.esc(p.ch_sites ?? '—')}</b></div><div class="result-field"><small>Rings</small><b>${SR.esc(p.rings ?? '—')}</b></div><div class="result-field"><small>Formal charge</small><b>${SR.esc(p.formal_charge ?? '—')}</b></div></div></div>`).join('')}<p style="font-size:11px;color:var(--muted);margin:12px 0">${SR.esc(observation.source || 'Recorded tool observations')} · Forward validation: ${SR.esc((observation.forward_validation || 'not recorded').replaceAll('_', ' '))}.</p>` : '<p style="font-size:12px;color:var(--muted);margin-top:14px">Run a new search to attach molecular properties and inventory observations to each step.</p>'}</details>`;
}

function routeAssessment(route) {
  const assessment = current.route_assessment || {}, grade = route.llm_assessment;
  const labels = {chemical_plausibility:'反应合理性', strategic_coherence:'全局策略', route_efficiency:'路线效率', starting_material_suitability:'起始原料适合度'};
  const status = {not_configured:'LLM 未连接', disabled:'尚未评价', partial:'部分评价未完成', no_routes:'暂无完整候选路线'}[assessment.status] || '尚未评价';
  return `<section class="panel" id="whole-route-assessment" style="margin-top:18px"><div class="split-title"><h3>LLM 整条路线评价</h3><button type="button" class="button secondary small" id="evaluate-routes">${assessment.status==='complete'?'重新评价候选路线':'评价候选路线'}</button></div>${grade?.status==='complete'?`<div class="eyebrow-row" style="margin-top:16px"><span class="tag">路线总分 ${number(grade.total_score,2)} / 10</span>${grade.rank?`<span class="tag">LLM 排名 ${grade.rank}</span>`:''}${current.route_selection_policy==='whole_route_llm_v1'&&selectedRoute===0?'<span class="tag">Agent 推荐路线</span>':''}</div><p>${SR.esc(grade.rationale)}</p><div class="check-grid">${Object.entries(grade.scores).map(([key,value])=>`<article class="check-card"><b>${SR.esc(labels[key]||key)}</b><p>${number(value,1)} / 10</p></article>`).join('')}</div>${grade.strengths?.length?`<p><b>路线优势</b></p><ul>${grade.strengths.map(v=>`<li>${SR.esc(v)}</li>`).join('')}</ul>`:''}${grade.concerns?.length?`<p><b>需要关注</b></p><ul>${grade.concerns.map(v=>`<li>${SR.esc(v)}</li>`).join('')}</ul>`:''}${grade.critical_steps?.length?`<p><b>关键步骤</b></p><ul>${grade.critical_steps.map(v=>`<li>Step ${Number(v.step)} · ${SR.esc(v.concern)}</li>`).join('')}</ul>`:''}`:`<p class="field-note">${SR.esc(status)}</p>`}${assessment.status==='partial'?'<p class="field-note">评价未全部完成，当前保留原路线排序。</p>':''}<details class="reaction-evidence"><summary>评分依据与调用记录</summary><pre>${SR.esc(JSON.stringify({status:assessment.status,criteria_weights:assessment.criteria_weights,preferences:assessment.preferences,calls:assessment.calls,errors:assessment.errors,observation:(assessment.observations||[]).find(o=>o.route_index===(route.assessment_input_index??route.model_order_index??selectedRoute))},null,2))}</pre></details></section>`;
}

async function evaluateRoutes() {
  const button = SR.$('#evaluate-routes');
  button.disabled=true; button.textContent='正在评价整条路线…';
  try {
    const response=await SR.api('/api/tools/execute',{method:'POST',body:JSON.stringify({name:'evaluate_routes',arguments:{analysis_id:id,route_preferences:current.route_assessment?.preferences||''}})});
    location.href=response.result?.url||response.url||`/routes.html?id=${encodeURIComponent(response.result?.analysis_id||response.analysis_id)}`;
  } catch(error) {button.disabled=false;button.textContent='重新评价';const message=document.createElement('p');message.className='field-note';message.textContent=error.message;button.closest('section').append(message);}
}

function renderRoute(route, index) {
  const steps = route.steps || [];
  if (!steps.length) return '<div class="empty"><h3>The target is a starting material.</h3><p>It already belongs to the configured terminal inventory.</p></div>';
  return `<div class="split-title"><div><div class="kicker">Selected pathway</div><h3 style="font-size:28px;margin:14px 0 8px">Route ${index + 1}</h3><p style="color:var(--muted);font-size:12px;margin:0">${steps.length} step${steps.length === 1 ? '' : 's'} · ${route.starting_materials?.length || 0} starting materials · ${number(route.score)} search cost${route.all_edges_documented === true ? ' · Literature-matched' : route.all_edges_documented === false ? ' · Extra candidate: unverified edge' : ''}</p></div><button class="button ghost small" data-ask-agent="${SR.esc(`Review route ${index + 1} in analysis ${id}. Inspect its model provenance, inventory closure and C-H selectivity evidence.`)}">Ask the Agent ↗</button></div><div class="connected-route" id="connected-route"></div>${routeAssessment(route)}<section class="route-suppliers" id="route-suppliers"></section><section class="route-validation" id="route-validation"></section><details class="connected-step-details"><summary>Inspect individual reactions and provenance</summary>${steps.map((step, i) => `<article id="reaction-step-${i + 1}" style="padding:26px 0 18px;border-bottom:1px solid var(--line);scroll-margin-top:110px"><div class="split-title" style="gap:10px"><div class="kicker">Step ${String(i + 1).padStart(2, '0')}</div><div class="eyebrow-row" style="margin:0">${sourceTag(step.source)}<span class="tag amber">Model rank #${SR.esc(step.model_rank ?? '—')}</span></div></div><div class="reaction" style="border-bottom:0;padding-bottom:10px"><div>${molecule(step.product, 'Product', route, i)}</div><div class="arrow" aria-label="Retrosynthetic disconnection">→</div><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(140px,100%),1fr));gap:10px;min-width:0">${(step.precursors || []).map((p, j) => molecule(p, `Precursor ${j + 1}`, route, i)).join('')}</div></div>${evidence(step)}<details class="reaction-evidence" style="margin-top:14px"><summary>Prediction provenance · score ${number(step.model_score, 4)}</summary><pre>${SR.esc(JSON.stringify({source: step.source, model_rank: step.model_rank, model_score: step.model_score, metadata: step.metadata}, null, 2))}</pre></details></article>`).join('')}</details><div class="inline-prompt"><span>Inspect the target’s C–H environments alongside this route.</span><a href="/ch-activation.html?smiles=${encodeURIComponent(current.target)}" class="button secondary small">Open Site Lab ↗</a></div>`;
}

function selectRoute(index) {
  selectedRoute = index;
  SR.$$('#route-tabs .route-tab').forEach((node, i) => {
    node.classList.toggle('active', i === index);
    node.setAttribute('aria-pressed', String(i === index));
  });
  SR.$('#route-detail').innerHTML = renderRoute(current.routes[index], index);
  ConnectedRouteView.mount(SR.$('#connected-route'),current,index);
  ResultEvidence.routeChecks(SR.$('#route-validation'),id,index);
  RouteSuppliers.mount(SR.$('#route-suppliers'),id,index);
  const assessButton=SR.$('#evaluate-routes');if(assessButton)assessButton.onclick=evaluateRoutes;
}

function renderCandidates() {
  const candidates = current.one_step_candidates || [];
  const pageCount = Math.max(1, Math.ceil(candidates.length / PAGE_SIZE));
  candidatePage = Math.max(0, Math.min(candidatePage, pageCount - 1));
  const start = candidatePage * PAGE_SIZE;
  SR.$('#candidate-rows').innerHTML = candidates.slice(start, start + PAGE_SIZE).map((candidate, offset) => `<tr><td><b>#${start + offset + 1}</b></td><td class="smiles" style="min-width:210px">${(candidate.precursors || []).map(SR.esc).join(' . ')}</td><td style="white-space:nowrap">${SR.esc(candidate.metadata?.draw_frequency ?? candidate.metadata?.empirical_fragment_frequency_product ?? '—')} / ${SR.esc(candidate.metadata?.combination_slots ?? candidate.metadata?.draws ?? current.draws_per_product ?? '—')}</td><td>${number(candidate.score, 4)}</td><td>${sourceTag(candidate.source)}</td></tr>`).join('') || '<tr><td colspan="5"><div class="empty">No one-step candidates were returned for this target.</div></td></tr>';
  SR.$('#candidate-range').textContent = candidates.length ? `${start + 1}–${Math.min(start + PAGE_SIZE, candidates.length)} of ${candidates.length} candidates` : '0 candidates';
  SR.$('#candidate-page').textContent = `Page ${candidatePage + 1} of ${pageCount}`;
  SR.$('#candidate-prev').disabled = candidatePage === 0;
  SR.$('#candidate-next').disabled = candidatePage >= pageCount - 1;
}

function render(data) {
  current = data;
  window.SR_ANALYSIS = data;
  window.SR_ANALYSIS_ID = id;
  const gate = data.stage1 || {}, candidates = data.one_step_candidates || [], routes = data.routes || [];
  const industrial = data.mode === 'constrained_product_only_prototype';
  const fixedCore = data.mode === 'fixed_core_latest_candidates';
  const encodedId = encodeURIComponent(id);
  root.innerHTML = `<section class="surface-heading"><div class="split-title"><div><div class="kicker">Route intelligence · Saved analysis</div><h1>${data.solved ? 'Your paths, resolved.' : 'Explore the next move.'}</h1><p class="lead">${data.solved ? 'Compare model-generated pathways, inspect every step, and bring the evidence into the conversation.' : 'Inspect candidate coverage and the search record to guide the next planning step.'}</p></div><a class="button secondary small" href="/history.html">← Saved work</a></div><div class="smiles" style="max-width:850px;margin-bottom:22px">${SR.esc(data.target)}</div><div class="toolbar" style="margin-bottom:0"><button class="button lime" data-ask-agent="${SR.esc(`Review my current analysis ${id} for ${data.target}. Compare the routes and identify useful chemistry checks.`)}">✳ Discuss this analysis</button><a class="button secondary" href="/workspace.html?product=${encodeURIComponent(data.target)}">Plan again ↗</a><a class="button ghost small" href="/api/reports/${encodedId}?format=json" id="export-json">JSON ↓</a><a class="button ghost small" href="/api/reports/${encodedId}?format=markdown" id="export-markdown">Markdown ↓</a></div></section>
  <div class="metric-grid"><div class="metric"><strong>${routes.length}</strong><span>Resolved routes</span></div><div class="metric"><strong>${data.model_calls || 0}</strong><span>${fixedCore ? 'Candidate-bank queries' : 'Model expansions'}</span></div><div class="metric"><strong>${fixedCore ? data.provided_step_templates : industrial ? Number(data.generation_budget?.used_paired_draws || 0) : `${gate.hits ?? '—'}/${gate.samples ?? '—'}`}</strong><span>${fixedCore ? 'Case step templates' : industrial ? 'Paired generation draws' : 'Stage1 center hits'}</span></div><div class="metric"><strong>${candidates.length}</strong><span>One-step candidates</span></div><div class="metric"><strong>${data.seconds ?? '—'}<span style="font-size:16px;display:inline"> s</span></strong><span>Search time</span></div></div>
  <p style="font-size:11px;color:var(--muted);margin:16px 0 30px">Route completion is measured against the configured starting-material inventory. Search cost describes ranking; experimental feasibility remains to be assessed.</p>
  ${fixedCore ? `<div class="field-note">SpectraRetro <details class="reaction-evidence" style="margin-bottom:22px"><summary>Model and generation provenance</summary><pre>${SR.esc(JSON.stringify({weights_sha256:data.weights_sha256,generation:data.generation,scope:data.scope},null,2))}</pre></details></div>` : ''}<section class="result-layout" aria-label="Route comparison"><aside class="panel" style="min-width:0"><div class="kicker">Alternatives</div><h3 style="margin:12px 0 20px">${routes.length} pathway${routes.length === 1 ? '' : 's'} to explore</h3><div class="route-list" id="route-tabs" style="max-height:520px;overflow:auto">${routes.map((route, i) => `<button class="route-tab ${i === 0 ? 'active' : ''}" aria-pressed="${i === 0}" aria-controls="route-detail"><b>Route ${String(i + 1).padStart(2, '0')} <span style="float:right">↗</span></b><small>${(route.steps || []).length} step${route.steps?.length === 1 ? '' : 's'} · ${number(route.score)} cost</small><small>${(route.starting_materials || []).length} terminal materials</small>${route.all_edges_documented != null ? `<small>${route.all_edges_documented ? 'Literature-matched' : 'Extra candidate · unverified'}</small>` : ''}</button>`).join('') || '<div class="notice">No route closed within the current search budget.</div>'}</div><hr style="border:0;border-top:1px solid var(--line);margin:24px 0"><div class="side-item"><b>Search policy</b><small>${data.llm_policy === 'openai_compatible' ? 'External LLM candidate and frontier ranking' : 'Model frequency and search cost'}</small></div><div class="side-item" style="margin-top:10px"><b>Model evidence</b><small>Predictions and tool observations remain linked to each reaction.</small></div><a class="evidence-link" style="display:block;margin-top:22px" href="/history.html">Compare saved analyses ↗</a></aside><div class="panel" id="route-detail" style="min-width:0;align-self:start">${routes.length ? renderRoute(routes[0], 0) : '<div class="empty"><h3>No complete route yet.</h3><p>Inspect the candidates below, increase the search budget, or ask the Agent to review the available evidence.</p><button class="button secondary" data-ask-agent="Review why this analysis did not find a complete route and suggest a useful next action.">Review with the Agent ↗</button></div>'}</div></section>
  <section class="section" id="candidate-section"><div class="section-head"><div><div class="kicker">Single-step model evidence</div><h2 style="margin-top:12px">A closer look at the candidates.</h2></div><button class="button secondary small" id="export-csv">Candidate CSV ↓</button></div><p style="font-size:12px;color:var(--muted)">Generation frequencies and model scores are recorded ranking signals. Each source label identifies the origin of that candidate.</p><div class="panel matrix" style="padding:0"><table class="candidate-table"><caption class="sr-only">Ranked one-step precursor candidates</caption><thead><tr><th scope="col">Rank</th><th scope="col">Precursor set</th><th scope="col">${fixedCore ? 'Combination weight' : 'Frequency'}</th><th scope="col">Model score</th><th scope="col">Prediction source</th></tr></thead><tbody id="candidate-rows"></tbody></table></div><div class="pagination"><span id="candidate-range" aria-live="polite"></span><div class="toolbar"><span id="candidate-page"></span><button class="button secondary small" type="button" id="candidate-prev" aria-label="Previous candidate page">← Previous</button><button class="button secondary small" type="button" id="candidate-next" aria-label="Next candidate page">Next →</button></div></div></section>
  <section class="panel" style="min-width:0"><details class="reaction-evidence"><summary>Search record & computational budget</summary><div class="eyebrow-row"><span class="tag">${Number(data.llm_calls || 0)} LLM calls</span><span class="tag">${SR.esc((data.termination || 'recorded search').replaceAll('_', ' '))}</span></div><pre>${SR.esc(JSON.stringify({policy: data.policy || data.llm_policy, search_settings: data.search_settings, generation_budget: data.generation_budget, stock_source: data.stock_source || data.terminal_set, generation: data.generation_trace, search: data.trace, tool_observations: data.tool_observations, errors: data.errors}, null, 2))}</pre></details>${data.model_mode === 'known_class_agent' ? `<details class="reaction-evidence" style="margin-top:18px"><summary>Agent reaction-class hypotheses</summary><p style="font-size:12px;margin-top:14px">These are class hypotheses selected during search.</p><pre>${SR.esc(JSON.stringify(data.class_selection_trace || [], null, 2))}</pre></details>` : ''}</section>`;
  SR.$$('#route-tabs .route-tab').forEach((button, index) => button.onclick = () => selectRoute(index));
  if(routes.length)selectRoute(data.primary_route_index ?? Math.max(0,routes.findIndex(route=>route.all_edges_documented===true)));
  SR.$('#candidate-prev').onclick = () => { candidatePage--; renderCandidates(); };
  SR.$('#candidate-next').onclick = () => { candidatePage++; renderCandidates(); };
  SR.$('#export-csv').onclick = () => {
    const cell = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
    const rows = [['rank', 'score', 'frequency', 'draws', 'precursors', 'source'], ...candidates.map((candidate, index) => [index + 1, candidate.score, candidate.metadata?.draw_frequency, candidate.metadata?.draws, (candidate.precursors || []).join('.'), candidate.source])];
    SR.download(`CHRetroAgent-${id}-candidates.csv`, rows.map(row => row.map(cell).join(',')).join('\n'), 'text/csv');
  };
  renderCandidates();
  if(document.body.classList.contains('chat-route-mode')){const title=root.querySelector('.surface-heading h1');if(title)title.textContent=data.case_name||'Connected route';}
  window.dispatchEvent(new CustomEvent('sr:analysis-rendered',{detail:{analysis_id:id,session_id:data.conversation_session_id||null}}));
}

if (!id) root.innerHTML = '<div class="empty"><div class="kicker" style="justify-content:center">Route workspace</div><h2 style="margin-top:20px">Every pathway starts with a target.</h2><p>Start a planning task or open a report from your saved work.</p><div class="toolbar" style="justify-content:center"><a class="button" href="/workspace.html">Open the planner ↗</a><a class="button secondary" href="/history.html">Saved analyses</a></div></div>';
else SR.api(`/api/analyses/${encodeURIComponent(id)}`).then(record => render(record.result)).catch(error => root.innerHTML = `<div class="empty"><h2>This report could not be loaded.</h2><p>${SR.esc(error.message)}</p><div class="toolbar" style="justify-content:center"><a class="button" href="/history.html">Find a saved analysis</a><a class="button secondary" href="/workspace.html">Start a new search</a></div></div>`);
