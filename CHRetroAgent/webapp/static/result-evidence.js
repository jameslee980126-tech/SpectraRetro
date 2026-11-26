const ResultEvidence = (() => {
  let currentSession=null;
  const labels={passed:'Passed',failed:'Failed',partial:'Partial support',consistent:'Template consistent',not_checked:'Not checked',not_verified:'Not verified',complete:'Completed',error:'Error'};
  function badge(status){return `<span class="tag ${['passed','complete','consistent'].includes(status)?'':['failed','error'].includes(status)?'rose':'amber'}">${SR.esc(labels[status]||String(status||'Not recorded').replaceAll('_',' '))}</span>`;}
  async function routeChecks(host,analysisId,index){
    host.innerHTML='<h3>Route checks</h3><p class="field-note">Running structural and connectivity checks…</p>';
    try{const result=await SR.api(`/api/route-checks?analysis_id=${encodeURIComponent(analysisId)}&route_index=${index}`);host.innerHTML=`<h3>Route checks</h3><div class="check-grid">${result.checks.map(check=>`<article class="check-card">${badge(check.status)}<b>${SR.esc(check.name)}</b><p>${SR.esc(check.detail)}</p></article>`).join('')}</div>`;}catch(error){host.innerHTML=`<h3>Route checks unavailable</h3><p class="field-note">${SR.esc(error.message)}</p>`;}
  }
  function executions(calls){const host=SR.$('#tool-executions');if(!host)return;host.innerHTML=`<h3>Tool activity</h3>${(calls||[]).slice(-8).map(call=>`<div class="execution-row"><span class="execution-name">${SR.esc(call.name.replaceAll('_',' '))}</span>${badge(call.status)}</div>`).join('')||'<p class="field-note">No LLM tool executions are attached to this report.</p>'}`;}
  window.addEventListener('sr:conversation-rendered',event=>{currentSession=event.detail.session_id;executions(event.detail.tool_calls);});
  window.addEventListener('sr:analysis-rendered',async event=>{const id=event.detail.session_id;if(id&&id!==currentSession){try{const response=await SR.api('/api/sessions/'+encodeURIComponent(id));const session=response.session||response;executions(session.tool_calls);currentSession=id;}catch{executions([]);}}else if(!id&&!currentSession)executions([]);});
  return {routeChecks};
})();
