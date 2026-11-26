SR.init('industrial');
const get = id => document.getElementById(id);
const lines = id => get(id).value.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
get('model-mode').addEventListener('change', () => {
  const known = get('model-mode').value === 'known_class_agent';
  if (known) get('policy').value = 'llm';
  get('policy').disabled = known;
});
get('industrial-form').addEventListener('submit', async event => {
  event.preventDefault(); get('submit').disabled=true;
  const request={target:get('target').value.trim(),stock:lines('stock'),stock_source:get('stock-source').value.trim(),
    constraints:{unavailable_stock:lines('unavailable'),forbidden_smiles:lines('forbidden'),require_forward_check:false},
    model_mode:get('model-mode').value,policy:get('policy').value,
    budgets:{max_paired_draws:Number(get('max-draws').value),max_steps:Number(get('max-steps').value),
      max_class_hypotheses:2,max_centers:get('model-mode').value==='known_class_agent'?4:3}};
  try {
    get('status').textContent='Submitting your planning task…';
    const {job_id}=await SR.api('/api/industrial/jobs',{method:'POST',body:JSON.stringify(request)});
    while(true){
      await new Promise(resolve=>setTimeout(resolve,1500));
      const job=await SR.api(`/api/jobs/${encodeURIComponent(job_id)}`);
      get('status').textContent=job.status==='queued'?'Waiting for the GPU worker…':'Generating model candidates and searching constrained routes…';
      if(job.status==='complete'){location.href=`/routes.html?id=${encodeURIComponent(job.analysis_id)}`;break;}
      if(job.status==='error')throw new Error(job.error);
    }
  } catch(error){get('status').textContent=error.message;get('submit').disabled=false;}
});
