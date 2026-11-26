SR.init('plan');
const productInput = SR.$('#product');
const preview=SR.$('#target-preview');
let fixedCases=[];
function selectedFixedCase(){return fixedCases.find(item=>item.product===productInput.value.trim());}
function updateCaseMode(){const live=SR.$('input[name=mode][value=live]');live.disabled=!!selectedFixedCase();if(live.disabled)SR.$('input[name=mode][value=history]').checked=true;}
function updatePreview(){if(preview)preview.src=productInput.value.trim()?SR.mol(productInput.value.trim()):'/molecule-placeholder.svg';}
productInput.addEventListener('change',()=>{updatePreview();updateCaseMode();});
productInput.addEventListener('input',()=>{clearTimeout(updatePreview.timer);updatePreview.timer=setTimeout(updatePreview,500);});
const queryProduct = new URLSearchParams(location.search).get('product');
if (queryProduct) productInput.value = queryProduct;
SR.api('/api/meta').then(meta => {
  fixedCases=meta.fixed_core_cases||[];
  if (!productInput.value && meta.examples?.length) productInput.value = meta.examples[0];
  updatePreview();
  SR.$('#examples').innerHTML = [...fixedCases,...(meta.showcases || []).slice(0,4)].map(item => `<button type="button" class="route-tab" data-product="${SR.esc(item.product)}"><b>${SR.esc(item.title)}</b><small>${SR.esc(item.product)}</small></button>`).join('');
  SR.$$('#examples [data-product]').forEach(button => button.onclick = () => { productInput.value = button.dataset.product; updatePreview(); updateCaseMode(); productInput.focus(); });
  updateCaseMode();
}).catch(error=>SR.toast(error.message));
const stageNames = {queued:'Task queued',loading_model:'Loading models',sampling:'Generating Stage2 candidates',decoding:'Recovering chemical structures',planning:'Searching multistep routes',assessing_routes:'Evaluating complete routes with the LLM',complete:'Complete'};
SR.$('#plan-form').addEventListener('submit', async event => {
  event.preventDefault(); const button = SR.$('#run-button'); button.disabled = true;
  const mode = SR.$('input[name=mode]:checked').value;
  const request = {product: productInput.value.trim(), top_k:+SR.$('#top-k').value, max_depth:+SR.$('#depth').value, max_routes:+SR.$('#routes').value};
  const progress = SR.$('#progress-wrap'); progress.classList.add('show');
  try {
    if (selectedFixedCase()) {
      SR.$('#progress-label').textContent = 'Connecting latest case candidates'; SR.$('#progress-bar').style.width='48%';
      const result=await SR.api('/api/fixed-core/plan',{method:'POST',body:JSON.stringify({case_id:selectedFixedCase().case_id,max_routes:request.max_routes})});
      location.href=`/routes.html?id=${encodeURIComponent(result.analysis_id)}`;
    } else if (mode === 'history') {
      SR.$('#progress-label').textContent = 'Searching candidate routes'; SR.$('#progress-bar').style.width = '48%';
      const result = await SR.api('/api/analyses', {method:'POST', body:JSON.stringify(request)});
      SR.$('#progress-bar').style.width = '100%'; location.href = `/routes.html?id=${encodeURIComponent(result.analysis_id)}`;
    } else {
      Object.assign(request, {draws:300, steps:100, max_model_calls:3, seed:0});
      const created = await SR.api('/api/jobs', {method:'POST', body:JSON.stringify(request)});
      while (true) {
        await new Promise(resolve => setTimeout(resolve, 1200));
        const job = await SR.api(`/api/jobs/${created.job_id}`); const fraction = Math.round((job.fraction || 0) * 100);
        SR.$('#progress-label').textContent = stageNames[job.stage] || job.stage; SR.$('#progress-value').textContent = `${fraction}%`; SR.$('#progress-bar').style.width = `${fraction}%`;
        if (job.status === 'complete') { location.href = `/routes.html?id=${encodeURIComponent(job.analysis_id)}`; break; }
        if (job.status === 'error') throw new Error(job.error || 'Live inference failed');
      }
    }
  } catch (error) { SR.toast(error.message); button.disabled = false; SR.$('#progress-label').textContent = 'Task did not complete'; }
});
