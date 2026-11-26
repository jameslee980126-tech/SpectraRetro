const ConnectedRouteView = (() => {
  let observer=null;
  function mount(host,result,index){
    observer?.disconnect();
    const url=`/api/route-diagram.svg?analysis_id=${encodeURIComponent(result.analysis_id||window.SR_ANALYSIS_ID)}&route_index=${index}`;
    host.innerHTML=`<div class="connected-toolbar"><span class="field-note">${result.routes[index].steps.length} reactions · molecular scheme · follow Step arrows</span><div class="toolbar"><button class="button secondary small diagram-fit" type="button">Fit width</button><button class="button secondary small diagram-all" type="button">Whole route</button><button class="button secondary small diagram-minus" type="button" aria-label="Zoom out route diagram">−</button><button class="button secondary small diagram-plus" type="button" aria-label="Zoom in route diagram">+</button><a class="button secondary small" href="${url}" target="_blank" rel="noopener">Open diagram ↗</a><a class="button ghost small" href="${url}" download="connected-route-${index+1}.svg">SVG ↓</a></div></div><div class="connected-viewport" tabindex="0" aria-label="Complete paper-style molecular route"><img class="connected-diagram" src="${url}" alt="Complete retrosynthesis route, wrapping left to right with every precursor branch retained"></div>`;
    const image=host.querySelector('img'),viewport=host.querySelector('.connected-viewport');let zoom=1,mode='all';
    function apply(value){zoom=Math.max(.08,Math.min(2,value));image.style.width=`${image.naturalWidth*zoom}px`;image.style.height='auto';}
    function fit(){if(!image.naturalWidth)return;const available=Math.max(260,Math.min(700,innerHeight-220));const width=(viewport.clientWidth-16)/image.naturalWidth;const ratio=mode==='all'?Math.min(1,width,(available-16)/image.naturalHeight):Math.min(1,width);apply(ratio);viewport.style.minHeight='160px';const height=Math.max(160,Math.min(available,image.naturalHeight*ratio+16));if(Math.abs(viewport.getBoundingClientRect().height-height)>1)viewport.style.height=`${height}px`;}
    image.onload=fit;image.onerror=()=>{viewport.innerHTML='<p class="notice">Diagram unavailable; the recorded reaction steps are available below.</p>';};
    host.querySelector('.diagram-fit').onclick=()=>{mode='width';fit();};host.querySelector('.diagram-all').onclick=()=>{mode='all';fit();viewport.scrollTop=0;viewport.scrollLeft=0;};
    host.querySelector('.diagram-minus').onclick=()=>{mode=null;apply(zoom/1.25);};host.querySelector('.diagram-plus').onclick=()=>{mode=null;apply(zoom*1.25);};
    observer=new ResizeObserver(()=>requestAnimationFrame(()=>{if(mode)fit();}));observer.observe(viewport);if(image.complete&&image.naturalWidth)fit();
  }
  return {mount};
})();
