'use strict';
const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = v => Number(v).toLocaleString(undefined,{maximumFractionDigits:Math.abs(Number(v))>0&&Math.abs(Number(v))<.01?6:2});
const score = v => (Number(v)*100).toFixed(1);
const signed = v => v !== 0 && Math.abs(v) < .0005 ? `${v>0?'+':'−'}<0.1` : `${v>0?'+':''}${(v*100).toFixed(1)}`;
const state={overview:null,city:'',page:0,total:0,items:[],detail:null,overrides:{},history:[],result:null,saved:[],model:'lightgbm',map:null,layer:null,geo:null,cityEpoch:0,selectionToken:0,revision:0,pending:false,controllers:new Map(),timer:null,searchTimer:null};
function notify(message){$('notice').querySelector('span').textContent=message;$('notice').hidden=false;}
function act(fn){return (...args)=>Promise.resolve().then(()=>fn(...args)).catch(err=>{if(err.name!=='AbortError')notify(err.message||'Unable to complete this action. Please retry.');});}
function cancel(key){state.controllers.get(key)?.abort();state.controllers.delete(key);}
async function api(path,{key,body,method='GET'}={}){
 if(state.overview?.browser_storage && path.startsWith('/api/scenarios')){
  const storageKey='study-atlas-scenarios-v1';
  let records=JSON.parse(localStorage.getItem(storageKey)||'[]');
  if(method==='GET')return records;
  if(method==='POST'){
   const result=await api('/api/predict',{method:'POST',body:{grid_id:body.grid_id,overrides:body.overrides,model:body.model}});
   const record={...result,id:crypto.randomUUID(),grid_id:body.grid_id,name:body.name,overrides:body.overrides,created:new Date().toISOString()};
   localStorage.setItem(storageKey,JSON.stringify([record,...records]));return record;
  }
  if(method==='DELETE'){records=records.filter(r=>r.id!==path.split('/').pop());localStorage.setItem(storageKey,JSON.stringify(records));return {deleted:true};}
 }
 let controller;
 if(key){cancel(key);controller=new AbortController();state.controllers.set(key,controller);}
 try{
  const response=await fetch(path,{method,signal:controller?.signal,headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});
  const data=await response.json();
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'The request contains an invalid value. Check your inputs.');
  return data;
 }finally{if(key && state.controllers.get(key)===controller)state.controllers.delete(key);}
}
function invalidate(){clearTimeout(state.timer);cancel('predict');cancel('sensitivity');cancel('optimize');$('optimizerResult').innerHTML='';state.revision++;$('sensitivityChart').innerHTML='';}
function showView(view){
 document.querySelectorAll('.view').forEach(e=>{const active=e.id===`view-${view}`;e.classList.toggle('active',active);e.hidden=!active;});
 document.querySelectorAll('.nav').forEach(e=>{const active=e.dataset.view===view;e.classList.toggle('active',active);if(active)e.setAttribute('aria-current','page');else e.removeAttribute('aria-current');});
 if(view==='explore')setTimeout(()=>state.map?.invalidateSize(),0);
 if(view==='saved')return refreshSaved();
}
function tab(name){
 document.querySelectorAll('[data-tab]').forEach(e=>{const yes=e.dataset.tab===name;e.classList.toggle('active',yes);e.setAttribute('aria-selected',String(yes));e.tabIndex=yes?0:-1;});
 document.querySelectorAll('.detail-tab').forEach(e=>{const yes=e.id===`tab-${name}`;e.classList.toggle('active',yes);e.hidden=!yes;});
}
function color(v){
 const t=Math.max(0,Math.min(1,Number(v))), a=t<.5?[255,232,148]:[235,90,56], b=t<.5?[235,90,56]:[115,0,0], f=t<.5?t*2:(t-.5)*2;
 return `rgb(${a.map((n,i)=>Math.round(n+(b[i]-n)*f)).join(',')})`;
}
function initMap(){
 if(!window.L){$('mapFallback').hidden=false;return;}
 state.map=L.map('map',{preferCanvas:true,zoomControl:true,scrollWheelZoom:false}).setView([38.8,-97],4);
 const tiles=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'&copy; OpenStreetMap contributors'}).addTo(state.map);
 let errors=0;tiles.on('tileerror',()=>{if(++errors>=5)$('mapFallback').hidden=false;});tiles.on('tileload',()=>{$('mapFallback').hidden=true;});
 state.layer=L.featureGroup().addTo(state.map);
}
function metric(p){
 const layer=$('layer').value;
 if(layer==='incidents')return p.incidents;
 if(layer==='scenario' && state.detail?.id===p.id && state.result && !state.pending)return state.result.score;
 return p.score;
}
function updateLegend(maxIncidents=0){
 const observed=$('layer').value==='incidents';
 $('legendTitle').textContent=observed?'Recorded incident count':'LightGBM baseline · 0–100';
 $('legendLow').textContent='0';$('legendMid').textContent=observed?num(Math.round(maxIncidents/2)):'50';$('legendHigh').textContent=observed?num(maxIncidents):'100';
}
function drawMap(fit=false){
 const schematic=!!state.city&&!!state.geo&&state.geo.features.some(f=>!f.properties.verified);
 $('schematic').hidden=!schematic;$('map').hidden=schematic;
 document.querySelector('.map-legend').hidden=schematic;
 if(schematic){
  $('mapFallback').hidden=true;
  const observed=$('layer').value==='incidents',features=[...state.geo.features].sort((a,b)=>b.properties.score-a.properties.score),max=Math.max(1,...features.map(f=>f.properties.incidents));
  $('schematic').innerHTML='<p class="schematic-intro"><strong>Study records · not a geographic map</strong><br>Yellow = lower score · dark red = higher. Choose a record to edit its variables. GIS boundaries are needed for accurate placement.</p><div class="record-grid">'+features.map(f=>{const p=f.properties,v=observed?p.incidents:metric(p),selected=p.id===state.detail?.id;return `<button class="record-tile ${selected?'selected':''}" data-record="${esc(p.id)}" style="--score-color:${color(observed?v/max:v)}" aria-pressed="${selected}"><strong>${esc(p.id)}</strong><span>${observed?num(v)+' incidents':score(v)+' / 100'}</span></button>`;}).join('')+'</div>';
  $('schematic').querySelectorAll('[data-record]').forEach(b=>b.addEventListener('click',act(()=>selectHex(b.dataset.record))));
  $('mapStatus').textContent=observed?'Recorded incidents':$('layer').value==='scenario'?`Selected: ${state.overview.models[state.model].name} scenario · others: LightGBM baseline`:'LightGBM baseline · 0–100';
  $('geometryNote').textContent='Unverified coordinates are not plotted. Record order and spacing do not represent location, distance, or adjacency.';
  document.querySelector('.geometry-note .badge').textContent='BOUNDARIES NEEDED';
  state.layer?.clearLayers();return;
 }
 if(!state.map)return;
 state.map.invalidateSize();state.layer.clearLayers();
 if(!state.city){
  const max=Math.max(...state.overview.cities.map(c=>c.incidents));updateLegend(max);
  for(const c of state.overview.cities){
   const observed=$('layer').value==='incidents';
   const marker=L.circleMarker([c.lat,c.lon],{radius:7+Math.sqrt(c.count)*.28,color:'#fff',weight:1.7,fillColor:color(observed?c.incidents/Math.max(1,max):c.score),fillOpacity:.94});
   marker.bindTooltip(`<strong>${esc(c.name)}</strong><br>${num(c.count)} study hexagons<br>${num(c.incidents)} recorded incidents<br>Mean model score ${score(c.score)} / 100<br><small>Approximate study center</small>`);
   marker.on('click',act(()=>chooseCity(c.name)));marker.addTo(state.layer);
  }
  if(fit)state.map.setView([38.8,-97],4);
  $('mapStatus').textContent='City summaries · LightGBM';return;
 }
 if(!state.geo)return;
 const max=Math.max(0,...state.geo.features.map(f=>f.properties.incidents));updateLegend(max);
 const style=f=>{const p=f.properties,selected=p.id===state.detail?.id;return {radius:selected?7:4,color:selected?'#102f38':'#fff',weight:selected?2.5:.6,fillColor:color($('layer').value==='incidents'?metric(p)/Math.max(max,1):metric(p)),fillOpacity:selected?1:.83};};
 const group=L.geoJSON(state.geo,{style,pointToLayer:(f,latlng)=>L.circleMarker(latlng,style(f)),onEachFeature:(f,lyr)=>{
  const p=f.properties, scenario=$('layer').value==='scenario' && p.id===state.detail?.id && state.result && !state.pending;
  lyr.bindTooltip(`<strong>${esc(p.id)}</strong><br>${scenario?'Scenario':'Baseline'} score ${score(metric({...p,incidents:p.incidents}))}${$('layer').value==='incidents'?' (incident layer; see count below)':' / 100'}<br>${num(p.incidents)} recorded incidents<br><small>${p.verified?'Supplied GIS boundary':'Estimated location · not a boundary'}</small>`);
  if($('layer').value==='incidents')lyr.setTooltipContent(`<strong>${esc(p.id)}</strong><br>${num(p.incidents)} recorded incidents<br>Baseline score ${score(p.score)} / 100<br><small>${p.verified?'Supplied GIS boundary':'Estimated location · not a boundary'}</small>`);
  lyr.on('click',act(()=>selectHex(p.id)));
 }}).addTo(state.layer);
 if(fit && group.getLayers().length)state.map.fitBounds(group.getBounds().pad(.12),{maxZoom:11,animate:false});
 const verified=state.geo.features.filter(f=>f.properties.verified).length;
 $('mapStatus').textContent=$('layer').value==='scenario'?'Selected hex: scenario · others: baseline':`${num(state.geo.features.length)} locations${verified?` · ${verified} GIS polygons`:''}`;
 $('geometryNote').textContent=verified===state.geo.features.length?'Displaying supplied original GIS polygons.':`${num(state.geo.features.length-verified)} estimated locations shown as circles, not hexagon boundaries. Do not interpret them as precise neighborhood locations.`;
 const badge=document.querySelector('.geometry-note .badge');badge.textContent=verified===state.geo.features.length?'SUPPLIED GIS GEOMETRY':'ESTIMATED GEOGRAPHY';
}
async function chooseCity(city,preserveSelection=false){
 state.city=city;$('city').value=city;state.page=0;state.geo=null;state.cityEpoch++;const epoch=state.cityEpoch;
 $('search').value='';
 if(!preserveSelection){state.selectionToken++;cancel('detail');invalidate();state.detail=null;state.result=null;state.overrides={};state.history=[];$('detail').hidden=true;$('emptyInspector').hidden=false;}
 $('mapTitle').textContent=city || '35 cities. One research workspace.';
 if(!city){cancel('map');$('geometryNote').textContent='City markers summarize the study. Hexagon locations are approximate; original GIS boundaries are not supplied.';document.querySelector('.geometry-note .badge').textContent='ESTIMATED GEOGRAPHY';drawMap(true);return loadList();}
 $('mapStatus').textContent='Loading locations…';state.layer?.clearLayers();
 await Promise.all([loadList(),(async()=>{const geo=await api(`/api/map/hexes?city=${encodeURIComponent(city)}`,{key:'map'});if(epoch!==state.cityEpoch)return;state.geo=geo;drawMap(true);})()]);
 if(!preserveSelection&&epoch===state.cityEpoch&&state.items.length)await selectHex(state.items[0].id);
}
async function loadList(){
 const q=$('search').value.trim(),sort=$('sort').value;
 if(!state.city && !q){
  cancel('list');let cities=[...state.overview.cities];cities.sort(sort==='id'?(a,b)=>a.name.localeCompare(b.name):sort==='incidents'?(a,b)=>b.incidents-a.incidents:(a,b)=>b.score-a.score);
  $('resultCount').textContent=`${cities.length} cities`;$('pagination').hidden=true;
  $('locations').innerHTML=cities.map(c=>`<button class="location" data-city="${esc(c.name)}"><i class="marker" style="background:${color(c.score)}"></i><span><span class="name">${esc(c.name)}</span><span class="sub">${num(c.count)} hexagons · ${num(c.incidents)} incidents</span></span><span class="right">${score(c.score)}<small>mean score</small></span></button>`).join('');
  $('locations').querySelectorAll('[data-city]').forEach(b=>b.addEventListener('click',act(()=>chooseCity(b.dataset.city))));return;
 }
 const epoch=state.cityEpoch,page=state.page;
 $('locations').innerHTML='<p class="loading">Loading locations…</p>';
 const params=new URLSearchParams({q,sort,offset:String(page*60),limit:'60'});if(state.city)params.set('city',state.city);
 const result=await api(`/api/hexes?${params}`,{key:'list'});if(epoch!==state.cityEpoch || page!==state.page)return;
 state.total=result.total;state.items=result.items;
 $('resultCount').textContent=`${num(result.total)} hexagons`;$('pagination').hidden=result.total<=60;
 $('pageCount').textContent=`${page+1} / ${Math.max(1,Math.ceil(result.total/60))}`;$('prev').disabled=page===0;$('next').disabled=(page+1)*60>=result.total;
 $('locations').innerHTML=result.items.map(h=>`<button class="location ${state.detail?.id===h.id?'active':''}" data-id="${esc(h.id)}"><i class="marker" style="background:${color(h.score)}"></i><span><span class="name">${esc(h.id)}</span><span class="sub">${state.city?`${num(h.incidents)} incidents`:esc(h.city)}${h.quality_flags.length?' · data flag':''}</span></span><span class="right">${score(h.score)}<small>score</small></span></button>`).join('') || '<p class="loading">No matching hexagons. Try another GRID_ID or city.</p>';
 $('locations').querySelectorAll('[data-id]').forEach(b=>b.addEventListener('click',act(()=>selectHex(b.dataset.id))));
}
async function selectHex(id){
 const token=++state.selectionToken;invalidate();state.detail=null;state.result=null;state.pending=true;$('detail').hidden=true;$('emptyInspector').hidden=false;
 const d=await api(`/api/hex/${encodeURIComponent(id)}`,{key:'detail'});if(token!==state.selectionToken)return;
 if(state.city!==d.city){await chooseCity(d.city,true);if(token!==state.selectionToken)return;}
 state.detail=d;state.overrides={};state.history=[];state.pending=false;
 const selectedBaseline=d.model_scores?.[state.model]??d.score;
 state.result={id:d.id,city:d.city,baseline:selectedBaseline,score:selectedBaseline,delta:0,changes:[],values:d.values,warnings:d.quality_flags,model:state.model,model_version:state.overview.model_version,data_version:state.overview.data_version};
 $('emptyInspector').hidden=true;$('detail').hidden=false;$('selectedCity').textContent=d.city;$('selectedId').textContent=d.id;
 $('splitBadge').textContent=d.split==='train'?'TRAINING ROW':'HELD-OUT ROW';populateModelPicker();$('baselineScore').textContent=score(d.model_scores?.[state.model]??d.score);$('incidents').textContent=num(d.incidents);$('population').textContent=num(d.population);renderModelScores();
 $('scenarioName').value='';renderControls();renderContext();renderDrivers();renderResult();tab('scenario');drawMap();
 document.querySelectorAll('.location[data-id]').forEach(b=>b.classList.toggle('active',b.dataset.id===d.id));
}
function renderControls(){
 const features=state.overview.features,groups={};
 Object.entries(features).forEach(([key,m])=>(groups[m.group]??=[]).push([key,m]));
 const editable=[];
 $('controls').innerHTML=Object.entries(groups).map(([group,entries],index)=>`<details class="control-group" ${index===0?'open':''}><summary>${esc(group)} <span>${entries.length} controls</span></summary><div class="control-group-body">${entries.map(([key,m])=>{
  const base=state.detail.values[key],value=state.overrides[key]??base,boundaryMin=Number(m.boundary_min??m.scenario_min??m.min),boundaryMax=Number(m.boundary_max??m.scenario_max??m.max),limit=Math.max(boundaryMax,base),floor=Math.min(boundaryMin,base),step=Number(m.step??(m.facility?1:.01)),isFacility=!!m.facility;
  editable.push([key,m]);
  return `<div class="control ${value!==base?'changed':''}" data-control="${esc(key)}"><div class="control-head"><label for="count-${esc(key)}">${esc(m.label)}</label><input id="count-${esc(key)}" type="number" min="${floor}" max="${limit}" step="${isFacility?1:'any'}" value="${value}" data-count="${esc(key)}" aria-describedby="hint-${esc(key)}"></div><label class="sr-only" for="range-${esc(key)}">${esc(m.label)} slider</label><input id="range-${esc(key)}" type="range" min="${floor}" max="${limit}" step="${step}" value="${value}" data-range="${esc(key)}"><div class="control-foot" id="hint-${esc(key)}"><span>Baseline ${num(base)} · ${esc(m.unit)}</span><span>Allowed ${num(floor)}–${num(limit)}${base<boundaryMin||base>boundaryMax?" · includes baseline":""}</span></div></div>`;
 }).join('')}</div></details>`).join('');
 $('sensitivityFeature').innerHTML=editable.map(([k,m])=>`<option value="${esc(k)}">${esc(m.label)}</option>`).join('');
 $('controls').querySelectorAll('[data-count]').forEach(input=>{input.addEventListener('input',()=>{if(!input.validity.valid||input.value===''){queuePrediction();return;}change(input.dataset.count,Number(input.value));});input.addEventListener('change',()=>{if(!input.validity.valid||input.value==='')input.reportValidity();});});
 $('controls').querySelectorAll('[data-range]').forEach(input=>input.addEventListener('input',()=>change(input.dataset.range,Number(input.value))));
}

function populateModelPicker(){
 const models=state.overview.models||{};
 $('modelSelect').innerHTML=Object.entries(models).map(([key,m])=>`<option value="${esc(key)}">${esc(m.name||key)}</option>`).join('');
 $('modelSelect').value=state.model;
 $('modelHelp').textContent=`${models[state.model]?.name||state.model} · F1 ${(models[state.model]?.metrics?.f1??0).toFixed(3)} · ROC-AUC ${(models[state.model]?.metrics?.roc_auc??0).toFixed(3)}`;
}
function renderModelScores(){
 const models=state.overview.models||{}, scores=state.detail?.model_scores||{};
 $('modelScores').innerHTML=Object.entries(models).map(([key,m])=>`<div class="model-row ${key===state.model?'selected':''}"><span>${esc(m.name||key)}</span><strong>${scores[key]==null?'—':score(scores[key])}</strong><small>F1 ${(m.metrics?.f1??0).toFixed(3)} · AUC ${(m.metrics?.roc_auc??0).toFixed(3)}</small></div>`).join('');
}
function changeModel(key){
 if(!state.overview.models?.[key]||state.model===key)return;
 state.model=key;populateModelPicker();renderModelScores();
 if(state.detail)$('baselineScore').textContent=score(state.detail.model_scores?.[key]??state.detail.score);
 if(state.detail){const baseline=state.detail.model_scores?.[key]??state.detail.score;state.result={...state.result,baseline,score:baseline,delta:0,model:key};renderResult();queuePrediction();}
}
function change(key,value){
 if(!state.detail)return;
 const old=state.overrides[key]??state.detail.values[key];if(old===value){queuePrediction();return;}
 state.history.push({...state.overrides});if(state.history.length>100)state.history.shift();
 if(value===state.detail.values[key])delete state.overrides[key];else state.overrides[key]=value;
 $(`count-${key}`).value=value;$(`range-${key}`).value=value;$(`count-${key}`).closest('.control').classList.toggle('changed',value!==state.detail.values[key]);
 queuePrediction();
}
function queuePrediction(){
 invalidate();state.pending=true;renderResult();drawMap();
 if([...document.querySelectorAll('[data-count]')].some(input=>input.value==='' || !input.validity.valid)){
  $('scenarioScore').textContent='—';$('delta').textContent='Enter a valid number';return;
 }
 state.timer=setTimeout(act(predict),160);
}
async function predict(){
 if(!state.detail)return;
 const token=state.selectionToken,revision=state.revision,id=state.detail.id;
 try{const result=await api('/api/predict',{method:'POST',key:'predict',body:{grid_id:id,overrides:{...state.overrides},model:state.model}});
  if(token!==state.selectionToken || revision!==state.revision)return;
  state.result=result;state.pending=false;renderResult();drawMap();
 }catch(err){if(err.name==='AbortError')return;if(token===state.selectionToken && revision===state.revision){state.pending=true;$('scenarioScore').textContent='Error';$('delta').textContent='Adjust or reset to retry';}throw err;}
}
function renderResult(){
 if(!state.detail)return;
 const count=Object.keys(state.overrides).length;
 $('undo').disabled=!state.history.length;$('reset').disabled=!count && !state.pending;$('save').disabled=state.pending;$('export').disabled=state.pending;$('runSensitivity').disabled=state.pending;$('runOptimize').disabled=state.pending||state.controllers.has('optimize');
 $('changeCount').textContent=count?`${count} changed ${count===1?'input':'inputs'}`:'Baseline · no changes';
 $('scenarioScore').textContent=state.pending?'…':score(state.result.score);
 $('delta').textContent=state.pending?'Updating model…':count?`${signed(state.result.delta)} score points`:'No changes';
 $('changes').innerHTML=count?`<div class="change-summary"><strong>Current assumptions</strong>${Object.entries(state.overrides).map(([k,v])=>`<div><span>${esc(state.overview.features[k].label)}</span><span>${num(state.detail.values[k])} → <b>${num(v)}</b></span></div>`).join('')}</div>`:'';
 $('scenarioWarnings').innerHTML=state.pending?'':(state.result.warnings||[]).map(w=>`<div class="warning">${esc(w)}</div>`).join('');
}
function reset(){if(!state.detail)return;state.history.push({...state.overrides});state.overrides={};invalidate();state.pending=false;state.result={...state.result,baseline:state.detail.model_scores?.[state.model]??state.detail.score,score:state.detail.model_scores?.[state.model]??state.detail.score,delta:0,changes:[],values:{...state.detail.values},warnings:state.detail.quality_flags};renderControls();renderResult();drawMap();}
function undo(){if(!state.history.length)return;state.overrides=state.history.pop();renderControls();queuePrediction();}
function renderContext(){
 const features=state.overview.features,groups=[...new Set(Object.values(features).map(m=>m.group))];
 $('qualityWarnings').innerHTML=state.detail.quality_flags.map(w=>`<div class="warning">${esc(w)}</div>`).join('');
 $('context').innerHTML=groups.map(g=>`<section class="context-group"><h3>${esc(g)}</h3>${Object.entries(features).filter(([,m])=>m.group===g).map(([k,m])=>`<details class="context-row"><summary><span>${esc(m.label)}</span><strong>${num(state.detail.values[k])}<small>${esc(m.unit)}</small></strong></summary><p>${esc(m.note)}</p><p>Source field: <code>${esc(k)}</code></p></details>`).join('')}</section>`).join('');
}
function renderDrivers(){
 const d=state.detail,max=Math.max(...d.contributions.map(x=>Math.abs(x.contribution)),.001);
 $('drivers').innerHTML=d.contributions.map(c=>`<div class="driver"><div><span>${esc(c.label)}</span><span>${c.contribution>0?'+':''}${c.contribution.toFixed(3)}</span></div><div class="driver-track"><i style="width:${Math.abs(c.contribution)/max*48}%;left:${c.contribution<0?50-Math.abs(c.contribution)/max*48:50}%;background:${c.contribution<0?'#80b8a5':'#315e69'}"></i></div></div>`).join('');
 $('percentile').textContent=`The LightGBM baseline is at the ${d.city_percentile.toFixed(1)}th percentile among study hexagons in ${d.city}. This is a within-study rank, not an estimate of individual risk.`;
}
function lineChart(points,{xLabel='Facility count',yLabel='Model score',reference=false}={}){
 const w=380,h=215,left=44,right=18,top=22,bottom=42,xmin=Math.min(...points.map(p=>p.x)),xmax=Math.max(...points.map(p=>p.x));
 const xx=x=>left+(x-xmin)/(xmax-xmin||1)*(w-left-right),yy=y=>h-bottom-y*(h-top-bottom);
 let svg=`<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(yLabel)} by ${esc(xLabel)}. Values also available below.">`;
 for(const y of [0,.25,.5,.75,1])svg+=`<line x1="${left}" x2="${w-right}" y1="${yy(y)}" y2="${yy(y)}" stroke="#dce6e1"/><text x="${left-7}" y="${yy(y)+3}" text-anchor="end">${y*100}</text>`;
 if(reference)svg+=`<line x1="${xx(0)}" y1="${yy(0)}" x2="${xx(1)}" y2="${yy(1)}" stroke="#b79b63" stroke-dasharray="4 4"/>`;
 svg+=`<polyline points="${points.map(p=>`${xx(p.x)},${yy(p.y)}`).join(' ')}" fill="none" stroke="#087e83" stroke-width="2"/>`;
 points.forEach((p,i)=>{svg+=`<circle cx="${xx(p.x)}" cy="${yy(p.y)}" r="3.5" fill="#087e83"><title>${num(p.x)}: ${score(p.y)}</title></circle>`;if(points.length<=12 && (!reference || i%2===0))svg+=`<text x="${xx(p.x)}" y="${h-bottom+18}" text-anchor="middle">${reference?(p.x*100).toFixed(0):num(p.x)}</text>`;});
 return svg+`<text x="${w/2}" y="${h-5}" text-anchor="middle">${esc(xLabel)}</text><text x="${left}" y="12">${esc(yLabel)} · 0–100</text></svg>`;
}
async function sensitivity(){
 if(!state.detail || state.pending)return;
 const token=state.selectionToken,revision=state.revision,feature=$('sensitivityFeature').value;
 $('sensitivityChart').innerHTML='<p class="loading">Calculating model response…</p>';
 const data=await api('/api/sensitivity',{method:'POST',key:'sensitivity',body:{grid_id:state.detail.id,overrides:{...state.overrides},feature,model:state.model}});
 if(token!==state.selectionToken || revision!==state.revision)return;
 $('sensitivityChart').innerHTML=lineChart(data.points.map(p=>({x:p.value,y:p.score})),{xLabel:state.overview.features[feature].label})+`<p class="chart-caption">${esc(data.note)} Vertical axis always spans 0–100.</p><details><summary class="helper">View response values</summary><table class="response-table"><tr><th>Value</th><th>Model score</th></tr>${data.points.map(p=>`<tr><td>${p.value}</td><td>${score(p.score)}</td></tr>`).join('')}</table></details>`;
}
async function runOptimize(){
 if(!state.detail||state.pending)return;
 invalidate();const token=state.selectionToken,revision=state.revision,model=state.model; $('runOptimize').disabled=true; $('optimizerResult').innerHTML='<p class="loading">Calculating…</p>';
 try{const result=await api('/api/optimize',{method:'POST',key:'optimize',body:{grid_id:state.detail.id,model:state.model,max_changes:22}}); if(token!==state.selectionToken||revision!==state.revision||model!==state.model)return;
  state.history.push({...state.overrides});state.overrides={...result.overrides};state.result=result.result; renderControls(); renderResult(); drawMap();
  $('optimizerResult').innerHTML=`<div class="optimizer-result"><strong>Best score found: ${score(result.optimized_score)}</strong><span>${signed(result.delta)} score points · ${result.changes.length} changes</span>${result.changes.length?`<div class="optimizer-changes">${result.changes.map(c=>`<div><span>${esc(c.label)}</span><span>${num(c.before)} → <b>${num(c.after)}</b></span></div>`).join('')}</div>`:'<p class="helper">No tested change lowered the score.</p>'}<p class="helper">${esc(result.interpretation)}</p></div>`;
  tab('optimizer');
 }finally{$('runOptimize').disabled=false;}
}
function download(name,data){const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportCurrent(){if(!state.detail || state.pending)return;download(`scenario-${state.detail.id}.json`,{name:$('scenarioName').value||'Unsaved scenario',grid_id:state.detail.id,exported:new Date().toISOString(),...state.result,overrides:state.overrides,period:state.overview.period,interpretation:'Uncalibrated model sensitivity, not causal intervention effects or a future forecast.',sources:state.overview.sources});}
async function saveScenario(){
 if(!state.detail || state.pending)return;
 const name=$('scenarioName').value.trim();if(!name){$('scenarioName').focus();notify('Give this scenario a name before saving.');return;}
 const body={name,grid_id:state.detail.id,overrides:{...state.overrides},model:state.model};
 $('save').disabled=true;
 try{await api('/api/scenarios',{method:'POST',body});await refreshSaved();notify(`Saved “${name}” in the local scenario library.`);}finally{if(state.detail)$('save').disabled=state.pending;}
}
async function refreshSaved(){
 state.saved=await api('/api/scenarios',{key:'saved'});$('savedCount').textContent=state.saved.length;$('savedEmpty').hidden=state.saved.length>0;
 $('savedTable').innerHTML=state.saved.length?`<div class="table-wrap"><table class="data-table"><thead><tr><th>Scenario / location</th><th>Baseline</th><th>Scenario</th><th>Change</th><th>Assumptions</th><th>Actions</th></tr></thead><tbody>${state.saved.map(s=>`<tr><td><strong>${esc(s.name)}</strong><small>${esc(s.city)} · ${esc(s.grid_id)}</small><small>${esc(new Date(s.created).toLocaleString())}</small><small>${esc(state.overview.models?.[s.model]?.name||s.model||'LightGBM')} · ${esc(s.model_version)}${s.model_version!==state.overview.model_version || s.data_version!==state.overview.data_version?' · different version':''}</small></td><td>${score(s.baseline)}</td><td>${score(s.score)}</td><td>${signed(s.delta)}<small>score points</small></td><td class="changes-cell">${s.changes.map(c=>`${esc(c.label)}: ${num(c.before)} → ${num(c.after)}`).join('<br>')||'Unchanged baseline'}</td><td><button class="small" data-open="${s.id}">Open copy</button><button class="small" data-export="${s.id}">Export</button><button class="text-button" data-delete="${s.id}">Delete</button></td></tr>`).join('')}</tbody></table></div>`:'';
 $('savedTable').querySelectorAll('[data-open]').forEach(b=>b.addEventListener('click',act(()=>openSaved(b.dataset.open))));
 $('savedTable').querySelectorAll('[data-export]').forEach(b=>b.addEventListener('click',()=>download(`saved-scenario-${b.dataset.export}.json`,{...state.saved.find(s=>s.id===b.dataset.export),interpretation:'Uncalibrated model sensitivity, not a forecast or a causal effect.'})));
 $('savedTable').querySelectorAll('[data-delete]').forEach(b=>b.addEventListener('click',act(async()=>{const s=state.saved.find(x=>x.id===b.dataset.delete);if(!window.confirm(`Delete saved scenario “${s.name}”?`))return;await api(`/api/scenarios/${b.dataset.delete}`,{method:'DELETE'});await refreshSaved();})));
}
async function openSaved(id){
 const s=state.saved.find(x=>x.id===id);if(!s)return;
 if(s.model_version!==state.overview.model_version || s.data_version!==state.overview.data_version){notify('This snapshot uses a different model or data version. Export it for review; reopening with the current version is disabled.');return;}
 state.model=s.model||'lightgbm';showView('explore');await selectHex(s.grid_id);if(state.detail?.id!==s.grid_id)return;
 state.overrides={...s.overrides};state.history=[];$('scenarioName').value=s.name+' · copy';renderControls();queuePrediction();
}
function renderEvidence(){
 const o=state.overview,e=o.evaluation;
 $('evidence').innerHTML=`<div class="evidence-grid">
 <article class="evidence-card"><p class="eyebrow">01 / OBSERVATIONS</p><h2>A fixed study snapshot.</h2><div class="metric-strip"><div><strong>${num(o.n_hexes)}</strong><span>hexagons</span></div><div><strong>${o.n_cities}</strong><span>cities</span></div><div><strong>${num(o.total_incidents)}</strong><span>recorded incidents</span></div></div><p>The outcome is whether a hexagon has at least one recorded incident in <code>Total_Gun_Violence</code>. The model does not predict incident counts or future events.</p><p>${esc(o.period)}</p><p>All 22 original-unit transformations were verified against the project’s VIF-clean table. Baseline scores were checked against the saved model for every row.</p></article>
 <article class="evidence-card"><p class="eyebrow">02 / MODEL</p><h2>Useful signals. Explicit limits.</h2><div class="metric-strip"><div><strong>${e.test_f1_at_0_50.toFixed(3)}</strong><span>F1 at threshold 0.50</span></div><div><strong>${e.test_roc_auc.toFixed(3)}</strong><span>ROC-AUC</span></div><div><strong>${e.average_precision.toFixed(3)}</strong><span>average precision</span></div></div><p>LightGBM · ${e.n_features} features · ${num(e.n_train)} training rows · ${num(e.n_test)} held-out rows. Precision ${e.precision.toFixed(3)}; recall ${e.recall.toFixed(3)}.</p><p>${esc(e.note)}</p><p>The interface displays 100 × model output as a score. A score of 70 is not an established 70% chance of a future incident. Training-row scores are identified in the inspector.</p></article>
 <article class="evidence-card"><p class="eyebrow">03 / CALIBRATION CHECK</p><h2>Scores versus observed frequency.</h2>${lineChart(e.calibration.map(p=>({x:p.mean_score,y:p.observed_fraction})),{xLabel:'Mean model score',yLabel:'Observed fraction',reference:true})}<p class="chart-caption">Eight equal-frequency bins on the reconstructed holdout. Dashed line indicates perfect calibration. Brier score: ${e.brier.toFixed(3)}. Holdout prevalence: ${(e.prevalence*100).toFixed(1)}%.</p><details><summary class="helper">View calibration values</summary><table class="response-table"><tr><th>Mean score</th><th>Observed fraction (%)</th></tr>${e.calibration.map(p=>`<tr><td>${score(p.mean_score)}</td><td>${score(p.observed_fraction)}</td></tr>`).join('')}</table></details></article>
 <article class="evidence-card"><p class="eyebrow">04 / GEOGRAPHIC TRANSFER</p><h2>Location changes performance.</h2><table class="data-table"><thead><tr><th>Evaluation</th><th>LightGBM F1</th></tr></thead><tbody>${o.regional_evidence.map(r=>`<tr><td>${esc(r.split)}</td><td>${r.f1.toFixed(3)}</td></tr>`).join('')}</tbody></table><p>The city and regional results come from the project’s separate holdout experiments. They are contextual evidence, not evaluations of this exact saved model. Source: tests/holdout_comparison/COMPARISON.md.</p></article>
 <article class="evidence-card"><p class="eyebrow">05 / DATA QUALITY</p><h2>Preserve the data. Surface the questions.</h2><p><strong>${num(o.quality.zero_population)}</strong> rows have zero recorded population. The original pipeline encodes undefined population percentages as zero.</p><p><strong>${o.quality.poverty_above_100}</strong> rows have a derived poverty percentage above 100%. These values are preserved to reproduce the saved model, and affected hexagons carry data flags.</p><p><code>Park</code> and <code>Parks</code> are distinct source columns without a documented relationship. They are separate scenario inputs; their relationship needs source review. Other ambiguous measures are labeled “source units”.</p></article>
 <article class="evidence-card"><p class="eyebrow">06 / GEOMETRY</p><h2>Honest about where.</h2><p>${num(o.geometry.verified_count)} supplied GIS polygons · ${num(o.geometry.estimated_count)} estimated locations.</p><p>${esc(o.geometry.note)}</p><p>The earlier reconstructed polygons are not displayed. To enable original boundaries, place a WGS84 GeoJSON export with unique <code>GRID_ID</code> properties at <code>digital_twin/artifacts/authoritative_hexes.geojson</code> and restart. Cities with complete verified geometry use the supplied polygons; cities with missing geometry use the nonspatial record view.</p></article>
 <article class="evidence-card wide"><p class="eyebrow">07 / REPRODUCIBILITY</p><h2>A traceable model and data record.</h2><p>Model version <code>${esc(o.model_version)}</code> · data version <code>${esc(o.data_version)}</code>. ${o.browser_storage?'Scenarios are saved in this browser; export them to share or back them up.':'Scenarios are stored locally in the digital twin database.'} No original research tables are edited.</p><table class="data-table"><thead><tr><th>Source / artifact</th><th>SHA-256 fingerprint</th></tr></thead><tbody>${Object.entries(o.sources).map(([name,m])=>`<tr><td>${esc(name)}</td><td class="hash">${esc(m.sha256)}</td></tr>`).join('')}</tbody></table><p>Model drivers are baseline TreeSHAP contributions in log-odds. Scenarios can change all 22 inputs within their allowed study ranges; facility counts use whole numbers. These are sensitivity experiments; they do not account for costs, timing, spillovers, or causal intervention effects.</p></article></div>`;
}
async function boot(){
 $('notice').querySelector('button').onclick=()=>$('notice').hidden=true;
 document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',act(()=>showView(b.dataset.view))));
 document.querySelectorAll('[data-tab]').forEach((b,i,all)=>{b.addEventListener('click',()=>tab(b.dataset.tab));b.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const index=e.key==='Home'?0:e.key==='End'?all.length-1:(i+(e.key==='ArrowRight'?1:-1)+all.length)%all.length;tab(all[index].dataset.tab);all[index].focus();}});});
 state.overview=await api('/api/overview');const o=state.overview;
 $('statCities').textContent=o.n_cities;$('statHexes').textContent=num(o.n_hexes);$('statIncidents').textContent=num(o.total_incidents);$('period').textContent=o.period;
 document.querySelector('.local-badge').textContent='RESEARCH WORKSPACE';document.querySelector('.save-box p').textContent=o.browser_storage?'Saved in this browser. Export to share or back up.':'Saved in the local digital twin database.';
 $('version').textContent=`MODEL ${o.model_version} / DATA ${o.data_version}`;
 $('city').innerHTML='<option value="">All 35 cities</option>'+o.cities.map(c=>`<option value="${esc(c.name)}">${esc(c.name)} · ${num(c.count)} hexagons</option>`).join('');
 $('city').addEventListener('change',act(()=>chooseCity($('city').value)));
 $('allCities').addEventListener('click',act(()=>chooseCity('')));
 $('search').addEventListener('input',()=>{clearTimeout(state.searchTimer);state.page=0;cancel('list');state.searchTimer=setTimeout(act(loadList),180);});
 $('sort').addEventListener('change',act(()=>{state.page=0;return loadList();}));
 $('prev').addEventListener('click',act(()=>{state.page--;return loadList();}));$('next').addEventListener('click',act(()=>{state.page++;return loadList();}));
 $('layer').addEventListener('change',()=>drawMap());$('modelSelect').addEventListener('change',e=>changeModel(e.target.value));$('reset').addEventListener('click',reset);$('undo').addEventListener('click',undo);
 $('save').addEventListener('click',act(saveScenario));$('export').addEventListener('click',exportCurrent);$('runSensitivity').addEventListener('click',act(sensitivity));$('runOptimize').addEventListener('click',act(runOptimize));
 $('sensitivityFeature').addEventListener('change',()=>{cancel('sensitivity');$('sensitivityChart').innerHTML='';});
 $('exportLibrary').addEventListener('click',()=>download('study-atlas-scenarios.json',{exported:new Date().toISOString(),interpretation:'Uncalibrated model sensitivity, not causal effects or forecasts.',scenarios:state.saved}));
 initMap();renderEvidence();await chooseCity('');await refreshSaved();tab('scenario');
}
act(boot)();
