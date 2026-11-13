'use strict';
const $ = id => document.getElementById(id);
const state = {data:null, catalog:[], match:null, plan:null, ranked:[], uploads:[], busy:0};
const names = {match:'Match analysis',ranking:'Candidate ranking',learning:'Learning path',experiments:'Experiments',catalog:'Resource catalog'};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = value => (Number(value || 0)*100).toFixed(0)+'%';
const score = value => value == null ? '—' : Number(value).toFixed(1);
const human = text => String(text).replaceAll('_',' ');
const chips = (items, cls='') => '<div class="chips">'+items.map(s=>`<span class="chip ${cls}" data-i18n-ignore>${esc(s)}</span>`).join('')+'</div>';
const eligibility = text => ({PENDING_VERIFICATION:'Pending human review',ELIGIBLE:'Verified requirements met',NOT_QUALIFIED:'Mandatory requirements not met',NO_MANDATORY_CONSTRAINTS:'No mandatory constraints',NO_REQUIREMENTS_EXTRACTED:'No requirements extracted'})[text] || human(text || 'Review required');
function toast(message,error=false){$('toast').textContent=message;$('toast').className=error?'error':'';$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,error?9000:4500);}
async function api(path, data){
  if(window.SKILLBRIDGE_OFFLINE)return offlineApi(path,data);
  const response=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const value=await response.json();if(!response.ok)throw new Error(value.error||`Request failed (${response.status})`);return value;
}
function offlineApi(path,data){
  const bundle=window.SKILLBRIDGE_OFFLINE;
  if(path==='/api/showcase')return bundle.bootstrap;
  if(path==='/api/catalog')return bundle.catalog;
  if(path==='/api/export')return {markdown:['# SkillBridge learning plan','',data.plan.backend,'',...data.plan.recommendations.flatMap(c=>['## '+c.step+'. '+c.title,c.description,c.url,'Targets: '+c.covered_gaps.join(', '),''])].join('\n')};
  throw new Error('Run Start-Showcase.cmd and open http://127.0.0.1:8877 to analyze your documents. This standalone file only previews the interface.');
}
async function action(button,label,fn){
  if(state.busy){toast('An operation is in progress. Please wait.');return;}
  state.busy++;const old=button.textContent;button.disabled=true;button.textContent=label;button.classList.add('loading');
  try{await fn();}catch(error){toast(error.message,true);}finally{state.busy--;button.disabled=false;button.textContent=old;button.classList.remove('loading');}
}
function navigate(){
  const view=location.hash.slice(1) in names?location.hash.slice(1):'match';
  document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!=='view-'+view);
  document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.dataset.view===view));
  $('breadcrumb').textContent=names[view];
  if(view==='ranking')$('rank-jd').value=$('jd').value;
}
window.addEventListener('hashchange',navigate);navigate();
function counts(){for(const id of ['jd','resume'])$(id+'-count').textContent=$(id).value.length.toLocaleString()+' characters';}
function invalidate(){
  state.match=null;$('bridge-learning').disabled=true;$('match-score').textContent='—';$('coverage-score').textContent='—';$('eligibility').textContent='Not analyzed';$('score-ring').style.setProperty('--score',0);$('evidence-count').textContent='0';
  $('score-description').textContent='Documents are ready. Select Analyze match to continue.';
  $('evidence').className='empty';$('evidence').innerHTML='<span class="empty-symbol">⌕</span><h3>Ready to analyze these documents</h3><p>Results will include scores, requirement categories and original evidence.</p>';
  $('skill-gaps').innerHTML='<p class="muted">Analyze documents to identify skills without sufficient resume evidence.</p>';counts();
}
['jd','resume'].forEach(id=>$(id).addEventListener('input',invalidate));
async function readFile(file){
  if(file.size>5*1024*1024)throw new Error('A single file cannot exceed 5 MB.');
  if(file.name.toLowerCase().endsWith('.txt')){
    const text=(await file.text()).replace(/^\uFEFF/,'');
    if(!text.trim())throw new Error('The document is empty.');
    if(text.length>40000)throw new Error('Document exceeds 40000 characters.');
    return text;
  }
  const content=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=()=>reject(new Error('Unable to read the file.'));r.readAsDataURL(file);});
  return (await api('/api/read-document',{name:file.name,content})).text;
}
for(const id of ['jd','resume'])$(id+'-file').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{const text=await readFile(file);$(id).value=text;invalidate();toast('Loaded '+file.name);}catch(e){toast(e.message,true);}finally{event.target.value='';}};
function gapsFromMatch(match){return match.details.filter(d=>d.category?.toLowerCase()==='skill'&&d.match_type!=='DISABLED'&&(d.match_score==null||Number(d.match_score)<50));}
function showEvidence(){
  if(!state.match)return;
  const filter=$('evidence-filter').value;
  const rows=state.match.details.filter(d=>filter==='all'||(filter==='strong'?d.match_score!=null&&d.match_score>=70:d.match_score==null||d.match_score<50));
  $('evidence-count').textContent=state.match.details.length;
  $('evidence').className='';
  $('evidence').innerHTML=rows.length?'<div class="table-scroll"><table><thead><tr><th>Requirement / category</th><th>Priority</th><th>Match score</th><th>Original resume / status</th></tr></thead><tbody>'+rows.map(d=>{
    const weak=d.match_score==null||d.match_score<50;
    return `<tr><td data-i18n-ignore>${esc(d.requirement)}<small>${esc(d.category)}</small></td><td><span class="badge ${d.priority==='PREFERRED'?'neutral':''}">${esc(d.priority)}</span></td><td><span class="badge ${weak?'gap':''}">${score(d.match_score)}</span></td><td class="evidence-text">${weak?'<span class="badge gap">Insufficient evidence</span><br>':''}${d.resume_evidence?`${weak?'Closest text (not verified evidence):':''}<span data-i18n-ignore>${esc(d.resume_evidence)}</span>`:'No usable resume evidence found'}<small>${esc(human(d.match_type))}</small></td></tr>`;
  }).join('')+'</tbody></table></div><div class="evidence-actions">≥70: strong match · &lt;50: evidence gap · 50–69: partial match. Hard constraints require human review.</div>':'<div class="empty"><p>No requirements match this filter.</p></div>';
}
function showMatch(match){
  state.match=match;const s=match.scores,gaps=gapsFromMatch(match);
  $('match-score').textContent=score(s.hybrid_score);$('coverage-score').textContent=score(s.final_score);$('eligibility').textContent=eligibility(s.eligibility);
  $('score-ring').style.setProperty('--score',Math.max(0,Math.min(100,Number(s.hybrid_score)||0)));
  $('score-description').textContent=`${match.details.length} requirements extracted · ${gaps.length} skills need evidence`;
  $('skill-gaps').innerHTML=gaps.length?chips([...new Set(gaps.map(g=>g.requirement))],'amber')+'<p class="caption">These skills need more resume evidence. Add relevant experience or plan your learning.</p>':'<p class="muted">No actionable skill gaps found. Review education, experience and hard constraints separately.</p>';
  $('bridge-learning').disabled=!gaps.length;$('evidence-filter').value='all';showEvidence();
}
$('evidence-filter').onchange=showEvidence;
$('analyze').onclick=()=>action($('analyze'),'Analyzing documents…',async()=>{
  const jd=$('jd').value,resume=$('resume').value;
  if(!jd.trim()||!resume.trim())throw new Error('Please paste or upload both a job description and a resume.');
  const result=await api('/api/match',{jd,resume});
  if(jd!==$('jd').value||resume!==$('resume').value){toast('Documents changed. Please analyze the current content again.');return;}
  showMatch(result);toast('Analysis complete. Requirements and resume evidence are aligned.');
});
$('bridge-learning').onclick=()=>{
  if(!state.match)return;
  $('gaps').value=gapsFromMatch(state.match).map(d=>d.requirement+' | '+(d.priority||'REQUIRED')).join('\n');
  $('known').value=state.match.profile.skills.join(', ');location.hash='learning';$('recommend').click();
};
function renderCandidates(){
  const previous=new Map(Array.from(document.querySelectorAll('#candidate-checks input')).map(input=>[input.value,input.checked]));
  $('candidate-checks').innerHTML=state.uploads.length?state.uploads.map((r,i)=>`<div class="candidate-check"><input type="checkbox" id="cv-${i}" value="${i}" ${previous.get(String(i))!==false?'checked':''}><label for="cv-${i}"><span data-i18n-ignore>${esc(human(r.name))}</span> <span class="badge">Uploaded</span></label></div>`).join(''):'<p class="caption">Upload resumes to add candidates. No sample resumes are included.</p>';
}
$('rank-files').onchange=async event=>{
  const files=Array.from(event.target.files);if(!files.length)return;
  if(state.uploads.length+files.length>12){toast('You can add up to 12 custom resumes.',true);return;}
  try{const added=[];for(const f of files)added.push({name:f.name,text:await readFile(f)});state.uploads.push(...added);renderCandidates();toast('Added '+added.length+' resumes.');}catch(e){toast(e.message,true);}finally{event.target.value='';}
};
$('rank-button').onclick=()=>action($('rank-button'),'Matching candidates…',async()=>{
  const candidates=state.uploads;
  const resumes=Array.from(document.querySelectorAll('#candidate-checks input:checked')).map(i=>candidates[Number(i.value)]);
  const jd=$('rank-jd').value;
  if(!jd.trim())throw new Error('Please enter a job description before ranking candidates.');
  if(!resumes.length)throw new Error('Please upload and select at least one resume.');
  const {ranking}=await api('/api/rank',{jd,resumes});state.ranked=ranking;state.rankedJd=jd;state.rankedInputs=resumes;
  $('ranking-results').className='';$('ranking-results').innerHTML=ranking.map((r,i)=>`<div class="rank-row" role="button" tabindex="0" data-rank="${i}"><span class="rank-number">${i+1}</span><div class="rank-name"><span data-i18n-ignore>${esc(human(r.name))}</span><small>Coverage ${score(r.scores.final_score)} · ${esc(eligibility(r.scores.eligibility))}</small></div><span class="rank-value">${score(r.scores.hybrid_score)}</span><span>↗</span></div>`).join('')+'<p class="row-hint">Select a candidate → Inspect requirements and evidence</p>';
});
$('ranking-results').addEventListener('click',event=>openRank(event.target.closest('[data-rank]')));
$('ranking-results').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openRank(event.target.closest('[data-rank]'));}});
function openRank(row){if(!row)return;const r=state.ranked[Number(row.dataset.rank)];$('jd').value=state.rankedJd;const candidate=state.rankedInputs[r.input_index];$('resume').value=candidate?.text||'';counts();showMatch(r);location.hash='match';toast('Viewing match evidence for '+human(r.name)+'.');}
function options(){return {gaps:$('gaps').value,known:$('known').value,method:$('method').value,top_k:Number($('top-k').value),budget_hours:$('budget').value?Number($('budget').value):null,optimize:$('optimize').checked};}
function safeLink(url){try{const u=new URL(url);return ['https:','http:'].includes(u.protocol)?esc(u.href):'#';}catch{return '#';}}
function courseHTML(c,step=false){return `<article class="course"><div class="course-head">${step?`<span class="step">${c.step}</span>`:''}<div><h3 data-i18n-ignore>${esc(c.title)}</h3><div class="provider">${esc(c.provider)}</div></div>${step?`<a class="course-link" href="${safeLink(c.url)}" target="_blank" rel="noopener noreferrer">Start learning ↗</a>`:''}</div><p data-i18n-ignore>${esc(c.description)}</p>${chips(c.covered_gaps||c.skills)}${c.unmet_prerequisites?.length?`<p>Check prerequisites:${esc(c.unmet_prerequisites.join(', '))}</p>`:''}<details><summary>Source evidence and topic mappings</summary>${Object.entries(c.evidence).map(([k,v])=>`<blockquote data-i18n-ignore><b>${esc(k)}</b> · ${esc(v)}</blockquote>`).join('')}<span>Duration: ${c.duration_hours==null?'Not specified by provider':esc(c.duration_hours)+' hours'} · Suggested prerequisites: ${esc((c.prerequisites||[]).join(', ')||'None')}</span></details>${step?'':`<a class="course-link" href="${safeLink(c.url)}" target="_blank" rel="noopener noreferrer">Open official resource ↗</a>`}</article>`;}
function showPlan(plan){
  state.plan=plan;$('export-plan').disabled=false;
  $('learning-stats').innerHTML=`<div class="stat"><small>Weighted topic coverage</small><strong>${pct(plan.weighted_coverage)}</strong></div><div class="stat"><small>Learning units</small><strong>${plan.recommendations.length}<span style="font-size:12px"> / ${$('top-k').value}</span></strong></div><div class="stat"><small>Topic redundancy</small><strong>${pct(plan.redundancy_ratio)}</strong></div>`;
  $('learning-results').className='';
  $('learning-results').innerHTML=plan.recommendations.length?plan.recommendations.map(c=>courseHTML(c,true)).join(''):'<div class="card empty"><h3>No resources fit the current settings</h3><p>Check skill names or remove the time budget. Topics outside the catalog will not receive forced recommendations.</p></div>';
  if(plan.uncovered_skills.length)$('learning-results').innerHTML+=`<article class="card padded"><h2>Topics not covered by this catalog</h2>${chips(plan.uncovered_skills,'amber')}</article>`;
  $('learning-results').innerHTML+=`<div class="context-note"><span>ⓘ</span><p>${esc(plan.backend)} · ${plan.latency_ms} ms · Resources come from a curated starter catalog. Topic coverage does not imply skill mastery.</p></div>`;
}
$('recommend').onclick=()=>action($('recommend'),'Building your learning path…',async()=>{const plan=await api('/api/recommend',options());showPlan(plan);$('comparison').innerHTML='';toast('Learning path created.');});
$('compare').onclick=()=>action($('compare'),'Comparing strategies…',async()=>{
  const rows=await api('/api/compare',options());
  $('comparison').innerHTML='<article class="card padded"><h2>Strategy comparison with the same input</h2><div class="table-scroll"><table><thead><tr><th>Strategy</th><th>Weighted coverage</th><th>Redundancy</th><th>Units</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${esc(r.backend)}${r.optimized?' + coverage optimizer':''}</td><td>${pct(r.weighted_coverage)}</td><td>${pct(r.redundancy_ratio)}</td><td>${r.recommendations.length}</td></tr>`).join('')+'</tbody></table></div><p class="caption">These are results for the current input, not accuracy on a human-labeled test set.</p></article>';
});
$('export-plan').onclick=()=>action($('export-plan'),'Preparing report…',async()=>{if(!state.plan)return;const data=await api('/api/export',{plan:state.plan});const url=URL.createObjectURL(new Blob([data.markdown],{type:'text/markdown;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='SkillBridge-learning-plan.md';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
function experiments(key){
  if(!state.data)return;const exp=state.data.experiments[key];
  document.querySelectorAll('[data-exp]').forEach(b=>b.classList.toggle('active',b.dataset.exp===key));
  $('experiment-heading').innerHTML=`<div class="experiment-meta"><div><h2>${esc({skills:'Skill extraction methods',classification:'Requirement priority classification',matching:'Resume matching methods',ablation:'System component ablation'}[key])}</h2><p>${esc(exp.context)} · Metric ${esc(exp.metric)}</p></div><span class="tag">${esc(exp.source)}</span></div>`;
  const valid=exp.rows.filter(r=>Number.isFinite(Number(r[exp.metric]))&&r[exp.metric]!=='');const max=Math.max(...valid.map(r=>Math.abs(Number(r[exp.metric]))),.01);const best=Math.max(...valid.map(r=>Number(r[exp.metric])));
  $('experiment-chart').innerHTML=valid.map(r=>{const n=Number(r[exp.metric]);return `<div class="bar-row ${n===best?'best':''}"><span>${esc(r.method)}</span><div class="bar-track"><div class="bar-fill ${n<0?'negative':''}" style="width:${Math.abs(n)/max*100}%"></div></div><b>${n.toFixed(3)}</b></div>`;}).join('')||'<p class="muted">Result file is not available.</p>';
  const fields=Object.keys(exp.rows[0]||{});
  $('experiment-table').innerHTML=fields.length?'<details><summary class="caption">View original result data</summary><table><thead><tr>'+fields.map(f=>`<th>${esc(f)}</th>`).join('')+'</tr></thead><tbody>'+exp.rows.map(r=>'<tr>'+fields.map(f=>`<td>${esc(r[f])}</td>`).join('')+'</tr>').join('')+'</tbody></table></details>':'';
  if(key==='matching')$('experiment-chart').innerHTML+='<p class="caption">Negative values indicate negative correlation; bar length shows absolute magnitude. Ranking quality must be evaluated separately for each dataset.</p>';
}
$('experiment-tabs').onclick=event=>{const b=event.target.closest('[data-exp]');if(b)experiments(b.dataset.exp);};
function catalog(){const query=$('catalog-search').value.toLowerCase().trim();const filtered=state.catalog.filter(c=>[c.title,c.provider,...c.skills].join(' ').toLowerCase().includes(query));$('catalog-grid').innerHTML=filtered.length?filtered.map(c=>courseHTML(c)).join(''):'<div class="card empty"><h3>No matching resources</h3><p>Try Python, SQL, Docker or FastAPI.</p></div>';$('catalog-count').textContent=filtered.length+' LEARNING UNITS';}
$('catalog-search').oninput=catalog;
$('presentation').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{toast('Fullscreen is unavailable. You can use F11.');}};
async function init(){
  try{const [data,cat]=await Promise.all([api('/api/showcase'),api('/api/catalog')]);state.data=data;state.catalog=cat.resources;
    renderCandidates();counts();catalog();experiments('skills');navigate();
    $('connection').innerHTML=window.SKILLBRIDGE_OFFLINE?'<i></i>Interface preview':'<i></i>Local service connected';
    if(window.SKILLBRIDGE_OFFLINE)$('offline-notice').hidden=false;
  }catch(e){$('connection').textContent='Connection failed';toast('Unable to load data: '+e.message+'. Run Start-Showcase.cmd to start the app.',true);}
}
init();
