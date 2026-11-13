'use strict';
const $ = id => document.getElementById(id);
const state = {data:null, catalog:[], match:null, plan:null, ranked:[], uploads:[], busy:0};
const names = {match:'匹配分析',ranking:'候选人排序',learning:'学习路径',experiments:'实验对比',catalog:'资源目录'};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = value => (Number(value || 0)*100).toFixed(0)+'%';
const score = value => value == null ? '—' : Number(value).toFixed(1);
const human = text => String(text).replaceAll('_',' ');
const chips = (items, cls='') => '<div class="chips">'+items.map(s=>`<span class="chip ${cls}">${esc(s)}</span>`).join('')+'</div>';
const eligibility = text => ({PENDING_VERIFICATION:'待人工核验',ELIGIBLE:'满足已核验要求',NOT_QUALIFIED:'未满足硬性要求',NO_MANDATORY_CONSTRAINTS:'无强制限制项',NO_REQUIREMENTS_EXTRACTED:'未提取到要求'})[text] || human(text || '待核验');
function toast(message,error=false){$('toast').textContent=message;$('toast').className=error?'error':'';$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,error?9000:4500);}
async function api(path, data){
  if(window.SKILLBRIDGE_OFFLINE)return offlineApi(path,data);
  const response=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const value=await response.json();if(!response.ok)throw new Error(value.error||`请求失败 (${response.status})`);return value;
}
function offlineApi(path,data){
  const bundle=window.SKILLBRIDGE_OFFLINE;
  if(path==='/api/showcase')return bundle.bootstrap;
  if(path==='/api/catalog')return bundle.catalog;
  if(path==='/api/export')return {markdown:['# SkillBridge learning plan','',data.plan.backend,'',...data.plan.recommendations.flatMap(c=>['## '+c.step+'. '+c.title,c.description,c.url,'Targets: '+c.covered_gaps.join(', '),''])].join('\n')};
  const find=(jd,resume)=>bundle.matches.find(m=>m.jd===jd&&m.resume===resume);
  if(path==='/api/match'){const item=find(data.jd,data.resume);if(item)return item.result;}
  if(path==='/api/rank'){
    if(!data.resumes?.length)throw new Error('请至少选择一位候选人。');
    const ranking=data.resumes.map((r,i)=>{const m=find(data.jd,r.text);if(!m)throw new Error('离线展示仅支持内置案例，请启动实时版本分析自定义文档。');return {name:r.name,input_index:i,...m.result};});
    ranking.sort((a,b)=>(b.scores.hybrid_score??-1)-(a.scores.hybrid_score??-1));return {ranking};
  }
  if(path==='/api/recommend'||path==='/api/compare'){
    const key=JSON.stringify([data.gaps,data.known]);const plans=bundle.plans[key];
    if(plans)return path==='/api/compare'?plans:plans[3];
    throw new Error('此学习输入未包含在离线案例中，请启动实时版本自定义学习目标。');
  }
  throw new Error('离线展示仅支持内置案例，请双击 Start-Showcase.cmd 使用实时功能。');
}
async function action(button,label,fn){
  if(state.busy){toast('正在处理，请等待当前操作完成。');return;}
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
function counts(){for(const id of ['jd','resume'])$(id+'-count').textContent=$(id).value.length.toLocaleString()+' 字符';}
function invalidate(){
  state.match=null;$('bridge-learning').disabled=true;$('match-score').textContent='—';$('coverage-score').textContent='—';$('eligibility').textContent='待分析';$('score-ring').style.setProperty('--score',0);$('evidence-count').textContent='0';
  $('score-description').textContent='文档已准备好，点击开始匹配分析。';
  $('evidence').className='empty';$('evidence').innerHTML='<span class="empty-symbol">⌕</span><h3>等待分析当前文档</h3><p>结果将显示匹配分数、要求分类与原文证据。</p>';
  $('skill-gaps').innerHTML='<p class="muted">分析后自动识别缺少简历证据的技能。</p>';counts();
}
['jd','resume'].forEach(id=>$(id).addEventListener('input',invalidate));
function chooseExample(){
  $('jd').value=state.data.jobs[Number($('job-example').value)]?.text||'';
  $('resume').value=state.data.resumes[Number($('resume-example').value)]?.text||'';invalidate();
}
$('job-example').onchange=()=>{$('jd').value=state.data.jobs[Number($('job-example').value)]?.text||'';invalidate();};
$('resume-example').onchange=()=>{$('resume').value=state.data.resumes[Number($('resume-example').value)]?.text||'';invalidate();};
$('load-example').onclick=()=>{if(!state.data)return;$('job-example').value='0';$('resume-example').value='2';chooseExample();toast('已加载合成后端岗位与候选人示例。');};
async function readFile(file){
  if(file.size>5*1024*1024)throw new Error('单个文件不能超过 5 MB。');
  const content=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=()=>reject(new Error('无法读取文件。'));r.readAsDataURL(file);});
  return (await api('/api/read-document',{name:file.name,content})).text;
}
for(const id of ['jd','resume'])$(id+'-file').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{const text=await readFile(file);$(id).value=text;invalidate();toast('已读取 '+file.name);}catch(e){toast(e.message,true);}finally{event.target.value='';}};
function gapsFromMatch(match){return match.details.filter(d=>d.category?.toLowerCase()==='skill'&&d.match_type!=='DISABLED'&&(d.match_score==null||Number(d.match_score)<50));}
function showEvidence(){
  if(!state.match)return;
  const filter=$('evidence-filter').value;
  const rows=state.match.details.filter(d=>filter==='all'||(filter==='strong'?d.match_score!=null&&d.match_score>=70:d.match_score==null||d.match_score<50));
  $('evidence-count').textContent=state.match.details.length;
  $('evidence').className='';
  $('evidence').innerHTML=rows.length?'<div class="table-scroll"><table><thead><tr><th>要求 / 类别</th><th>优先级</th><th>匹配分</th><th>简历原文 / 状态</th></tr></thead><tbody>'+rows.map(d=>{
    const weak=d.match_score==null||d.match_score<50;
    return `<tr><td>${esc(d.requirement)}<small>${esc(d.category)}</small></td><td><span class="badge ${d.priority==='PREFERRED'?'neutral':''}">${esc(d.priority)}</span></td><td><span class="badge ${weak?'gap':''}">${score(d.match_score)}</span></td><td class="evidence-text">${weak?'<span class="badge gap">尚无充分证据</span><br>':''}${d.resume_evidence?`${weak?'最近文本（不视为已证实）：':''}${esc(d.resume_evidence)}`:'未找到可用的简历证据'}<small>${esc(human(d.match_type))}</small></td></tr>`;
  }).join('')+'</tbody></table></div><div class="evidence-actions">匹配 ≥70：充分匹配 · &lt;50：证据缺口 · 50–69：部分匹配。硬性限制仍需人工核验。</div>':'<div class="empty"><p>当前筛选没有对应要求。</p></div>';
}
function showMatch(match){
  state.match=match;const s=match.scores,gaps=gapsFromMatch(match);
  $('match-score').textContent=score(s.hybrid_score);$('coverage-score').textContent=score(s.final_score);$('eligibility').textContent=eligibility(s.eligibility);
  $('score-ring').style.setProperty('--score',Math.max(0,Math.min(100,Number(s.hybrid_score)||0)));
  $('score-description').textContent=`${match.details.length} 条提取要求 · ${gaps.length} 项技能需补充证据`;
  $('skill-gaps').innerHTML=gaps.length?chips([...new Set(gaps.map(g=>g.requirement))],'amber')+'<p class="caption">这些技能在简历中尚无充分证据，可以补充经历或规划学习。</p>':'<p class="muted">未发现可转为学习目标的技能差距。学历、年限与硬性限制请另外核验。</p>';
  $('bridge-learning').disabled=!gaps.length;$('evidence-filter').value='all';showEvidence();
}
$('evidence-filter').onchange=showEvidence;
$('analyze').onclick=()=>action($('analyze'),'正在分析文档…',async()=>{
  const jd=$('jd').value,resume=$('resume').value;
  const result=await api('/api/match',{jd,resume});
  if(jd!==$('jd').value||resume!==$('resume').value){toast('文档已更改，请重新分析当前内容。');return;}
  showMatch(result);toast('分析完成，已对齐岗位要求与简历证据。');
});
$('bridge-learning').onclick=()=>{
  if(!state.match)return;
  $('gaps').value=gapsFromMatch(state.match).map(d=>d.requirement+' | '+(d.priority||'REQUIRED')).join('\n');
  $('known').value=state.match.profile.skills.join(', ');location.hash='learning';$('recommend').click();
};
function renderCandidates(){
  const candidates=[...state.data.resumes,...state.uploads];
  $('candidate-checks').innerHTML=candidates.map((r,i)=>`<div class="candidate-check"><input type="checkbox" id="cv-${i}" value="${i}" ${i<3||i>=state.data.resumes.length?'checked':''}><label for="cv-${i}">${esc(human(r.name))}${i>=state.data.resumes.length?' <span class="badge">已上传</span>':''}</label></div>`).join('');
}
$('rank-files').onchange=async event=>{
  const files=Array.from(event.target.files);if(!files.length)return;
  if(state.uploads.length+files.length>12){toast('最多添加 12 份自定义简历。',true);return;}
  try{const added=[];for(const f of files)added.push({name:f.name,text:await readFile(f)});state.uploads.push(...added);renderCandidates();toast('已添加 '+added.length+' 份简历。');}catch(e){toast(e.message,true);}finally{event.target.value='';}
};
$('rank-button').onclick=()=>action($('rank-button'),'正在逐一匹配候选人…',async()=>{
  const candidates=[...state.data.resumes,...state.uploads];
  const resumes=Array.from(document.querySelectorAll('#candidate-checks input:checked')).map(i=>candidates[Number(i.value)]);
  const jd=$('rank-jd').value;
  const {ranking}=await api('/api/rank',{jd,resumes});state.ranked=ranking;state.rankedJd=jd;state.rankedInputs=resumes;
  $('ranking-results').className='';$('ranking-results').innerHTML=ranking.map((r,i)=>`<div class="rank-row" role="button" tabindex="0" data-rank="${i}"><span class="rank-number">${i+1}</span><div class="rank-name">${esc(human(r.name))}<small>覆盖分 ${score(r.scores.final_score)} · ${esc(eligibility(r.scores.eligibility))}</small></div><span class="rank-value">${score(r.scores.hybrid_score)}</span><span>↗</span></div>`).join('')+'<p class="row-hint">点击候选人 → 查看要求与证据</p>';
});
$('ranking-results').addEventListener('click',event=>openRank(event.target.closest('[data-rank]')));
$('ranking-results').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openRank(event.target.closest('[data-rank]'));}});
function openRank(row){if(!row)return;const r=state.ranked[Number(row.dataset.rank)];$('jd').value=state.rankedJd;const candidate=state.rankedInputs[r.input_index];$('resume').value=candidate?.text||'';counts();showMatch(r);location.hash='match';toast('正在查看 '+human(r.name)+' 的匹配证据。');}
function options(){return {gaps:$('gaps').value,known:$('known').value,method:$('method').value,top_k:Number($('top-k').value),budget_hours:$('budget').value?Number($('budget').value):null,optimize:$('optimize').checked};}
function safeLink(url){try{const u=new URL(url);return ['https:','http:'].includes(u.protocol)?esc(u.href):'#';}catch{return '#';}}
function courseHTML(c,step=false){return `<article class="course"><div class="course-head">${step?`<span class="step">${c.step}</span>`:''}<div><h3>${esc(c.title)}</h3><div class="provider">${esc(c.provider)}</div></div>${step?`<a class="course-link" href="${safeLink(c.url)}" target="_blank" rel="noopener noreferrer">前往学习 ↗</a>`:''}</div><p>${esc(c.description)}</p>${chips(c.covered_gaps||c.skills)}${c.unmet_prerequisites?.length?`<p>建议先确认：${esc(c.unmet_prerequisites.join(', '))}</p>`:''}<details><summary>查看来源证据与主题映射</summary>${Object.entries(c.evidence).map(([k,v])=>`<blockquote><b>${esc(k)}</b> · ${esc(v)}</blockquote>`).join('')}<span>时长：${c.duration_hours==null?'官方未标注':esc(c.duration_hours)+' 小时'} · 先修建议：${esc((c.prerequisites||[]).join(', ')||'无')}</span></details>${step?'':`<a class="course-link" href="${safeLink(c.url)}" target="_blank" rel="noopener noreferrer">打开官方资源 ↗</a>`}</article>`;}
function showPlan(plan){
  state.plan=plan;$('export-plan').disabled=false;
  $('learning-stats').innerHTML=`<div class="stat"><small>加权主题覆盖率</small><strong>${pct(plan.weighted_coverage)}</strong></div><div class="stat"><small>学习单元</small><strong>${plan.recommendations.length}<span style="font-size:12px"> / ${$('top-k').value}</span></strong></div><div class="stat"><small>主题重复率</small><strong>${pct(plan.redundancy_ratio)}</strong></div>`;
  $('learning-results').className='';
  $('learning-results').innerHTML=plan.recommendations.length?plan.recommendations.map(c=>courseHTML(c,true)).join(''):'<div class="card empty"><h3>当前条件下没有合适资源</h3><p>检查技能拼写，或取消时间预算。目录未覆盖的主题不会被强行推荐。</p></div>';
  if(plan.uncovered_skills.length)$('learning-results').innerHTML+=`<article class="card padded"><h2>当前目录尚未覆盖</h2>${chips(plan.uncovered_skills,'amber')}</article>`;
  $('learning-results').innerHTML+=`<div class="context-note"><span>ⓘ</span><p>${esc(plan.backend)} · ${plan.latency_ms} ms · 资源来自整理的起始目录。主题覆盖不代表技能掌握。</p></div>`;
}
$('recommend').onclick=()=>action($('recommend'),'正在规划学习路径…',async()=>{const plan=await api('/api/recommend',options());showPlan(plan);$('comparison').innerHTML='';toast('学习路径已生成。');});
$('compare').onclick=()=>action($('compare'),'正在比较策略…',async()=>{
  const rows=await api('/api/compare',options());
  $('comparison').innerHTML='<article class="card padded"><h2>相同输入下的策略比较</h2><div class="table-scroll"><table><thead><tr><th>策略</th><th>加权覆盖率</th><th>重复率</th><th>单元数</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${esc(r.backend)}${r.optimized?' + coverage optimizer':''}</td><td>${pct(r.weighted_coverage)}</td><td>${pct(r.redundancy_ratio)}</td><td>${r.recommendations.length}</td></tr>`).join('')+'</tbody></table></div><p class="caption">这是当前输入的运行结果，不代表人工标注测试集上的准确率。</p></article>';
});
$('export-plan').onclick=()=>action($('export-plan'),'正在生成报告…',async()=>{if(!state.plan)return;const data=await api('/api/export',{plan:state.plan});const url=URL.createObjectURL(new Blob([data.markdown],{type:'text/markdown;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='SkillBridge-learning-plan.md';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
function experiments(key){
  if(!state.data)return;const exp=state.data.experiments[key];
  document.querySelectorAll('[data-exp]').forEach(b=>b.classList.toggle('active',b.dataset.exp===key));
  $('experiment-heading').innerHTML=`<div class="experiment-meta"><div><h2>${esc({skills:'技能抽取方法',classification:'岗位要求优先级分类',matching:'简历匹配方法',ablation:'系统组件消融'}[key])}</h2><p>${esc(exp.context)} · 指标 ${esc(exp.metric)}</p></div><span class="tag">${esc(exp.source)}</span></div>`;
  const valid=exp.rows.filter(r=>Number.isFinite(Number(r[exp.metric]))&&r[exp.metric]!=='');const max=Math.max(...valid.map(r=>Math.abs(Number(r[exp.metric]))),.01);const best=Math.max(...valid.map(r=>Number(r[exp.metric])));
  $('experiment-chart').innerHTML=valid.map(r=>{const n=Number(r[exp.metric]);return `<div class="bar-row ${n===best?'best':''}"><span>${esc(r.method)}</span><div class="bar-track"><div class="bar-fill ${n<0?'negative':''}" style="width:${Math.abs(n)/max*100}%"></div></div><b>${n.toFixed(3)}</b></div>`;}).join('')||'<p class="muted">结果文件暂不可用。</p>';
  const fields=Object.keys(exp.rows[0]||{});
  $('experiment-table').innerHTML=fields.length?'<details><summary class="caption">查看原始结果数据</summary><table><thead><tr>'+fields.map(f=>`<th>${esc(f)}</th>`).join('')+'</tr></thead><tbody>'+exp.rows.map(r=>'<tr>'+fields.map(f=>`<td>${esc(r[f])}</td>`).join('')+'</tr>').join('')+'</tbody></table></details>':'';
  if(key==='matching')$('experiment-chart').innerHTML+='<p class="caption">负值表示负相关，条形长度表示绝对值；不同数据集的排序质量需要分别评估。</p>';
}
$('experiment-tabs').onclick=event=>{const b=event.target.closest('[data-exp]');if(b)experiments(b.dataset.exp);};
function catalog(){const query=$('catalog-search').value.toLowerCase().trim();const filtered=state.catalog.filter(c=>[c.title,c.provider,...c.skills].join(' ').toLowerCase().includes(query));$('catalog-grid').innerHTML=filtered.length?filtered.map(c=>courseHTML(c)).join(''):'<div class="card empty"><h3>没有找到对应资源</h3><p>试试 Python、SQL、Docker 或 FastAPI。</p></div>';$('catalog-count').textContent=filtered.length+' LEARNING UNITS';}
$('catalog-search').oninput=catalog;
$('presentation').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{toast('浏览器未允许全屏，可使用 F11。');}};
async function init(){
  try{const [data,cat]=await Promise.all([api('/api/showcase'),api('/api/catalog')]);state.data=data;state.catalog=cat.resources;
    $('job-example').innerHTML=data.jobs.map((j,i)=>`<option value="${i}">${esc(human(j.name))}</option>`).join('');
    $('resume-example').innerHTML=data.resumes.map((r,i)=>`<option value="${i}">${esc(human(r.name))}</option>`).join('');
    $('resume-example').value='2';chooseExample();renderCandidates();catalog();experiments('skills');navigate();
    $('connection').innerHTML=window.SKILLBRIDGE_OFFLINE?'<i></i>离线案例演示':'<i></i>本地服务已连接';
    if(window.SKILLBRIDGE_OFFLINE){$('offline-notice').hidden=false;['jd','resume','rank-jd'].forEach(id=>$(id).readOnly=true);['gaps','known','method','top-k','budget','optimize','jd-file','resume-file','rank-files'].forEach(id=>$(id).disabled=true);}
  }catch(e){$('connection').textContent='连接失败';toast('无法加载数据：'+e.message+'。请通过 Start-Showcase.cmd 启动。',true);}
}
init();
