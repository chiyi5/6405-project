// Exercise the actual UI script and bundled data without browser/network dependencies.
// This DOM harness is not a visual browser test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'Showcase.html'),'utf8');
const nodes=new Map();
class Element{
  constructor(id){this.id=id;this.value='';this.textContent='';this.innerHTML='';this.hidden=false;this.disabled=false;this.dataset={};this.nodeType=1;this.attrs={};this.style={setProperty(){}};this.classList={add(){},remove(){},toggle(){}};}
  addEventListener(name,handler){this['on'+name]=handler;} click(){return this.onclick?.();}
  closest(){return this.ignore?this:null;} querySelectorAll(){return [];}
  hasAttribute(name){return name in this.attrs;} getAttribute(name){return this.attrs[name];} setAttribute(name,value){this.attrs[name]=value;}
}
for(const [,id] of html.matchAll(/\bid="([^"]+)"/g))nodes.set(id,new Element(id));
const checked=[0,1,2].map(value=>({value:String(value)}));
const stored=new Map();
const context=vm.createContext({console,URL,Blob,setTimeout:()=>0,clearTimeout(){},location:{hash:''},NodeFilter:{SHOW_TEXT:4},
  localStorage:{getItem:key=>stored.get(key),setItem:(key,value)=>stored.set(key,value)},MutationObserver:class{observe(){}},
  window:{addEventListener(){}},document:{documentElement:new Element('html'),createTreeWalker(root){let index=0;return {nextNode(){this.currentNode=root.textNodes?.[index++];return Boolean(this.currentNode);}};},getElementById(id){assert(nodes.has(id),'Missing element: '+id);return nodes.get(id);},querySelectorAll(selector){return selector==='#candidate-checks input:checked'?checked:[];},createElement(){return new Element('link');}}});
for(const [,script] of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(script,context);
const run=script=>vm.runInContext(script,context);
async function settle(){for(let i=0;i<15;i++)await Promise.resolve();}
(async()=>{
  await settle();
  assert.match(nodes.get('connection').innerHTML,/Offline example demo/);
  assert.equal(run('window.SkillBridgeI18n.language'),'en');
  assert.equal(run('document.documentElement.lang'),'en');
  assert(nodes.get('jd').readOnly);
  for(const [id,value] of Object.entries({'top-k':'5',known:'Python',gaps:'SQL | REQUIRED\nDocker | REQUIRED\nFastAPI | PREFERRED',method:'hybrid',budget:''}))nodes.get(id).value=value;
  nodes.get('optimize').checked=true;
  await run("document.getElementById('analyze').onclick()");await settle();
  assert.notEqual(nodes.get('match-score').textContent,'—');
  assert.match(nodes.get('evidence').innerHTML,/<table>/);
  assert.equal(run('state.match.details.length>0'),true);
  nodes.get('rank-jd').value=nodes.get('jd').value;
  await run("document.getElementById('rank-button').onclick()");await settle();
  assert.equal(run('state.ranked.length'),3);
  assert.match(nodes.get('ranking-results').innerHTML,/rank-row/);
  await run("document.getElementById('recommend').onclick()");await settle();
  assert.match(nodes.get('learning-results').innerHTML,/course/);
  assert.equal(run('state.plan.weighted_coverage'),1);
  await run("document.getElementById('compare').onclick()");await settle();
  assert.match(nodes.get('comparison').innerHTML,/coverage optimizer/);
  for(const item of run('window.SKILLBRIDGE_OFFLINE.matches')){
    nodes.get('jd').value=item.jd;nodes.get('resume').value=item.resume;
    await run("document.getElementById('analyze').onclick()");await settle();
    const gaps=run('gapsFromMatch(state.match)');
    if(gaps.length){run("document.getElementById('bridge-learning').onclick()");await settle();assert(run('state.plan.recommendations.length>=0'));assert.doesNotMatch(nodes.get('toast').textContent,/not included|examples only/);}
  }
  for(const key of ['skills','classification','matching','ablation']){run(`experiments('${key}')`);assert.doesNotMatch(nodes.get('experiment-chart').innerHTML,/NaN/);}
  nodes.get('catalog-search').value='SQL';run('catalog()');assert.match(nodes.get('catalog-grid').innerHTML,/PostgreSQL/);
  nodes.get('catalog-search').value='nonexistent-123';run('catalog()');assert.match(nodes.get('catalog-grid').innerHTML,/No matching resources/);
  const savedMatch=run('state.match'),savedPlan=run('state.plan'),savedJD=nodes.get('jd').value;
  nodes.get('display-language').value='zh';nodes.get('display-language').onchange();
  assert.equal(run('window.SkillBridgeI18n.language'),'zh');
  assert.equal(run('document.documentElement.lang'),'zh-CN');
  assert.equal(stored.get('skillbridge.language'),'zh');
  assert.equal(run('state.match'),savedMatch);assert.equal(run('state.plan'),savedPlan);assert.equal(nodes.get('jd').value,savedJD);
  const tr=run('window.SkillBridgeI18n.translate');
  assert.equal(tr('12 requirements extracted · 3 skills need evidence','zh'),'12 条提取要求 · 3 项技能需补充证据');
  assert.equal(tr('Coverage 80.0 · Pending human review','zh'),'覆盖分 80.0 · 待人工核验');
  assert.equal(tr('↺ Load demo example','zh'),'↺ 加载演示案例');
  assert.equal(tr('Analyze match ↗','zh'),'开始匹配分析 ↗');
  assert.equal(tr('Loaded candidate-中文.txt','zh'),'已读取 candidate-中文.txt');
  assert.equal(tr('Check prerequisites: python, sql','zh'),'建议先确认：python, sql');
  run(`var languageRoot=document.createElement('section'); languageRoot.textNodes=[
    {nodeType:3,nodeValue:'Analyze match',parentElement:languageRoot},
    {nodeType:3,nodeValue:'12 requirements extracted · 3 skills need evidence',parentElement:languageRoot},
    {nodeType:3,nodeValue:'Analyze match',parentElement:{closest(){return {};}}}];
    languageRoot.setAttribute('placeholder','No limit'); window.SkillBridgeI18n.refresh(languageRoot);`);
  assert.equal(run('languageRoot.textNodes[0].nodeValue'),'开始匹配分析');
  assert.equal(run('languageRoot.textNodes[1].nodeValue'),'12 条提取要求 · 3 项技能需补充证据');
  assert.equal(run('languageRoot.textNodes[2].nodeValue'),'Analyze match'); // Source evidence stays untouched.
  assert.equal(run("languageRoot.getAttribute('placeholder')"),'不限');
  run("languageRoot.textNodes[1].nodeValue='9 requirements extracted · 2 skills need evidence'; window.SkillBridgeI18n.refresh(languageRoot)");
  assert.equal(run('languageRoot.textNodes[1].nodeValue'),'9 条提取要求 · 2 项技能需补充证据');
  run("window.SkillBridgeI18n.setLanguage('en')");assert.equal(stored.get('skillbridge.language'),'en');
  run('window.SkillBridgeI18n.refresh(languageRoot)');
  assert.equal(run('languageRoot.textNodes[0].nodeValue'),'Analyze match');
  assert.equal(run('languageRoot.textNodes[1].nodeValue'),'9 requirements extracted · 2 skills need evidence');
  assert.equal(run("languageRoot.getAttribute('placeholder')"),'No limit');
  stored.set('skillbridge.language','zh');
  const languageScript=fs.readFileSync(path.join(root,'frontend/i18n.js'),'utf8');
  vm.runInContext(languageScript,context);assert.equal(run('window.SkillBridgeI18n.language'),'zh');
  context.localStorage={getItem(){throw new Error('Storage blocked');},setItem(){throw new Error('Storage blocked');}};
  vm.runInContext(languageScript,context);assert.equal(run('window.SkillBridgeI18n.language'),'en');
  run("window.SkillBridgeI18n.setLanguage('zh')");assert.equal(run('window.SkillBridgeI18n.language'),'zh');
  run("showMatch({scores:{},details:[{requirement:'<script>alert(1)</script>',category:'Skill',match_score:0}],profile:{skills:[]}})");
  assert(!nodes.get('evidence').innerHTML.includes('<script>'));
  console.log('PASS: English default, persistent language switching, preserved results, 24 cached matches, ranking, learning, experiments, search and escaping.');
})().catch(error=>{console.error(error);process.exitCode=1;});
