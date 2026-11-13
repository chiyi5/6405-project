// Exercise the actual UI script and bundled data without browser/network dependencies.
// This DOM harness is not a visual browser test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'Showcase.html'),'utf8');
const nodes=new Map();
class Element{
  constructor(id){this.id=id;this.value='';this.textContent='';this.innerHTML='';this.hidden=false;this.disabled=false;this.dataset={};this.style={setProperty(){}};this.classList={add(){},remove(){},toggle(){}};}
  addEventListener(){} click(){return this.onclick?.();}
}
for(const [,id] of html.matchAll(/\bid="([^"]+)"/g))nodes.set(id,new Element(id));
const checked=[0,1,2].map(value=>({value:String(value)}));
const context=vm.createContext({console,URL,Blob,setTimeout:()=>0,clearTimeout(){},location:{hash:''},
  window:{addEventListener(){}},document:{getElementById(id){assert(nodes.has(id),'Missing element: '+id);return nodes.get(id);},querySelectorAll(selector){return selector==='#candidate-checks input:checked'?checked:[];},createElement(){return new Element('link');}}});
for(const [,script] of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(script,context);
const run=script=>vm.runInContext(script,context);
async function settle(){for(let i=0;i<15;i++)await Promise.resolve();}
(async()=>{
  await settle();
  assert.match(nodes.get('connection').innerHTML,/离线案例演示/);
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
    if(gaps.length){run("document.getElementById('bridge-learning').onclick()");await settle();assert(run('state.plan.recommendations.length>=0'));assert.doesNotMatch(nodes.get('toast').textContent,/未包含|仅支持/);}
  }
  for(const key of ['skills','classification','matching','ablation']){run(`experiments('${key}')`);assert.doesNotMatch(nodes.get('experiment-chart').innerHTML,/NaN/);}
  nodes.get('catalog-search').value='SQL';run('catalog()');assert.match(nodes.get('catalog-grid').innerHTML,/PostgreSQL/);
  nodes.get('catalog-search').value='nonexistent-123';run('catalog()');assert.match(nodes.get('catalog-grid').innerHTML,/没有找到/);
  run("showMatch({scores:{},details:[{requirement:'<script>alert(1)</script>',category:'Skill',match_score:0}],profile:{skills:[]}})");
  assert(!nodes.get('evidence').innerHTML.includes('<script>'));
  console.log('PASS: UI initialization, 24 cached matches, ranking, learning bridge, strategy comparison, experiments, search and escaping.');
})().catch(error=>{console.error(error);process.exitCode=1;});
