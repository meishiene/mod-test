// Run with: node --test tests/test_workspace_ui.cjs
// Exercise the production presentation code with isolated DOM/transport adapters.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('web/static/app.js', 'utf8');
const presentation = source.slice(source.indexOf('// Two-page presentation.'), source.indexOf("$('#runForm').addEventListener('submit'"));
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function harness(fetch = async () => { throw Error('unexpected fetch'); }) {
  const elements = new Map(), storage = new Map();
  function element(key) {
    if (!elements.has(key)) elements.set(key, {
      innerHTML:'', textContent:'', className:'', hidden:false, dataset:{}, attributes:{},
      classList:{toggle(){},remove(){}}, setAttribute(k,v){this.attributes[k]=v;},
      removeAttribute(k){delete this.attributes[k];}, replaceChildren(){this.innerHTML='';},
      scrollIntoView(){}, querySelector:k=>element(k), querySelectorAll:()=>[],
    });
    return elements.get(key);
  }
  const document={documentElement:element('html'),querySelector:element,querySelectorAll:()=>[]};
  const ctx={document,$:element,window:{scrollY:0,scrollTo(){}},location:{hash:'#test'},
    state:{runs:[],downloadSelected:new Set(),runsTotal:0,runsPage:1,runsPageSize:20},
    localStorage:{setItem:(k,v)=>storage.set(k,v)},toast(){},fetch,AbortController,
    escapeHtml,caseLabel:x=>x,kindLabel:x=>x,statusLabel:x=>x,formatDuration:()=> '12s',scoreBadge:()=>'',
    fmtMetricRate:v=>typeof v==='number'?(v*100).toFixed(1)+'%':'—',
    batchSuiteTable:m=>'<table>'+escapeHtml(m.results[0].case)+'</table>',
    renderBatchReport:escapeHtml,renderSingleReport:escapeHtml,renderScore:x=>escapeHtml(JSON.stringify(x)),
    artifactLinks:()=>'',scoreSummaryHTML:()=>'<div>评分汇总</div>',updateDownloadUI(){},renderRuns(){},
  };
  ctx.window=Object.assign(ctx,ctx.window);
  vm.createContext(ctx);vm.runInContext(fs.readFileSync('web/static/capability-radar.js','utf8'),ctx);vm.runInContext(presentation,ctx);
  return {ctx,element,storage,run:code=>vm.runInContext(code,ctx)};
}
const record=(id,other={})=>({id,model:'m',case_name:'suite',test_kind:'load',protocol:'openai',base_url:'https://example.test',created_at:'2026-10-22T00:00:00Z',status:'completed',passed:true,...other});
const response=(data,status=200)=>({ok:status<400,status,json:async()=>data,text:async()=>typeof data==='string'?data:JSON.stringify(data)});

test('themes use a separate preference and tolerate blocked storage',()=>{
  const h=harness();h.ctx.setTheme('dark',true);
  assert.equal(h.element('html').dataset.theme,'dark');
  assert.equal(h.storage.get('ml-test-platform.theme.v1'),'dark');
  assert.equal(h.storage.size,1);
  h.ctx.localStorage.setItem=()=>{throw Error('blocked')};
  assert.doesNotThrow(()=>h.ctx.setTheme('light',true));
  assert.equal(h.element('html').dataset.theme,'light');
  assert.equal(h.element('#themeToggle').attributes['aria-label'],'切换到深色主题');
});
test('routing hides pages without rebuilding the test form',()=>{
  const h=harness();h.element('#runForm').innerHTML='existing form and inputs';
  h.ctx.showPage('reports');assert.equal(h.element('#testPage').hidden,true);
  h.ctx.showPage('test');assert.equal(h.element('#reportsPage').hidden,true);
  assert.equal(h.element('#runForm').innerHTML,'existing form and inputs');
});
test('completed execution and high score do not imply admission passed',()=>{
  const h=harness();assert.equal(h.ctx.runVerdict(record('a',{passed:null,score:100})),'待完成');
  const html=h.ctx.reportOverview(record('a',{passed:null}),null,'report');
  assert.match(html,/测试尚未完成/);assert.doesNotMatch(html,/100%|所选测试通过/);
});
test('batch shows failures first and never fabricates aggregate rates or percentiles',()=>{
  const h=harness();const manifest={results:[{case:'pass-last',passed:true,metrics:{requests:4,success_rate:1,ttft_p90_seconds:2}},{case:'fail-first',passed:false,metrics:{requests:6,success_rate:0.5,ttft_p90_seconds:9}}]};
  const html=h.ctx.reportOverview(record('a',{passed:false}),manifest,'report');
  assert.ok(html.indexOf('fail-first')<html.indexOf('pass-last'));
  assert.match(html,/分轮展示/);assert.match(html,/<b>10<\/b>/);
  assert.doesNotMatch(html,/75\.0%|5\.5s/);
  manifest.results[1].metrics={};
  assert.match(h.ctx.reportOverview(record('a'),manifest,'report'),/正式请求数<\/span><b>—/);
});
test('late responses cannot overwrite a newer report, even when transport ignores abort',async()=>{
  let resolveSlow;
  const h=harness(url=>url.endsWith('/slow')?new Promise(resolve=>resolveSlow=resolve):Promise.resolve(url.endsWith('/fast')?response(record('fast')):response('fast raw')));
  const slow=h.ctx.openArtifact('slow','response');
  await h.ctx.openArtifact('fast','response');
  resolveSlow(response(record('slow')));await slow;
  assert.equal(h.run('workspace.report.id'),'fast');
  assert.equal(h.element('#reportBody').textContent,'fast raw');
  assert.ok(!h.element('#reportBody').attributes['aria-busy']);
});
test('missing report keeps available manifest evidence and offers full raw report fallback',async()=>{
  const h=harness(async url=>url.endsWith('/a')?response(record('a')):url.endsWith('/manifest')?response({record_type:'batch',results:[{case:'saved-evidence',passed:true,metrics:{requests:1,success_rate:1}}]}):response('',404));
  await h.ctx.openArtifact('a','report');
  assert.match(h.element('#reportBody').innerHTML,/saved-evidence/);
  assert.match(h.element('#reportBody').innerHTML,/报告文件尚不可用/);
});
test('missing manifest still displays original report and missing-data notice',async()=>{
  const h=harness(async url=>url.endsWith('/a')?response(record('a')):url.endsWith('/manifest')?response('',500):response('original retained report'));
  await h.ctx.openArtifact('a','report');
  assert.match(h.element('#reportBody').innerHTML,/original retained report/);
  assert.match(h.element('#reportBody').innerHTML,/指标与逐项证据暂不可用/);
});
test('raw artifacts stay text; unavailable records clear stale actions',async()=>{
  const h=harness(async url=>url.endsWith('/missing')?response('',404):url.endsWith('/a')?response(record('a')):response('<img src=x onerror=alert(1)>'));
  await h.ctx.openArtifact('a','requests');
  assert.equal(h.element('#reportBody').className,'report-body code-body');
  assert.equal(h.element('#reportBody').textContent,'<img src=x onerror=alert(1)>');
  await h.ctx.openArtifact('missing','report');
  assert.equal(h.run('workspace.report'),null);
  assert.equal(h.element('#reportActions').innerHTML,'');
  assert.match(h.element('#reportBody').innerHTML,/记录已不存在/);
});
test('clearing a deleted report invalidates an in-flight read',async()=>{
  let resolve;
  const h=harness(()=>new Promise(r=>resolve=r));const pending=h.ctx.openArtifact('a','report');
  h.ctx.clearReport();resolve(response(record('a')));await pending;
  assert.equal(h.run('workspace.report'),null);
  assert.match(h.element('#reportBody').innerHTML,/选择一条记录/);
});

test('raw failures are visible even when the saved admission flag is true',()=>{
  const h=harness();
  assert.equal(h.ctx.runVerdict(record('a',{passed:true,has_failure_items:true,rating_passed:true})),'有失败项');
  assert.equal(h.ctx.runVerdict(record('a',{passed:true,summary:{results:[{status:'failed'}]}})),'有失败项');
});

test('score details show both sides and multi-model results without a fabricated average',()=>{
  const h=harness();
  const scoreSource=source.slice(source.indexOf('function renderScore('),source.indexOf('// Two-page presentation.'));
  vm.runInContext(scoreSource,h.ctx);
  const html=h.ctx.renderScore({scores:[{model:'first',model_score:100,channel_score:null,score:100,grade:'十分优秀',rating_passed:true,components:{channel:{status:'not_selected'}}},
    {model:'second',model_score:95,channel_score:20,score:115,grade:'官方',rating_passed:true}]});
  for(const text of ['first','second','100 / 100','— / 20','本次未测','115 / 120','十分优秀','官方'])assert.ok(html.includes(text),text);
  assert.doesNotMatch(html,/107\.5/);
});


test('overview renders actual capability scores and tolerates an unavailable score file',async()=>{
  for(const unavailable of [false,true]) {
    const h=harness(async url=>url.endsWith('/a')?response(record('a')):url.endsWith('/score')?
      (unavailable?response('',404):response({faces:[{face:'core',label:'协议',side:'model',weight:20,earned:18,measured_weight:20}]})):
      url.endsWith('/manifest')?response({record_type:'batch',results:[{case:'retained-evidence',passed:true}]}):response('original report'));
    await h.ctx.openArtifact('a','report');
    const html=h.element('#reportBody').innerHTML;
    assert.match(html,/retained-evidence/);assert.match(html,/original report/);
    assert.match(html,/能力评价/);
    assert.match(html,unavailable?/暂无可评估的能力数据/:/90\.0/);
  }
});

test('overview connects saved case names to unscored radar evidence',()=>{
  const h=harness();
  const manifest={model:'m',results:[{case:'vision',summary:{results:[{case_id:'V01',name:'已测视觉理解',status:'passed'}]}}]};
  const score={model:'m',case_scores:{V01:{case_id:'V01',weight:0,status:'passed'}}};
  const html=h.ctx.reportOverview(record('a'),manifest,'original report',[],score);
  assert.match(html,/已测视觉理解：<b>通过/);
  assert.match(html,/未计分测试证据 · 1 项/);
});
