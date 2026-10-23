const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const ctx = {};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync('web/static/capability-radar.js','utf8'),ctx);
const face=(face,weight,earned,measured=weight,side='model')=>({face,weight,earned,measured_weight:measured,side,label:face});
const complete={faces:[face('core',20,16),face('params',5,5),face('envelope',5,3),face('tools',15,15),face('thinking',10,9),face('context',10,8),face('vision',10,7),face('quality',25,25)]};

test('dimensions merge weighted faces without changing the score object',()=>{
 const input=structuredClone(complete);const before=JSON.stringify(input);
 const dimensions=ctx.capabilityDimensions(input);
 assert.equal(dimensions.length,6);assert.equal(dimensions[0].score,80);
 assert.equal(dimensions[0].coverage,1);
 assert.equal(dimensions[1].score,100);
 assert.equal(JSON.stringify(input),before);
});
test('missing and insufficient evidence never become fake zero or full scores',()=>{
 const dimensions=ctx.capabilityDimensions({faces:[face('core',20,5,5),face('tools',10,0),face('context',10,0,0)]});
 assert.equal(dimensions[0].score,null);assert.equal(dimensions[0].state,'证据不足');
 assert.equal(dimensions[1].score,0);assert.equal(dimensions[1].rating,'待提升');
 assert.equal(dimensions[2].score,null);assert.equal(dimensions[3].score,null);
 const html=ctx.renderCapabilityRadar({faces:[face('tools',10,0)]});
 assert.ok(!html.includes('radar-area'));assert.match(html,/未评估/);assert.match(html,/0\.0/);
});
test('coverage threshold is 80 percent and observed credit uses the full denominator',()=>{
 const dimensions=ctx.capabilityDimensions({faces:[face('core',10,8,8),face('tools',10,7.99,7.99)]});
 assert.equal(dimensions[0].score,80);assert.equal(dimensions[1].score,null);
});
test('complete chart has six vertices and accessible dimension evidence',()=>{
 const html=ctx.renderCapabilityRadar(complete);
 assert.match(html,/class="radar-area"/);assert.match(html,/role="img"/);
 assert.equal((html.match(/class="radar-dimension"/g)||[]).length,6);
 assert.match(html,/核心对话、参数与错误契约/);assert.match(html,/模型侧/);
 assert.match(html,/优势维度/);
});
test('missing scores remain a small useful empty state',()=>{
 for(const input of [null,{}, {error:'missing'}, {faces:[]}]) {
  const html=ctx.renderCapabilityRadar(input);
  assert.match(html,/暂无可评估的能力数据/);assert.ok(!html.includes('<svg'));
 }
});
test('multiple models are separate with visible accessible switches and escaped labels',()=>{
 const html=ctx.renderCapabilityRadar({scores:[{...complete,model:'<img src=x onerror=bad>'},{...complete,model:'second'}]});
 assert.match(html,/role="tablist"/);assert.match(html,/aria-selected="true"/);assert.match(html,/aria-selected="false"/);
 assert.equal((html.match(/class="radar-area"/g)||[]).length,2);
 assert.ok(!html.includes('<img src=x'));assert.match(html,/&lt;img/);
});
test('trust veto is prominent and never becomes positive overall evaluation',()=>{
 const html=ctx.renderCapabilityRadar({...complete,valid:false,invalid_reason:'身份校验失败 <script>'});
 assert.match(html,/总评分无效/);assert.match(html,/&lt;script&gt;/);
 assert.ok(!html.includes('优势维度'));
});

test('GLM capability contracts are scored in their own axes, not a fabricated quality grade',()=>{
 const score={model:'glm-5.3-flash',score:87.4344,faces:[face('core',42.4242,42.4242),face('thinking',15.1515,3.0303),face('tools',21.2121,21.2121),face('context',15.1515,10.101),face('envelope',6.0606,0),face('tools',14,4.6667,14,'channel'),face('params',6,6,6,'channel')],case_scores:{
  'GLM-53-F02':{case_id:'GLM-53-F02',weight:3.0303,status:'passed'},
  'GLM-53-F08':{case_id:'GLM-53-F08',weight:0,status:'passed'},
  'GLM-53-V01':{case_id:'GLM-53-V01',weight:0,status:'passed'},
 }};
 const manifest={model:score.model,results:[{summary:{results:[{case_id:'GLM-53-V01',name:'图像内容理解（五级看图答题）',status:'passed'}]}}]};
 const before=JSON.stringify({score,manifest});
 const dims=ctx.capabilityDimensions(score);
 assert.equal(dims[5].label,'能力质量');assert.equal(dims[5].state,'无计分项');assert.equal(dims[5].score,null);
 assert.equal(Math.round(dims[2].score),20);
 const html=ctx.renderCapabilityRadar(score,manifest);
 assert.match(html,/未计分测试证据 · 2 项/);
 assert.match(html,/图像内容理解（五级看图答题）：<b>通过/);
 assert.doesNotMatch(html,/能力验证|GLM-53-F02/);
 assert.equal(JSON.stringify({score,manifest}),before);
});
test('unscored evidence preserves latest verdict, model isolation and unsafe text escaping',()=>{
 const score={model:'first',case_scores:{V:{case_id:'V',weight:0,status:'failed'},S:{case_id:'S',weight:0,status:'skipped'}}};
 const manifest={results:[{model:'first',summary:{results:[{case_id:'V',name:'<img src=x>',status:'passed'}]}},{model:'second',summary:{results:[{case_id:'V',name:'wrong-model-name',status:'passed'}]}}]};
 const html=ctx.renderCapabilityRadar(score,manifest);
 assert.match(html,/暂无可评估的能力数据/);assert.match(html,/未计分测试证据 · 2 项/);
 assert.match(html,/0 项通过/);assert.match(html,/&lt;img src=x&gt;：<b>失败/);assert.match(html,/未执行/);
 assert.doesNotMatch(html,/<svg|wrong-model-name|<img/);
});
test('unselected dimension differs from applicable but untested evidence',()=>{
 const dims=ctx.capabilityDimensions({faces:[face('quality',20,0,0)]});
 assert.equal(dims[5].state,'未评估');assert.equal(dims[5].score,null);
 assert.equal(dims[4].state,'无计分项');assert.equal(dims[4].score,null);
});

test('new six-axis payload shows actual case scores including quality and vision',()=>{
 const axes=['protocol','tools','thinking','context','vision','quality'].map((key,i)=>({key,label:key,weight:100,measured:100,earned:i===2?0:100,score:i===2?0:100,coverage:1,state:'已评估',evaluated:1,case_count:1,description:'真实用例',cases:[{case_id:key,name:'<unsafe>',weight:100,earned:i===2?0:100,lost:i===2?100:0,status:i===2?'failed':'passed',side:'model',total_weight:0}]}));
 const payload={model:'glm',capabilities:{version:'capabilities-1.0',axes},faces:[]};
 const before=JSON.stringify(payload);
 const dims=ctx.capabilityDimensions(payload);
 assert.equal(dims.filter(d=>d.score!==null).length,6);assert.equal(dims[2].score,0);
 const html=ctx.renderCapabilityRadar(payload);
 assert.match(html,/radar-area/);assert.match(html,/6 \/ 6/);assert.match(html,/&lt;unsafe&gt;/);
 assert.match(html,/\+100\.0/);assert.match(html,/−100\.0/);assert.doesNotMatch(html,/<unsafe>/);
 assert.equal(JSON.stringify(payload),before);
});
test('new axes keep untested and unsupported dimensions distinct with coverage',()=>{
 const axes=[{key:'quality',score:null,weight:100,measured:25,earned:25,coverage:.25,state:'证据不足',evaluated:1,case_count:2,cases:[]},
 {key:'vision',score:null,weight:0,measured:0,earned:0,coverage:0,state:'不适用',evaluated:0,case_count:0,cases:[]}];
 const html=ctx.renderCapabilityRadar({capabilities:{version:'capabilities-1.0',axes}});
 assert.match(html,/证据不足/);assert.match(html,/不适用/);assert.match(html,/25%/);assert.doesNotMatch(html,/radar-area/);
});
