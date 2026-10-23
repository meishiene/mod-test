const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const TestCasePresets=require('../web/static/case-presets.js');
const source=fs.readFileSync('web/static/app.js','utf8');
const render=source.slice(source.indexOf('function caseFacts('),source.indexOf('function selectedValues('));
const selection=source.slice(source.indexOf('function selectCase('),source.indexOf('function formatDuration('));
const item=(id,extra={})=>({id,title:id,test_kind:'load',cost:'低',...extra});
function harness(cases,{imageConfig=false}={}){
 const elements=new Map();
 const element=key=>{
  if(!elements.has(key))elements.set(key,{value:'',checked:false,dataset:{},style:{},html:'',hidden:false,
   get innerHTML(){return this.html},
   set innerHTML(value){this.html=value;if(key.startsWith('#image')||key.startsWith('#gemini'))this.options=[...value.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)].map(([,value,textContent])=>({value,textContent,selected:false}));},
   get selectedOptions(){return this.options.filter(option=>option.selected)},
   textContent:'',options:[],querySelector:element,querySelectorAll:()=>[]});
  return elements.get(key);
 };
 const ctx={TestCasePresets,state:{cases,selected:[],activeCase:'',testKind:'load',models:[],caseModels:[],faceChannel:{}},
  $:element,protocolSupports:item=>!item.unsupported,kindMatches:()=>true,
  escapeHtml:value=>String(value??'').replace(/[<>&"']/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'}[c])),
  updateSelectionUI(){if(imageConfig){ctx.renderImageConfig();ctx.renderGeminiImageConfig()}ctx.renderCaseDetails()},persistForm(){},hideModelSuggestions(){},updateEndpointPreview(){},
  renderImageConfig(){},renderGeminiImageConfig(){}};
 vm.createContext(ctx);vm.runInContext(render+'\n'+selection,ctx);
 if(imageConfig){
  ctx.document={querySelector:element,querySelectorAll:()=>[]};
  vm.runInContext(source.slice(source.indexOf('function selectedValues('),source.indexOf('// —— Seedance 视频生成 ——')),ctx);
 }
 return {ctx,element};
}
test('viewing another test leaves selection and typed configuration intact',()=>{
 const {ctx,element}=harness([item('a'),item('b')]);ctx.state.selected=['a'];
 element('#customConcurrency').value='7';element('#imagePrompt').value='保留提示词';
 ctx.renderCases();ctx.state.activeCase='b';ctx.renderCases();
 assert.deepEqual(ctx.state.selected,['a']);assert.equal(element('#testExecutionConfigs').hidden,true);
 assert.match(element('#caseDetail').innerHTML,/当前仅查看详情/);
 assert.equal(element('#customConcurrency').value,'7');assert.equal(element('#imagePrompt').value,'保留提示词');
 ctx.selectCase('b');assert.equal(ctx.state.selected.join(','),'a,b');
 assert.equal(element('#testExecutionConfigs').hidden,false);
});
const glmCases=()=>['CORE','AGENT','CACHE','FINGERPRINT','CAPABILITY','VERIFY','FULL'].map(id=>item('GLM_53_'+id,{test_kind:'glm'}));
test('GLM presets replace selection atomically and manual edits leave the preset',()=>{
 const {ctx,element}=harness(glmCases());ctx.state.testKind='glm';
 element('#model').value='my-model';element('#customConcurrency').value='7';ctx.state.caseModels=['m1','m2'];
 ctx.applyCasePreset('basic');assert.equal(ctx.state.selected.join(','),'GLM_53_CORE,GLM_53_VERIFY');
 ctx.applyCasePreset('agent');assert.equal(ctx.state.selected.join(','),'GLM_53_CORE,GLM_53_AGENT');
 ctx.applyCasePreset('agent');assert.equal(ctx.state.selected.length,2,'clicking the active combination is idempotent');
 ctx.selectCase('GLM_53_CACHE');assert.equal(ctx.state.selected.length,3);
 assert.match(element('#casePresetStatus').textContent,/自定义/);
 ctx.applyCasePreset('full');assert.equal(ctx.state.selected.join(','),'GLM_53_FULL,GLM_53_CACHE');
 assert.equal(element('#model').value,'my-model');assert.equal(element('#customConcurrency').value,'7');
 assert.equal(ctx.state.caseModels.join(','),'m1,m2');
 ctx.clearCaseSelection();assert.equal(ctx.state.selected.length,0);assert.equal(element('#testExecutionConfigs').hidden,true);
});
test('presets never silently select a partial unsupported combination or enable channel faces',()=>{
 const cases=glmCases();cases.find(x=>x.id==='GLM_53_AGENT').unsupported=true;
 const {ctx,element}=harness(cases);ctx.state.testKind='glm';ctx.state.selected=['GLM_53_CACHE'];
 ctx.applyCasePreset('agent');assert.equal(ctx.state.selected.join(','),'GLM_53_CACHE');
 const agent=cases.find(x=>x.id==='GLM_53_AGENT');delete agent.unsupported;
 agent.faces={channel_count:3,counts:{channel:3}};
 ctx.renderCases();assert.match(element('#casePresets').innerHTML,/需启用通道面/);
 ctx.applyCasePreset('agent');assert.equal(ctx.state.selected.join(','),'GLM_53_CACHE');
 assert.equal(ctx.state.faceChannel[agent.id],undefined);
 ctx.state.faceChannel[agent.id]=true;ctx.applyCasePreset('agent');assert.equal(ctx.state.selected.join(','),'GLM_53_CORE,GLM_53_AGENT');
});
test('Kimi temporary suite stays exclusive when adjusting a preset',()=>{
 const {ctx}=harness(['KVV_K3_FEATURES','KVV_K3_DYNAMIC_TOOLS','K3_TEMP_SUITE'].map(id=>item(id,{test_kind:'kvv'})));
 ctx.state.testKind='kvv';ctx.applyCasePreset('agent');assert.equal(ctx.state.selected.length,2);
 ctx.selectCase('K3_TEMP_SUITE');assert.equal(ctx.state.selected.join(','),'K3_TEMP_SUITE');
 ctx.selectCase('KVV_K3_FEATURES');assert.equal(ctx.state.selected.join(','),'KVV_K3_FEATURES');
});
test('preset definitions follow visible variants and omit incomplete groups',()=>{
 for(const [kind,prefixes] of [['fable',['FABLE5','OPUS5','SONNET5','HAIKU45','SONNET46']],['gpt',['GPT56','GPT61','GPT6']]]){
  for(const prefix of prefixes){
   const cases=['BASIC','P0','P1','FINGERPRINT'].map(suffix=>item(prefix+'_'+suffix,{test_kind:kind}));
   const presets=TestCasePresets.forCases(kind,cases);
   assert.ok(presets.length>=2);
   assert.ok(presets.every(p=>p.caseIds.every(id=>id.startsWith(prefix+'_'))));
  }
 }
 assert.equal(TestCasePresets.forCases('glm',[item('GLM_53_CORE')]).length,0);
 assert.equal(TestCasePresets.forCases('chat',[item('CHAT_PROMPT')]).length,0);
});
test('row buttons toggle selection while detail buttons only inspect',()=>{
 const {ctx,element}=harness([item('a'),item('b')]);ctx.state.selected=['a'];
 const row={dataset:{case:'b'},focus(){}};const detail={dataset:{case:'b'},focus(){}};
 element('#caseGrid').querySelectorAll=selector=>selector==='.test-case-button'?[row]:selector==='.test-case-detail-button'?[detail]:[];
 ctx.renderCases();assert.doesNotMatch(element('#caseGrid').innerHTML,/type="checkbox"/);
 detail.onclick();assert.equal(ctx.state.activeCase,'b');assert.equal(ctx.state.selected.join(','),'a');
 row.onclick();assert.equal(ctx.state.selected.join(','),'a,b');
 row.onclick();assert.equal(ctx.state.selected.join(','),'a');
});
test('restoring selection preserves clear and excludes hidden or unavailable tests',()=>{
 const {ctx}=harness([...glmCases(),item('other'),item('blocked',{test_kind:'glm',unsupported:true})]);
 ctx.state.testKind='glm';ctx.kindMatches=item=>item.test_kind==='glm';
 ctx.restoreCaseSelection([]);assert.equal(ctx.state.selected.length,0);
 ctx.restoreCaseSelection(['GLM_53_AGENT','other','blocked','GLM_53_CORE','GLM_53_AGENT']);
 assert.equal(ctx.state.selected.join(','),'GLM_53_AGENT,GLM_53_CORE');
 ctx.restoreCaseSelection(undefined);assert.equal(ctx.state.selected.join(','),'GLM_53_CORE');
 ctx.state.cases=[item('a'),item('fixed',{fixed_parameters:true})];ctx.kindMatches=()=>true;
 ctx.restoreCaseSelection(['a','fixed']);assert.equal(ctx.state.selected.join(','),'fixed');
});
test('fixed and manual cases stay exclusive and can be removed explicitly',()=>{
 const {ctx}=harness([item('a'),item('fixed',{fixed_parameters:true}),item('manual',{manual_parameters:true})]);
 ctx.selectCase('a');ctx.selectCase('fixed');assert.equal(ctx.state.selected.join(','),'fixed');
 ctx.selectCase('fixed');assert.equal(ctx.state.selected.length,0);
 ctx.selectCase('manual');ctx.selectCase('a');assert.equal(ctx.state.selected.join(','),'a');
});
test('channel-only and incompatible tests remain guarded but their details can be viewed',()=>{
 const {ctx,element}=harness([item('channel',{faces:{channel_count:3,counts:{model:0,verify:0}}}),item('bad',{unsupported:true})]);
 ctx.state.activeCase='channel';ctx.renderCases();ctx.selectCase('channel');assert.equal(ctx.state.selected.length,0);
 assert.match(element('#caseDetail').innerHTML,/含通道面/);
 ctx.state.faceChannel.channel=true;ctx.selectCase('channel');assert.equal(ctx.state.selected.join(','),'channel');
 ctx.selectCase('bad');assert.equal(ctx.state.selected.join(','),'channel');
});
test('multi-model controls follow the complete selected scope and retain model values',()=>{
 const {ctx,element}=harness([item('multi',{model_scope:true}),item('single')]);
 ctx.state.selected=['multi'];ctx.state.caseModels=['m1','m2'];element('#model').value='m1';
 ctx.renderTestModels();assert.equal(ctx.testModelMultiple(),true);
 assert.match(element('#testModelOptions').innerHTML,/m2/);
 ctx.state.selected.push('single');ctx.renderTestModels();assert.equal(ctx.testModelMultiple(),false);
 assert.equal(element('#testModelMode').textContent,'单模型执行');
 assert.equal(element('#model').value,'m1');assert.equal(ctx.state.caseModels.length,2);
});
test('model checkbox selection uses the same first-model and case_models contract',()=>{
 const {ctx,element}=harness([item('multi',{model_scope:true})]);
 ctx.state.selected=['multi'];ctx.state.models=[{id:'m1'},{id:'m2'}];element('#model').value='m1';
 const input={value:'m2',checked:true};element('#testModelOptions').querySelectorAll=()=>[input];
 ctx.renderTestModels();input.onchange();assert.equal(ctx.state.caseModels.join(','),'m1,m2');assert.equal(element('#model').value,'m1');
});
test('image model checkboxes write through to the original select options',()=>{
 const {ctx,element}=harness([item('image')]);ctx.state.testKind='gpt-image';ctx.state.selected=['image'];
 const options=[{value:'image-1',selected:true},{value:'image-2',selected:false}];element('#imageModels').options=options;
 const input={value:'image-2',checked:true};element('#testModelOptions').querySelectorAll=()=>[input];
 ctx.renderTestModels();input.onchange();assert.equal(options[1].selected,true);assert.equal(options[0].selected,true);
});
for(const [kind,selectors] of [
 ['gpt-image',['#imageModels','#imageQuality','#imageSizes']],
 ['gemini-image',['#geminiModels','#geminiQuality','#geminiAspects','#geminiCounts']],
])test(`${kind} joining a test preserves partial and empty detail selections`,()=>{
 const {ctx,element}=harness([item('image',{
  test_kind:kind,image_models:['model-a','model-b'],image_qualities:['low','high'],
  image_sizes:['1K','2K'],image_aspect_ratios:['1:1','16:9'],image_counts:[1,2],
 })],{imageConfig:true});
 ctx.state.testKind=kind;
 ctx.selectCase('image');
 for(const selector of selectors){
  assert.equal(element(selector).options.length,2);
  assert.equal(element(selector).selectedOptions.length,0,'initial selection must not select every option');
  element(selector).options[0].selected=true;
 }
 ctx.selectCase('image');ctx.selectCase('image');
 for(const selector of selectors){
  assert.equal(element(selector).selectedOptions.length,1);
  assert.equal(element(selector).options[0].selected,true);
  element(selector).options[0].selected=false;
 }
 ctx.selectCase('image');ctx.selectCase('image');ctx.updateSelectionUI();
 for(const selector of selectors)assert.equal(element(selector).selectedOptions.length,0,'cleared choices must stay empty after rejoining and rerendering');
 assert.doesNotMatch(element('#testModelOptions').innerHTML,/ checked/);
});
test('previewing image 2.5 shows its own models without changing image 2 execution',()=>{
 const cases=[item('image2',{image_models:['gpt-image-2']}),item('image25',{image_models:['gpt-image-2.5-flare','gpt-image-2.5-sunburst']})];
 const {ctx,element}=harness(cases);ctx.state.testKind='gpt-image';ctx.state.selected=['image2'];ctx.state.activeCase='image25';
 const options=[{value:'gpt-image-2',selected:true}];element('#imageModels').options=options;
 element('#customConcurrency').value='7';element('#imagePrompt').value='保留提示词';
 ctx.renderCases();
 assert.match(element('#testModelOptions').innerHTML,/gpt-image-2\.5-flare/);
 assert.match(element('#testModelOptions').innerHTML,/gpt-image-2\.5-sunburst/);
 assert.doesNotMatch(element('#testModelOptions').innerHTML,/>gpt-image-2</);
 assert.match(element('#testModelOptions').innerHTML,/disabled/);
 assert.equal(element('#testModelMode').textContent,'可用模型预览');
 assert.equal(ctx.state.selected.join(','),'image2');assert.equal(options[0].selected,true);
 assert.equal(element('#customConcurrency').value,'7');assert.equal(element('#imagePrompt').value,'保留提示词');
 ctx.state.activeCase='image2';ctx.renderCases();
 assert.match(element('#testModelOptions').innerHTML,/>gpt-image-2</);
 assert.equal(element('#testModelMode').textContent,'随机抽取已勾选模型');
});
test('fixed image tests display their prescribed models, not an editable global model',()=>{
 const {ctx,element}=harness([item('fixed',{fixed_parameters:true,image_models:['fixed-a','fixed-b']})]);
 ctx.state.testKind='gpt-image';ctx.state.selected=['fixed'];ctx.state.activeCase='fixed';element('#model').value='unrelated';
 ctx.renderCases();
 assert.match(element('#testModelOptions').innerHTML,/fixed-a/);assert.match(element('#testModelOptions').innerHTML,/fixed-b/);
 assert.doesNotMatch(element('#testModelOptions').innerHTML,/unrelated/);
 assert.match(element('#testModelOptions').innerHTML,/disabled checked/);
});
test('test titles and model identifiers are rendered as text',()=>{
 const {ctx,element}=harness([item('a',{title:'<script>bad</script>'})]);
 element('#model').value='"><img src=x>';ctx.renderCases();
 assert.ok(!element('#caseDetail').innerHTML.includes('<script>'));
 assert.ok(!element('#testModelOptions').innerHTML.includes('<img'));
});

test('economy preset selects only the budgeted card for each family',()=>{
 for(const kind of ['deepseek','glm','qwen','kvv','fable','gpt']){
  const card=item(`${kind.toUpperCase()}_ECONOMY`,{test_kind:kind,economy_validation:true});
  const cases=[card,item('KVV_TOOL_SCHEMA',{test_kind:kind}),...glmCases()];
  const {ctx}=harness(cases);ctx.state.testKind=kind;
  ctx.applyCasePreset('economy');
  assert.equal(ctx.state.selected.join(','),card.id);
 }
});
test('economy cards remain visible for every GPT and Claude variant',()=>{
 const ctx={state:{testKind:'gpt',gptVariant:'gpt6.1'}};vm.createContext(ctx);
 vm.runInContext(source.slice(source.indexOf('function kindMatches('),source.indexOf('function renderProtocols(')),ctx);
 for(const kind of ['gpt','fable']){
  ctx.state.testKind=kind;
  for(const variant of ['gpt5.6','gpt6','gpt6.1','opus5','sonnet46']){
   ctx.state.gptVariant=variant;ctx.state.claudeVariant=variant;
   assert.equal(ctx.kindMatches({test_kind:kind,always_visible:true}),true);
  }
 }
});

test('manual Kimi schema suite is selectable without changing other execution faces',()=>{
 const card=item('KVV_TOOL_SCHEMA',{test_kind:'kvv',manual_execution:true,scoring_excluded:true,
  admission:'本专项不参与模型评分',faces:{channel_count:1,counts:{channel:1}}});
 const {ctx,element}=harness([card]);ctx.state.testKind='kvv';
 ctx.renderCases();ctx.selectCase(card.id);
 assert.equal(ctx.state.selected.join(','),card.id);
 assert.equal(ctx.state.faceChannel[card.id],undefined);
 assert.doesNotMatch(element('#caseDetail').innerHTML,/case-face-toggle/);
 assert.match(element('#caseEvidence').innerHTML,/不参与模型评分/);
 card.unsupported=true;ctx.state.selected=[];ctx.selectCase(card.id);
 assert.equal(ctx.state.selected.length,0,'protocol restrictions still apply');
});
