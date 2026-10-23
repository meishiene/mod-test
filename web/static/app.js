const state = { protocols: [], protocol: 'openai', customPath: '', testKind: 'load', cases: [], activeCase: '', selected: [], temporaryCase: '', billingProbe: '', caseModels: [], claudeVariant: 'fable5', gptVariant: 'gpt5.6', geminiModel: 'gemini-3.8-flash', qwenModel: 'qwen3.8-max',
  // 执行面：默认只跑「模型能力 + 模型验真」。faceChannel 记录哪些卡片被显式勾选了
  // 「含通道面」——通道面＝网关端点/第二协议/beta/参数校验，中转通常不实现，
  // 失败只记网关缺口、不阻断准入。
  faceChannel: {},
  runs: [], runsTotal: 0, runsPage: 1, runsPageSize: 20, filter: 'all', conclusion: 'all', runsSearch: '', models: [], downloadSelected: new Set(), chatRunId: null, chatRawText: '', chatRawView: false,
  // Seedance 上传图片：后端只认本服务上传目录里的相对路径，这里保存服务端返回值 + 展示名。
  seedanceImagePath: '', seedanceImageName: '', seedanceUploading: false, seedanceUploadError: '',
  seedanceVideoPath: '', seedanceVideoName: '', seedanceVideoUploading: false, seedanceVideoError: '' };
const CLAUDE_VARIANT_MODELS = { fable5: 'claude-fable-5', opus5: 'claude-opus-5', sonnet5: 'claude-sonnet-5', haiku45: 'claude-haiku-4-5', sonnet46: 'claude-sonnet-4-6' };
// 「GPT 测试」模块内的档位组：用例列表按 gpt_variant 切换，MODEL 默认填该档位主型号 ID。
// gpt6 = GPT-6 家族（sol / astra / luna），档位相关判据由后端 ModelTier 矩阵按被测模型解析。
const GPT_VARIANT_MODELS = { 'gpt5.6': 'gpt-5.6', 'gpt6.1': 'gpt-6.1-sol', 'gpt6': 'gpt-6-sol' };
// 具体型号候选取自 /api/cases 的 gpt_models（含可用性标记，无通道型号会置灰）；
// 该字段缺失时用这里的兜底，至少保证存在一个可选型号。
const GPT_VARIANT_MODEL_FALLBACK = {
  'gpt5.6': [{id:'gpt-5.6',label:'GPT-5.6（别名 → sol）',available:true}],
  'gpt6.1': [{id:'gpt-6.1-sol',label:'GPT-6.1 Sol（单快照）',available:true}],
  'gpt6': [{id:'gpt-6-sol',label:'GPT-6 Sol',available:true},{id:'gpt-6-astra',label:'GPT-6 Astra',available:true},
           {id:'gpt-6-luna',label:'GPT-6 Luna · 网关无通道（留位不可选）',available:false}],
};
// 「Gemini 测试」模块：3.8 Flash / 3.1 Pro **共用同一批用例**，档位相关期望值由后端
// gemini3/core.py 的 GeminiTier 矩阵按被测模型解析（同 GPT-6 家族做法），因此只有一个档位组，
// 前端只提供一级「模型」下拉；切换模型只改 MODEL 默认值，不改用例列表。
const GEMINI3_MODEL_FALLBACK = {
  id: 'gemini-3.8-flash',
  models: [
    {id:'gemini-3.8-flash',      label:'Gemini 3.8 Flash（thinking low/medium/high，默认 medium）', available:true},
    {id:'gemini-3.1-pro-preview',label:'Gemini 3.1 Pro（thinking low/medium/high，默认 high）',    available:true},
  ],
};
// 「千问测试」模块：六款型号（3.6/3.7/3.8 × flash/plus/max）共用同一批用例，
// 档位相关判据由后端 qwen/cases.py 的 QWEN_MODELS 矩阵解析；切换模型只改 MODEL 默认值。
const QWEN_MODEL_FALLBACK = {
  id: 'qwen3.8-max',
  models: [
    {id:'qwen3.8-max',   label:'Qwen3.8 Max（旗舰）', available:true},
    {id:'qwen3.8-flash', label:'Qwen3.8 Flash（轻量）', available:true},
    {id:'qwen3.7-max',   label:'Qwen3.7 Max（旗舰）', available:true},
    {id:'qwen3.7-plus',  label:'Qwen3.7 Plus（均衡）', available:true},
    {id:'qwen3.6-plus',  label:'Qwen3.6 Plus（均衡）', available:true},
    {id:'qwen3.6-flash', label:'Qwen3.6 Flash（轻量）', available:true},
  ],
};
const $ = (s) => document.querySelector(s);
const escapeHtml = (s='') => String(s).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const FORM_STORAGE_KEY = 'ml-test-platform.form.v1';

function readPersistedForm() {
  try {
    const value = JSON.parse(localStorage.getItem(FORM_STORAGE_KEY) || 'null');
    return value && typeof value === 'object' ? value : {};
  } catch (e) {
    return {};
  }
}

function persistForm() {
  try {
    const values = {
      protocol: state.protocol,
      customPath: state.customPath,
      testKind: state.testKind,
      selected: state.selected,
      temporaryCase: state.temporaryCase,
      billingProbe: state.billingProbe,
      caseModels: state.caseModels,
      faceChannel: state.faceChannel,
      claudeVariant: state.claudeVariant,
      gptVariant: state.gptVariant,
      geminiModel: state.geminiModel,
      qwenModel: state.qwenModel,
      baseUrl: $('#baseUrl').value,
      apiKey: $('#apiKey').value,
      model: $('#model').value,
      quick: $('#quick').checked,
      customEnabled: $('#customEnabled').checked,
      customRequests: $('#customRequests').value,
      customConcurrency: $('#customConcurrency').value,
      gptRequests: $('#gptRequests').value,
      gptConcurrency: $('#gptConcurrency').value,
      moderationConcurrency: $('#moderationConcurrency').value,
      chatPrompt: $('#chatPrompt').value,
      chatSystem: $('#chatSystem').value,
      chatMaxTokens: $('#chatMaxTokens').value,
      chatTemperature: $('#chatTemperature').value,
      signatureMaxTokens: $('#signatureMaxTokens').value,
      imageQuality: selectedValues('#imageQuality'),
      imageModels: selectedValues('#imageModels'),
      imageSizes: selectedValues('#imageSizes'),
      imagePrompt: $('#imagePrompt').value,
      imageHoldEnabled: $('#imageHoldEnabled').checked,
      imageHoldSeconds: $('#imageHoldSeconds').value,
      geminiQuality: selectedValues('#geminiQuality'),
      geminiModels: selectedValues('#geminiModels'),
      geminiAspects: selectedValues('#geminiAspects'),
      geminiCounts: selectedValues('#geminiCounts'),
      seedancePrompt: $('#seedancePrompt')?.value,
      seedanceMode: $('#seedanceMode')?.value,
      seedanceResolution: $('#seedanceResolution')?.value,
      seedanceRatio: $('#seedanceRatio')?.value,
      seedanceDuration: $('#seedanceDuration')?.value,
      seedanceImageRole: $('#seedanceImageRole')?.value,
      seedanceImageUrl: $('#seedanceImageUrl')?.value,
      seedanceVideoRole: $('#seedanceVideoRole')?.value,
      seedanceVideoUrl: $('#seedanceVideoUrl')?.value,
      seedanceAudio: $('#seedanceAudio')?.checked,
      seedanceWatermark: $('#seedanceWatermark')?.checked
    };
    localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(values));
  } catch (e) {
    // Private browsing or a disabled storage API should not break the form.
  }
}


function inlineMarkdown(source='') {
  let text=escapeHtml(source);
  text=text.replace(/`([^`]+)`/g,'<code>$1</code>');
  text=text.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  text=text.replace(/\*([^*]+)\*/g,'<em>$1</em>');
  // 同时支持站内相对链接：报告的「产物」小节用 /api/runs/<id>/artifact/<name> 给出
  // 可直接打开的生成结果（https? 之外的斜杠开头写法原样保留，已被 escapeHtml 转义）。
  text=text.replace(/\[([^\]]+)\]\(((?:https?:\/\/|\/)[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return text;
}
function renderMarkdown(source='') {
  const lines=String(source).replace(/\r/g,'').split('\n'); let html='', paragraph=[], list=null, quote=[];
  const flushParagraph=()=>{if(paragraph.length){html+=`<p>${paragraph.map(inlineMarkdown).join('<br>')}</p>`;paragraph=[]}};
  const flushList=()=>{if(list){html+=`<${list.type}>${list.items.map(x=>`<li>${inlineMarkdown(x)}</li>`).join('')}</${list.type}>`;list=null}};
  const flushQuote=()=>{if(quote.length){html+=`<blockquote>${quote.map(inlineMarkdown).join('<br>')}</blockquote>`;quote=[]}};
  for(let i=0;i<lines.length;i++){
    const line=lines[i];
    if(line.startsWith('```')){flushParagraph();flushList();flushQuote();const lang=line.slice(3).trim();const code=[];while(++i<lines.length&&!lines[i].startsWith('```'))code.push(lines[i]);html+=`<pre data-lang="${escapeHtml(lang)}"><code>${escapeHtml(code.join('\n'))}</code></pre>`;continue}
    if(/^\s*\|/.test(line)&&i+1<lines.length&&/^\s*\|?\s*:?-+/.test(lines[i+1])){flushParagraph();flushList();flushQuote();const table=[];table.push(line);i++;while(++i<lines.length&&/^\s*\|/.test(lines[i]))table.push(lines[i]);i--;const cells=row=>row.trim().replace(/^\||\|$/g,'').split('|').map(x=>x.trim());const head=cells(table[0]);const body=table.slice(1).map(cells);html+='<div class="md-table-wrap"><table><thead><tr>'+head.map(x=>`<th>${inlineMarkdown(x)}</th>`).join('')+'</tr></thead><tbody>'+body.map(row=>'<tr>'+row.map(x=>`<td>${inlineMarkdown(x)}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';continue}
    const heading=line.match(/^(#{1,4})\s+(.+)$/);if(heading){flushParagraph();flushList();flushQuote();const n=heading[1].length;html+=`<h${n}>${inlineMarkdown(heading[2])}</h${n}>`;continue}
    if(/^\s*---+\s*$/.test(line)){flushParagraph();flushList();flushQuote();html+='<hr>';continue}
    const bullet=line.match(/^\s*[-*]\s+(.+)$/),numbered=line.match(/^\s*\d+[.)]\s+(.+)$/);
    if(bullet||numbered){flushParagraph();flushQuote();const type=bullet?'ul':'ol';if(list&&list.type!==type)flushList();list=list||{type,items:[]};list.items.push((bullet||numbered)[1]);continue}else flushList();
    const q=line.match(/^>\s?(.*)$/);if(q){flushParagraph();quote.push(q[1]);continue}else flushQuote();
    if(!line.trim()){flushParagraph();continue}paragraph.push(line.trim());
  }
  flushParagraph();flushList();flushQuote();return html;
}

async function api(path, options={}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}
function toast(message) { const el=$('#toast'); el.textContent=message; el.classList.add('show'); setTimeout(()=>el.classList.remove('show'),2600); }
function caseLabel(id) { return state.cases.find(item=>item.id===id)?.title || id; }
function kindLabel(kind){return kind==='open-source'?'开源测试集':kind==='kvv'?'Kimi 测试':kind==='deepseek'?'DeepSeek V4':kind==='glm'?'GLM 5.3':kind==='qwen'?'千问测试':kind==='gpt'?'GPT 测试':kind==='gemini'?'Gemini 测试':kind==='gpt-image'?'GPT 图片压测':kind==='fable'?'Claude 测试':kind==='gemini-image'?'Gemini 图片测试':kind==='seedance'?'视频生成':kind==='glm_stream_retest'?'GLM 流式复测':kind==='awsb'?'AWSB / Opus 5':kind==='moderation'?'内容审核':kind==='chat'?'对话窗口':kind==='forensics'?'中转验真':kind==='authenticity'?'满血真实性校验':kind==='arxivmath'?'ArXivMath':kind==='deepswe'?'DeepSWE':kind==='swe-atlas-tw'?'SWE-Atlas TW':kind==='swe-atlas-rf'?'SWE-Atlas RF':kind==='legal-agent-benchmark'?'Legal Agent Benchmark':kind==='strict'?'多轮压测（严格）':'多轮压测'}
function kindMatches(item){
  if(state.testKind==='open-source')return ['arxivmath','deepswe','swe-atlas-tw','swe-atlas-rf','legal-agent-benchmark'].includes(item.test_kind);
  // always_visible：Claude 模块下常驻的案例（如中转计费探针），不随模型变体切换。
  if(state.testKind==='fable')return item.test_kind==='fable'&&(item.always_visible||(item.claude_variant||'fable5')===state.claudeVariant);
  // GPT 模块下按模型档位（GPT-5.6 / GPT-6.1）切换用例列表。
  if(state.testKind==='gpt')return item.test_kind==='gpt'&&(item.always_visible||(item.gpt_variant||'gpt5.6')===state.gptVariant);
  return item.test_kind===state.testKind;
}


function renderProtocols() {
  const select=$('#protocolSelect'), providers=[...new Set(state.protocols.map(item=>item.provider))];
  select.innerHTML=providers.map(provider=>`<optgroup label="${escapeHtml(provider)}">${state.protocols.filter(item=>item.provider===provider).map(item=>`<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)} · ${escapeHtml(item.path_template)}</option>`).join('')}</optgroup>`).join('')+'<optgroup label="自定义"><option value="__custom__">自定义请求路径…</option></optgroup>';
  select.value=state.customPath?'__custom__':state.protocol;
  select.onchange=()=>selectProtocol(select.value);
}
function protocolSupports(item, protocolId=state.protocol) {
  const protocol=state.protocols.find(x=>x.id===protocolId);
  return Boolean(protocol && item.protocols.includes(protocol.family) && (!protocol.test_kinds.length || protocol.test_kinds.includes(item.test_kind)));
}
function updateEndpointPreview() {
  const item=state.protocols.find(x=>x.id===state.protocol), preview=$('#endpointPreview');
  if(!item)return;
  const base=$('#baseUrl').value.trim().replace(/\/+$/,'');
  const model=$('#model').value.trim();
  const path=(state.customPath||item.path_template).replace('{model}',model?encodeURIComponent(model):'{model}');
  preview.textContent=base?`请求地址：${base}${path}`:`路径将自动拼装：${path}`;
}
// 自定义请求路径：下拉里选「自定义请求路径…」后出现输入框；路径只覆盖请求地址，
// 鉴权与报文格式仍按底层协议（state.protocol 不变，用例列表随之不变）。
function syncCustomPathField(){
  const item=state.protocols.find(x=>x.id===state.protocol);
  const field=$('#customPathField'), input=$('#customPath');
  field.style.display=state.customPath?'':'none';
  if(state.customPath&&!input.value)input.value=state.customPath;
  $('#protocolSelect').value=state.customPath?'__custom__':state.protocol;
  if(state.customPath&&item)$('#protocolDescription').textContent=`自定义路径 · 基于 ${item.provider} ${item.name} 协议`;
}
function selectProtocol(id) {
  if(id==='__custom__'){
    const item=state.protocols.find(x=>x.id===state.protocol); if(!item)return;
    if(!state.customPath)state.customPath=item.path_template;
    syncCustomPathField(); updateEndpointPreview(); persistForm(); return;
  }
  state.customPath=''; $('#customPathField').style.display='none'; $('#customPath').value='';
  const previous=state.protocols.find(x=>x.id===state.protocol); const item=state.protocols.find(x=>x.id===id); if(!item)return; state.protocol=id; renderProtocols();
  const base=$('#baseUrl'); if(!base.value || (previous && base.value===previous.default_base_url)) base.value=item.default_base_url;
  $('#protocolDescription').textContent=`${item.provider} · ${item.description}`; updateEndpointPreview();
  setModelPlaceholder(item); renderCases(); state.selected=state.selected.filter(caseId=>{const selected=state.cases.find(x=>x.id===caseId);return selected&&protocolSupports(selected,id)}); if(!state.selected.length){const first=state.cases.find(x=>kindMatches(x)&&protocolSupports(x,id));if(first)state.selected=[first.id];} renderCases();updateSelectionUI(); state.models=[];renderTestModels();hideModelSuggestions(); $('#modelHint').textContent='填写接口后查询可用模型，在下方勾选测试模型'; $('#modelHint').className='field-hint'; persistForm();
}

function setModelPlaceholder(protocol){
  const item=protocol || state.protocols.find(x=>x.id===state.protocol) || {};
  $('#model').placeholder=state.testKind==='deepseek'?'输入任意 DeepSeek 模型名或网关模型 ID':
    state.testKind==='glm'?'输入 GLM 5.3 模型名或网关模型 ID':
    state.testKind==='qwen'?`输入 ${state.qwenModel || 'qwen3.8-max'} 或网关模型 ID/别名`:
    state.testKind==='gpt'?`输入 ${GPT_VARIANT_MODELS[state.gptVariant]} 或网关模型 ID`:
    state.testKind==='fable'?`输入 ${CLAUDE_VARIANT_MODELS[state.claudeVariant]} 或网关模型别名`:
    state.testKind==='gemini'?`输入 ${geminiModelOptions().find(x=>x.id===state.geminiModel)?state.geminiModel:'gemini-3.8-flash'} 或网关模型 ID/别名`:
    state.testKind==='gpt-image'?'图片模型/接入点 ID（请求模型按下方选项随机）':
    state.testKind==='gemini-image'?'Gemini 图像模型/接入点 ID（请求模型按下方选项随机）':
    state.testKind==='seedance'?'输入视频模型 ID（该面文档可选值：Seedance-2.0、Seedance-2.5）':
    state.testKind==='authenticity'?'输入 DeepSeek 或 Kimi K3 模型名/网关模型 ID':
    state.testKind==='moderation'?'输入审核模型或网关模型 ID':
    state.testKind==='open-source'?'输入用于运行开源测试集的模型或网关模型 ID':
    state.testKind==='chat'?'输入用于对话的模型或网关模型 ID':
    item.model_hint||'';
}

function caseFacts(item){
  const facts=[['测试案例数',item.case_count_text || (item.case_count != null ? `${item.case_count} 条` : null)],['并发 / 速率',item.concurrency_text],['输入 Token',item.input_tokens],['输出 Token 上限',item.output_tokens],['总请求数',item.request_text]];
  if(item.warmup)facts.push(['预热时间',item.warmup]);
  if(item.steady)facts.push(['稳定运行',item.steady]);
  if(item.workload_profile)facts.push(['负载画像',item.workload_profile]);
  if(item.show_expected_model)facts.push(['校验模型',$('#model').value.trim() || (item.test_kind==='fable'?CLAUDE_VARIANT_MODELS[state.claudeVariant]:'待填写')]);
  return facts;
}
function caseScopeField(item){
  // 子项选择的存放位置按案例类型分流：Claude 模块下的中转计费探针 vs
  // 「Kimi 测试」模块下的 K3 临时案例集（两者都是"一个入口 + 多个子项"的清单案例）。
  return item&&item.billing_probe?'billingProbe':'temporaryCase';
}
function caseScopeOptions(item){
  const options=item.case_options||[];
  const current=state[caseScopeField(item)];
  const lead=item.billing_probe?'':'<option value="">全部 K3 合并案例</option>';
  const buckets=[];
  options.forEach(option=>{
    const label=option.group||'可选案例';
    let bucket=buckets.find(x=>x.label===label);
    if(!bucket){bucket={label,items:[]};buckets.push(bucket);}
    bucket.items.push(option);
  });
  // 计费探针默认选中整组入口，避免「选了卡片却没选范围」的歧义；K3 案例集默认整组。
  const isSelected=option=>current?current===option.id:(item.billing_probe&&option.id===(options[0]||{}).id);
  return lead+buckets.map(bucket=>`<optgroup label="${escapeHtml(bucket.label)}">${bucket.items.map(option=>`<option value="${escapeHtml(option.id)}" ${isSelected(option)?'selected':''}>${escapeHtml(option.id)} · ${escapeHtml(option.name)}${option.merged?`（${option.repetitions} 次）`:''}</option>`).join('')}</optgroup>`).join('');
}
// 执行范围（签名校验卡片专属）：选项 = 「查询模型」拿到的该 Key 全部模型，可多选。
// 多选提交后按模型拆子任务成批执行，报告每个模型一行（batchSuiteTable 的多模型分支）。
function caseModelScopeOptions(){
  const models=state.models||[];
  if(!models.length) return '<div class="case-model-empty">请先点击「查询模型」获取该 Key 的可用模型</div>';
  const current=state.caseModels||[];
  return models.map(model=>`<label class="case-model-option"><input type="checkbox" class="case-model-check" value="${escapeHtml(model.id)}" ${current.includes(model.id)?'checked':''}><span>${escapeHtml(model.id)}${model.name&&model.name!==model.id?` · ${escapeHtml(model.name)}`:''}</span></label>`).join('');
}
// 成本只露档位（低/中/高/极高），括号里的口径说明不进卡片，避免头部被撑长。
function costLevel(cost){
  const text=String(cost||'');
  const cut=text.search(/[（(]/);
  return (cut>=0?text.slice(0,cut):text).trim()||text;
}
function caseAvailability(item) {
  const onlyChannel=Boolean(item.faces?.channel_count)&&!(item.faces.counts.model||item.faces.counts.verify);
  const protocol=protocolSupports(item);
  return {onlyChannel,supported:protocol&&(item.manual_execution||!onlyChannel||Boolean(state.faceChannel[item.id])),
    reason:!protocol?'当前协议不支持':onlyChannel&&!item.manual_execution&&!state.faceChannel[item.id]?'需启用通道面':''};
}
function availableCasePresets(cases=state.cases.filter(kindMatches)) {
  return TestCasePresets.forCases(state.testKind,cases).map(preset=>{
    const members=preset.caseIds.map(id=>cases.find(item=>item.id===id));
    const blocked=members.find(item=>!caseAvailability(item).supported);
    const exclusive=members.length>1&&members.some(item=>TestCasePresets.isExclusive(state.testKind,item));
    const reason=blocked?`${blocked.title}：${caseAvailability(blocked).reason}`:exclusive?'包含需单独运行的测试项':'';
    return {...preset,supported:!reason,reason,members};
  });
}
function renderCasePresets(cases) {
  const box=$('#casePresets');if(!box)return;
  const presets=availableCasePresets(cases);
  const active=presets.find(preset=>preset.supported&&preset.caseIds.length===state.selected.length&&preset.caseIds.every(id=>state.selected.includes(id)));
  $('#casePresetHeading').textContent=presets.length?'常用组合':'自由选择';
  box.hidden=!presets.length;
  box.innerHTML=presets.map(preset=>`<button type="button" class="test-preset-button" data-preset="${escapeHtml(preset.id)}" aria-pressed="${active?.id===preset.id}" ${preset.supported?'':'disabled'} title="${escapeHtml(preset.reason||preset.members.map(item=>item.title).join(' + '))}"><span>${escapeHtml(preset.label)}</span><small>${preset.caseIds.length} 项</small>${preset.reason?`<span class="test-preset-unavailable">${escapeHtml(preset.reason)}</span>`:''}</button>`).join('');
  box.querySelectorAll('[data-preset]').forEach(button=>button.onclick=()=>{
    applyCasePreset(button.dataset.preset);
    [...box.querySelectorAll('[data-preset]')].find(next=>next.dataset.preset===button.dataset.preset)?.focus();
  });
  const clear=$('#clearCaseSelection');clear.disabled=!state.selected.length;clear.onclick=()=>{
    clearCaseSelection();$('#caseGrid').querySelector('.test-case-button:not(:disabled)')?.focus();
  };
  $('#casePresetStatus').textContent=active?`当前组合：${active.label} · 可点单项调整`:state.selected.length?`自定义选择 · ${state.selected.length} 项`:'未选择测试项';
}
function applyCasePreset(id) {
  const preset=availableCasePresets().find(item=>item.id===id);
  if(!preset?.supported)return;
  state.selected=[...preset.caseIds];
  if(!state.selected.includes(state.activeCase))state.activeCase=state.selected[0];
  renderCases();updateSelectionUI();persistForm();
  const grid=$('#caseGrid');
  const first=[...grid.querySelectorAll('.test-case-button')].find(button=>button.dataset.case===state.selected[0]);
  if(first)grid.scrollTop=first.offsetTop-grid.offsetTop;
}
function clearCaseSelection() {
  state.selected=[];renderCases();updateSelectionUI();persistForm();
}
function restoreCaseSelection(saved) {
  const cases=state.cases.filter(item=>kindMatches(item)&&caseAvailability(item).supported);
  state.selected=Array.isArray(saved)?[...new Set(saved.filter(id=>cases.some(item=>item.id===id)))]:[];
  const exclusive=state.selected.find(id=>TestCasePresets.isExclusive(state.testKind,cases.find(item=>item.id===id)));
  if(exclusive)state.selected=[exclusive];
  if(!state.selected.length&&(!Array.isArray(saved)||saved.length)&&cases.length)state.selected=[cases[0].id];
}
function testModelMultiple() {
  return state.selected.length>0&&state.selected.every(id=>state.cases.find(item=>item.id===id)?.model_scope);
}
function renderTestModels() {
  const box=$('#testModelOptions');if(!box)return;
  const selectedItem=state.cases.find(item=>item.id===state.activeCase&&kindMatches(item))||state.cases.find(item=>state.selected.includes(item.id));
  // 查看详情不改变执行选择；预览模型来自当前测试项，不能借用另一项的执行参数。
  if(['gpt-image','gemini-image'].includes(state.testKind)&&selectedItem&&(!state.selected.includes(selectedItem.id)||selectedItem.fixed_parameters)){
    const fixed=Boolean(selectedItem.fixed_parameters),selected=state.selected.includes(selectedItem.id);
    $('#testModelSearch').hidden=true;
    $('#testModelMode').textContent=fixed?'固定测试模型':'可用模型预览';
    $('#testModelNote').textContent=fixed?'此专项按预设模型执行，具体参数见下方准入标准。':'加入本次测试后，可勾选这里的模型进行测试。';
    box.innerHTML=(selectedItem.image_models||[]).map(model=>`<label class="test-model-option"><input type="checkbox" disabled ${selected?'checked':''}><span>${escapeHtml(model)}</span></label>`).join('');
    return;
  }
  const imageSelector=state.testKind==='gpt-image'&&!selectedItem?.fixed_parameters?'#imageModels':state.testKind==='gemini-image'?'#geminiModels':null;
  if(imageSelector&&selectedItem){
    const select=$(imageSelector);
    $('#testModelSearch').hidden=true;
    $('#testModelMode').textContent='随机抽取已勾选模型';
    $('#testModelNote').textContent='图片请求从这里勾选的模型集合中随机抽取，沿用原有图片测试规则。';
    box.innerHTML=Array.from(select.options).map(option=>`<label class="test-model-option"><input type="checkbox" class="test-image-model-check" value="${escapeHtml(option.value)}" ${option.selected?'checked':''}><span>${escapeHtml(option.textContent||option.value)}</span></label>`).join('');
    box.querySelectorAll('.test-image-model-check').forEach(input=>input.onchange=()=>{
      Array.from(select.options).find(option=>option.value===input.value).selected=input.checked;
      persistForm();
    });
    return;
  }
  const current=$('#model').value.trim(),multiple=testModelMultiple();
  const models=[...(state.models||[])];
  for(const id of [current,...(multiple?state.caseModels:[])])if(id&&!models.some(m=>m.id===id))models.unshift({id,name:id});
  const query=$('#testModelSearch').value.trim().toLowerCase();
  const visible=models.filter(m=>`${m.id} ${m.name||''}`.toLowerCase().includes(query));
  $('#testModelSearch').hidden=models.length<6&&!query;
  $('#testModelMode').textContent=multiple?'支持多模型对照':'单模型执行';
  $('#testModelNote').textContent=multiple?'勾选多个模型后逐个执行，每个模型分别生成报告。':'本次已选测试按一个模型执行；可查询后勾选，也可直接填写模型 ID。';
  box.innerHTML=visible.length?visible.map(m=>`<label class="test-model-option"><input type="checkbox" class="test-model-check" value="${escapeHtml(m.id)}" ${(multiple&&state.caseModels.length?state.caseModels.includes(m.id):m.id===current)?'checked':''}><span>${escapeHtml(m.id)}</span></label>`).join(''):`<p class="test-config-note">${query?'没有匹配的模型。':'填写模型 ID，或点击「查询模型」加载可选项。'}</p>`;
  box.querySelectorAll('.test-model-check').forEach(input=>input.onchange=()=>{
    if(multiple){
      const selected=new Set(state.caseModels.length?state.caseModels:current?[current]:[]);
      if(input.checked)selected.add(input.value);else selected.delete(input.value);
      state.caseModels=[...selected];
      $('#model').value=state.caseModels[0]||'';
    }else{
      $('#model').value=input.value;
      state.caseModels=[];
    }
    hideModelSuggestions();updateEndpointPreview();updateSelectionUI();persistForm();
  });
}
function renderCaseDetails() {
  const item=state.cases.find(x=>x.id===state.activeCase&&kindMatches(x));
  if(!$('#caseDetail'))return;
  if(!item){$('#caseDetail').innerHTML='<p class="empty">暂无测试项，请选择其他测试类别。</p>';$('#caseEvidence').innerHTML='';$('#testExecutionConfigs').hidden=true;renderTestModels();return}
  const availability=caseAvailability(item),selected=state.selected.includes(item.id),facts=caseFacts(item);
  $('#caseDetail').innerHTML=`<header class="test-detail-heading"><div><span class="test-detail-id">${escapeHtml(item.id)}</span><h3>${escapeHtml(item.title)}</h3></div><button type="button" data-toggle-active-case ${availability.supported?'':'disabled'}>${selected?'移出本次测试':'加入本次测试'}</button></header>
    <p class="test-detail-status"><span class="${selected?'test-included':''}">${selected?'已加入本次测试':availability.reason||'当前仅查看详情'}</span><span class="test-cost">成本 ${escapeHtml(costLevel(item.cost))}</span></p>
    ${selected&&state.selected.length>1?'<p class="test-config-note">本次已选择 '+state.selected.length+' 个测试项，模型选择和可编辑执行参数按本次测试统一配置。</p>':''}
    ${!selected?'<p class="test-config-note">点击左侧组合、测试项或「加入本次测试」后配置执行参数。</p>':''}
    ${item.case_options?`<label class="case-scope-picker"><span>${escapeHtml(item.options_label||'执行范围')}</span><select class="case-scope-select" data-case="${escapeHtml(item.id)}">${caseScopeOptions(item)}</select></label>`:''}
    ${item.faces?.channel_count&&!item.manual_execution?`<label class="case-scope-picker case-face-picker"><span>测试面</span><input type="checkbox" class="case-face-toggle" data-case="${escapeHtml(item.id)}" ${state.faceChannel[item.id]?'checked':''}><b class="case-face-label">含通道面（+${item.faces.channel_count} 条）</b><small>${availability.onlyChannel?'此项全部为通道契约测试，启用后可选择执行。':'默认测试模型能力与验真；通道面单独计分。'}</small></label>`:''}`;
  $('#caseDetail').querySelector('[data-toggle-active-case]').onclick=()=>selectCase(item.id);
  const picker=$('.case-scope-select');
  if(picker)picker.onchange=event=>{state[caseScopeField(item)]=event.target.value;persistForm()};
  const face=$('.case-face-toggle');
  if(face)face.onchange=()=>{
    if(face.checked)state.faceChannel[item.id]=true;else delete state.faceChannel[item.id];
    if(!face.checked&&availability.onlyChannel)state.selected=state.selected.filter(id=>id!==item.id);
    renderCases();updateSelectionUI();persistForm();
  };
  $('#testExecutionConfigs').hidden=!selected;
  $('#caseEvidence').innerHTML=`<details class="test-execution-facts" open><summary>执行参数与准入标准</summary><dl class="case-facts">${facts.map(([label,value])=>`<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value??'—')}</dd></div>`).join('')}</dl><div class="admission-standard"><strong>准入标准</strong><p>${escapeHtml(item.admission||'以报告中的全部检查项通过为准。')}</p></div><p class="test-config-note">${escapeHtml(item.source||'专项自研代码')} · ${escapeHtml(item.executor||'')}</p>${item.fixed_parameters||!['load','gpt-image','gemini-image','seedance','moderation','chat'].includes(state.testKind)&&!item.multi_round?'<p class="test-config-note">此专项使用预设并发与测试提示词；可调整的参数在上方展示。</p>':''}</details>`;
  renderTestModels();
}
function renderCases() {
  const cases=state.cases.filter(item=>kindMatches(item));
  if(!cases.some(item=>item.id===state.activeCase))state.activeCase=cases.find(item=>state.selected.includes(item.id))?.id||cases[0]?.id||'';
  $('#testCaseCount').textContent=`${cases.length} 项 · 已选 ${cases.filter(item=>state.selected.includes(item.id)).length}`;
  renderCasePresets(cases);
  $('#caseGrid').innerHTML=cases.map(item=>{
    const availability=caseAvailability(item),selected=state.selected.includes(item.id),active=item.id===state.activeCase;
    return `<div class="test-case-row ${active?'active':''} ${selected?'selected':''}"><button type="button" class="test-case-button" data-case="${escapeHtml(item.id)}" aria-pressed="${selected}" ${availability.supported?'':'disabled'}><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.id)}</span><span class="test-case-meta"><small class="test-cost">成本 ${escapeHtml(costLevel(item.cost))}</small><small class="test-case-selection">${selected?'<span aria-hidden="true">✓</span> 已加入':'未选择'}</small>${availability.reason?`<small class="test-case-availability">${escapeHtml(availability.reason)}</small>`:''}</span></button><button type="button" class="test-case-detail-button" data-case="${escapeHtml(item.id)}" aria-label="${active?'正在查看':'查看'} ${escapeHtml(item.title)} 详情" aria-controls="caseDetail" aria-pressed="${active}" ${active?'aria-current="true"':''}>${active?'查看中':'详情'}<span aria-hidden="true"> ${active?'→':'›'}</span></button></div>`;
  }).join('')||'<p class="empty">暂无匹配的测试项</p>';
  $('#caseGrid').querySelectorAll('.test-case-button').forEach(button=>button.onclick=()=>{
    selectCase(button.dataset.case);
    [...$('#caseGrid').querySelectorAll('.test-case-button')].find(next=>next.dataset.case===button.dataset.case)?.focus();
  });
  $('#caseGrid').querySelectorAll('.test-case-detail-button').forEach(button=>button.onclick=()=>{
    state.activeCase=button.dataset.case;renderCases();
    [...$('#caseGrid').querySelectorAll('.test-case-detail-button')].find(next=>next.dataset.case===button.dataset.case)?.focus();
  });
  renderCaseDetails();
}
function selectedValues(selector){
  return [...document.querySelector(selector).selectedOptions].map(option=>option.value);
}
function renderMultiSelects(){
  document.querySelectorAll('[data-multi-select]').forEach(group=>{
    const select=document.getElementById(group.dataset.multiSelect);
    if(!select)return;
    group.innerHTML=[...select.options].map((option,index)=>`<label class="multi-choice-option"><input type="checkbox" data-option-index="${index}" ${option.selected?'checked':''} ${select.disabled||option.disabled?'disabled':''}><span>${escapeHtml(option.textContent)}</span></label>`).join('')||'<span class="field-hint">暂无可选项</span>';
    group.querySelectorAll('input').forEach(input=>input.onchange=()=>{
      const option=select.options[Number(input.dataset.optionIndex)];
      if(!option||select.disabled||option.disabled)return;
      option.selected=input.checked;
      select.dispatchEvent(new Event('change',{bubbles:true}));
    });
  });
}
// 压测持续时长：默认不启用。勾选框没勾上时秒数框不可编辑，且提交时不带该参数。
function syncImageHoldState(){
  const relevant=$('#imageHoldField').style.display!=='none';
  $('#imageHoldEnabled').disabled=!relevant;
  $('#imageHoldSeconds').disabled=!(relevant&&$('#imageHoldEnabled').checked);
}
function renderImageConfig(){
  const box=$('#imageConfig'), item=state.cases.find(x=>state.selected.includes(x.id));
  if(!box) return;
  const visible=state.testKind==='gpt-image' && item && !item.fixed_parameters;
  box.style.display=visible?'grid':'none';
  if(!visible) return;
  const renderSelect=(id, values, selected)=>{const el=$(id);el.innerHTML=(values||[]).map(value=>`<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join('');[...el.options].forEach(option=>option.selected=selected.includes(option.value));};
  const previousQuality=selectedValues('#imageQuality'), previousModels=selectedValues('#imageModels'), previousSizes=selectedValues('#imageSizes');
  // 加入测试只显示可选项，保留手动选择（包括全部取消），不自动全选。
  renderSelect('#imageQuality',item.image_qualities,previousQuality);
  renderSelect('#imageModels',item.image_models,previousModels);
  renderSelect('#imageSizes',item.image_sizes,previousSizes);
  const ratio=$('#imageAspectRatio');
  renderSelect('#imageAspectRatio',item.image_aspect_ratios,[ratio.value||'1:1']);
  ratio.disabled=!$('#imageAspectEnabled').checked;
  // 压测持续时长：默认不启用。只有用户自己动过勾选/数值才保留，否则按用例默认回填。
  if(!$('#imageHoldEnabled').dataset.edited) $('#imageHoldEnabled').checked=item.image_hold_seconds!=null;
  if(!$('#imageHoldSeconds').dataset.edited) $('#imageHoldSeconds').value=String(item.image_hold_seconds||60);
  renderMultiSelects();
}
function renderImagePromptConfig(){
  const item=state.cases.find(x=>state.selected.includes(x.id));
  const visible=['gpt-image','gemini-image'].includes(state.testKind)&&item&&!item.fixed_parameters;
  $('#imagePromptConfig').style.display=visible?'block':'none';
  $('#imagePrompt').disabled=!visible;
  if(visible&&item.image_prompt_max_chars) $('#imagePrompt').maxLength=Number(item.image_prompt_max_chars);
}
function renderGeminiImageConfig(){
  const box=$('#geminiImageConfig'), item=state.cases.find(x=>state.selected.includes(x.id));
  if(!box) return;
  const visible=state.testKind==='gemini-image' && item;
  box.style.display=visible?'grid':'none';
  if(!visible) return;
  const allowed=(values)=> (values||[]).map(value=>String(value));
  const previous=selector=>selectedValues(selector);
  const renderSelect=(selector, pool, selected)=>{
    const el=$(selector);
    el.innerHTML=pool.map(value=>`<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join('');
    [...el.options].forEach(option=>option.selected=selected.includes(option.value));
  };
  const qualities=allowed(item.image_qualities), models=allowed(item.image_models), aspects=allowed(item.image_aspect_ratios), counts=allowed(item.image_counts);
  renderSelect('#geminiQuality',qualities,previous('#geminiQuality'));
  renderSelect('#geminiModels',models,previous('#geminiModels'));
  renderSelect('#geminiAspects',aspects,previous('#geminiAspects'));
  renderSelect('#geminiCounts',counts,previous('#geminiCounts'));
  renderMultiSelects();
}

// —— Seedance 视频生成 ——
// 参数取值域由后端 /api/cases 下发（生成方式 / 分辨率 / 画面比例 / 时长区间 / 位置 / 上限），
// 前端只渲染控件并做一次预校验，服务端仍会二次校验。面板会被 updateCustomDefaults()
// 反复重渲染，因此 renderSeedanceConfig() 必须是幂等的——只回填空值或越界值，
// 不覆盖用户正在编辑的合法输入。
function seedanceCase(){
  return state.cases.find(x=>x.test_kind==='seedance');
}
// 生成方式 ↔ 参数显隐的唯一判定源。提交体也用它（见 runForm 的 videoMode），
// 这样"界面上看不见的输入"绝不会出现在请求里——否则就会重现"声明文生视频却带着图片提交"
// 那种上游跑到执行阶段才拒绝的错配。
function seedanceModeFlags(){
  const mode=$('#seedanceMode')?.value||'text';
  return {mode, image:mode==='image'||mode==='both', video:mode==='video'||mode==='both'};
}
const SEEDANCE_MODE_HINTS={
  text:'仅文生视频：只提交提示词与生成参数，不带图片、不带参考视频（相关输入与位置参数已隐藏）。',
  image:'带参考图：必须提供一张图片（上传或地址二选一）；图片位置决定它是首帧还是参考图。',
  video:'带参考视频：必须填写「视频地址」（异步任务面只接受 URL，本地上传的文件仅归档留痕，不算提交）。',
  both:'两者都带：图片与参考视频地址都必须提供；content 顺序固定为 文本 → 图片 → 视频。'
};
function applySeedanceModeVisibility(){
  const box=$('#seedanceConfig');
  if(!box) return {mode:'text',image:false,video:false};
  const flags=seedanceModeFlags();
  const show=(selector,on)=>{const element=$(selector); if(element) element.style.display=on?'':'none'};
  show('#seedanceImageRoleRow',flags.image);
  show('#seedanceVideoRoleRow',flags.video);
  show('#seedanceSources',flags.image||flags.video);
  document.querySelectorAll('.seedance-image-source').forEach(element=>{element.style.display=flags.image?'':'none'});
  document.querySelectorAll('.seedance-video-source').forEach(element=>{element.style.display=flags.video?'':'none'});
  const hint=$('#seedanceModeHint');
  if(hint) hint.textContent=SEEDANCE_MODE_HINTS[flags.mode]||'';
  return flags;
}
function renderSeedanceConfig(){
  const box=$('#seedanceConfig');
  if(!box) return;
  const visible=state.testKind==='seedance';
  box.style.display=visible?'flex':'none';
  if(!visible) return;
  const item=seedanceCase();
  if(!item) return;
  const fill=(selector,values,current,labels)=>{
    const el=$(selector); if(!el) return;
    const pool=(values||[]).map(String);
    el.innerHTML=pool.map(value=>`<option value="${escapeHtml(value)}">${escapeHtml(labels&&labels[value]?labels[value]:value)}</option>`).join('');
    // 参考视频角色允许空串（= 不带 role 字段），因此这里不能用 `pick!==''` 挡掉空值。
    const want=pool.includes(String(current==null?'':current))?String(current==null?'':current):pool[0];
    if(want!=null) el.value=want;
  };
  const roleLabels={};
  (item.seedance_image_roles||[]).forEach(role=>{roleLabels[role.id]=`${role.label}（${role.id}）`});
  const videoRoleLabels={};
  (item.seedance_video_roles||[]).forEach(role=>{videoRoleLabels[role.id]=role.label||`（${role.id}）`});
  const modeLabels={};
  (item.seedance_modes||[]).forEach(entry=>{modeLabels[entry.id]=entry.label||entry.id});
  fill('#seedanceMode',(item.seedance_modes||[]).map(entry=>entry.id),
    $('#seedanceMode')?.value||item.seedance_default_mode||'text',modeLabels);
  fill('#seedanceResolution',item.seedance_resolutions,$('#seedanceResolution')?.value||item.seedance_default_resolution);
  fill('#seedanceRatio',item.seedance_ratios,$('#seedanceRatio')?.value||item.seedance_default_ratio);
  fill('#seedanceImageRole',(item.seedance_image_roles||[]).map(role=>role.id),$('#seedanceImageRole')?.value||item.seedance_default_image_role,roleLabels);
  const videoRoleSelect=$('#seedanceVideoRole');
  fill('#seedanceVideoRole',(item.seedance_video_roles||[]).map(role=>role.id),
    videoRoleSelect&&videoRoleSelect.options.length?videoRoleSelect.value:item.seedance_default_video_role,
    videoRoleLabels);
  const duration=$('#seedanceDuration');
  if(duration){
    if(item.seedance_duration_min!=null) duration.min=String(item.seedance_duration_min);
    if(item.seedance_duration_max!=null) duration.max=String(item.seedance_duration_max);
    const parsed=Number(duration.value);
    const low=Number(item.seedance_duration_min), high=Number(item.seedance_duration_max);
    if(!duration.value||!Number.isFinite(parsed)||parsed<low||parsed>high)
      duration.value=String(item.seedance_default_duration);
  }
  const prompt=$('#seedancePrompt');
  if(prompt&&item.seedance_prompt_max_chars){
    prompt.maxLength=Number(item.seedance_prompt_max_chars);
    prompt.placeholder=`例如：镜头缓慢推进，一只白鹭在清晨的浅滩上展翅起飞，光线柔和，电影质感（上限 ${item.seedance_prompt_max_chars} 字符）`;
  }
  const file=$('#seedanceImageFile');
  if(file) file.disabled=state.seedanceUploading;
  const videoFile=$('#seedanceVideoFile');
  if(videoFile) videoFile.disabled=state.seedanceVideoUploading;
  // 显隐必须在填完选项之后：否则新渲染的选项可能落在上一次被隐藏的行里。
  applySeedanceModeVisibility();
  renderSeedanceUploadState();
  renderSeedanceVideoState();
}
function renderSeedanceUploadState(){
  const status=$('#seedanceUploadState');
  if(!status) return;
  const item=seedanceCase();
  const limit=item&&item.seedance_image_max_bytes?`，单张上限 ${(Number(item.seedance_image_max_bytes)/1024/1024).toFixed(0)} MB`:'';
  const base=`上传与图片地址二选一；两者都填时以上传的图片为准。生成方式为「带参考图 / 两者都带」时必须有图片（上传或地址）。支持 PNG / JPEG / WebP / GIF / BMP${limit}`;
  if(state.seedanceUploadError){status.textContent=`上传失败：${state.seedanceUploadError}`;status.className='video-source-hint error';return}
  if(state.seedanceUploading){status.textContent='正在上传图片…';status.className='video-source-hint';return}
  if(state.seedanceImagePath){
    status.textContent=`已上传：${state.seedanceImageName||state.seedanceImagePath}（任务结束后服务端自动清理临时文件）`;
    status.className='video-source-hint ok';return;
  }
  status.textContent=base;status.className='video-source-hint';
}
function renderSeedanceVideoState(){
  const status=$('#seedanceVideoState');
  if(!status) return;
  const item=seedanceCase();
  const limit=item&&item.seedance_video_max_bytes?`，单文件上限 ${(Number(item.seedance_video_max_bytes)/1024/1024).toFixed(0)} MB`:'';
  // 提示必须写清"只归档、不提交"：否则"我上传了视频"很容易被当成"上游用到了这个视频"。
  const base=`仅归档留痕（复制进结果目录 inputs/ 并记 sha256），不会提交给上游；生成方式为「带参考视频 / 两者都带」时必须填视频地址（只上传文件不算）。支持 MP4 / WebM / MOV / MKV / M4V${limit}`;
  if(state.seedanceVideoError){status.textContent=`上传失败：${state.seedanceVideoError}`;status.className='video-source-hint error';return}
  if(state.seedanceVideoUploading){status.textContent='正在上传视频…';status.className='video-source-hint';return}
  if(state.seedanceVideoPath){
    status.textContent=`已上传：${state.seedanceVideoName||state.seedanceVideoPath}（仅归档，任务结束后服务端自动清理临时文件）`;
    status.className='video-source-hint ok';return;
  }
  status.textContent=base;status.className='video-source-hint';
}
// 素材走原始 body 通道（POST /api/uploads）：JSON 通道有 64 KiB 上限，base64 必然超限，
// 而且 base64 还会被写进记录库。服务端只接受受控 MIME（图片 8 MiB / 视频 64 MiB）、
// 只认自己上传目录里的文件，落盘名由服务端随机生成（不采信客户端文件名），
// 返回相对仓库根的路径交给子进程读取。
// 图片由执行器读成 data URI 随任务提交；视频（动辄几十 MB）**不转 base64**，只归档留痕——
// 参考视频只有"视频地址"会真的发给上游。
async function postSeedanceUpload(file){
  const response=await fetch('/api/uploads',{method:'POST',
    headers:{'Content-Type':file.type||'application/octet-stream','X-File-Name':encodeURIComponent(file.name||'upload')},
    body:file});
  const data=await response.json();
  if(!response.ok) throw new Error(data.error||'上传失败');
  return data;
}
async function uploadSeedanceImage(file){
  if(!file){state.seedanceImagePath='';state.seedanceImageName='';state.seedanceUploadError='';renderSeedanceUploadState();persistForm();return}
  state.seedanceUploading=true;state.seedanceUploadError='';state.seedanceImagePath='';state.seedanceImageName='';
  renderSeedanceConfig();
  try{
    const data=await postSeedanceUpload(file);
    state.seedanceImagePath=data.path||'';
    state.seedanceImageName=file.name||safeDecode(data.name)||'';
    toast(`图片已上传（${(Number(data.bytes||0)/1024).toFixed(0)} KB）`);
  }catch(e){
    state.seedanceImagePath='';state.seedanceImageName='';state.seedanceUploadError=e.message;
    toast(`上传失败：${e.message}`);
  }finally{
    state.seedanceUploading=false;
    renderSeedanceConfig();
  }
}
async function uploadSeedanceVideo(file){
  if(!file){state.seedanceVideoPath='';state.seedanceVideoName='';state.seedanceVideoError='';renderSeedanceVideoState();persistForm();return}
  state.seedanceVideoUploading=true;state.seedanceVideoError='';state.seedanceVideoPath='';state.seedanceVideoName='';
  renderSeedanceConfig();
  try{
    const data=await postSeedanceUpload(file);
    state.seedanceVideoPath=data.path||'';
    state.seedanceVideoName=file.name||safeDecode(data.name)||'';
    toast(`参考视频已归档（${(Number(data.bytes||0)/1024/1024).toFixed(1)} MB，不会提交给上游）`);
  }catch(e){
    state.seedanceVideoPath='';state.seedanceVideoName='';state.seedanceVideoError=e.message;
    toast(`上传失败：${e.message}`);
  }finally{
    state.seedanceVideoUploading=false;
    renderSeedanceConfig();
  }
}
function safeDecode(value){
  try{return decodeURIComponent(String(value||''))}catch(e){return String(value||'')}
}
function gptModelOptions(variant){
  const item=state.cases.find(x=>x.test_kind==='gpt'&&(x.gpt_variant||'gpt5.6')===variant&&Array.isArray(x.gpt_models)&&x.gpt_models.length);
  return item?item.gpt_models:(GPT_VARIANT_MODEL_FALLBACK[variant]||[]);
}
function renderGptVariantConfig(){
  const box=$('#gptVariantConfig');
  if(box) box.style.display=state.testKind==='gpt'?'flex':'none';
  const select=$('#gptVariantSelect');
  if(select) select.value=state.gptVariant;
  const modelSelect=$('#gptModelSelect');
  if(!modelSelect) return;
  const options=gptModelOptions(state.gptVariant);
  const current=$('#model').value.trim();
  // 无通道型号（available=false）置灰：选中它只会得到整批 blocked 结果。
  modelSelect.innerHTML=options.map(option=>`<option value="${escapeHtml(option.id)}"${option.available===false?' disabled':''} title="${escapeHtml(option.reason||option.label||'')}">${escapeHtml(option.label||option.id)}</option>`).join('');
  const picked=options.find(option=>option.id===current&&option.available!==false)||options.find(option=>option.available!==false);
  if(picked) modelSelect.value=picked.id;
}
function applyGptModel(id){
  const option=gptModelOptions(state.gptVariant).find(x=>x.id===id);
  if(!option||option.available===false) return;
  $('#model').value=id; updateEndpointPreview(); setModelPlaceholder(); renderTestModels(); persistForm();
}
// 产物类用例（鹈鹕骑车 SVG，后端在 /api/cases 里带 multi_round）支持一次跑多轮：
// 只有选中的用例带该字段时才显示「请求轮数 / 并发数」，上限由后端下发，不在前端写死。
function renderPelicanConfig(){
  const box=$('#gptPelicanConfig'), item=state.cases.find(x=>state.selected.includes(x.id)&&x.multi_round);
  if(!box) return;
  box.style.display=item?'flex':'none';
  if(!item) return;
  const rounds=$('#gptRequests'), concurrency=$('#gptConcurrency');
  rounds.max=String(item.multi_round.max_requests); concurrency.max=String(item.multi_round.max_concurrency);
  if(!rounds.value) rounds.value=String(item.multi_round.requests||1);
  if(!concurrency.value) concurrency.value=String(item.multi_round.concurrency||1);
  box.title=item.multi_round.hint||'';
}
$('#gptModelSelect').onchange=event=>applyGptModel(event.target.value);
$('#gptVariantSelect').onchange=event=>{
  state.gptVariant=event.target.value in GPT_VARIANT_MODELS?event.target.value:'gpt5.6';
  state.selected=[];state.activeCase='';
  const modelInput=$('#model'), current=modelInput.value.trim();
  const known=Object.values(GPT_VARIANT_MODELS).concat(gptModelOptions(state.gptVariant).map(x=>x.id));
  if(!current || known.includes(current)) modelInput.value=GPT_VARIANT_MODELS[state.gptVariant];
  setModelPlaceholder(); updateEndpointPreview();
  const first=state.cases.find(x=>kindMatches(x)&&protocolSupports(x));
  if(first)state.selected=[first.id];
  renderCases(); updateSelectionUI(); renderGptVariantConfig(); persistForm();
};
function renderClaudeVariantConfig(){
  const box=$('#claudeVariantConfig');
  if(box) box.style.display=state.testKind==='fable'?'flex':'none';
  const select=$('#claudeVariantSelect');
  if(select) select.value=state.claudeVariant;
}
$('#claudeVariantSelect').onchange=event=>{
  state.claudeVariant=event.target.value in CLAUDE_VARIANT_MODELS?event.target.value:'fable5';
  state.selected=[];state.activeCase='';
  const modelInput=$('#model');
  modelInput.value=CLAUDE_VARIANT_MODELS[state.claudeVariant];
  setModelPlaceholder(); updateEndpointPreview();
  const first=state.cases.find(x=>kindMatches(x)&&protocolSupports(x));
  if(first)state.selected=[first.id];
  renderCases(); updateSelectionUI(); persistForm();
};
// 「Gemini 测试」的两个型号共用同一批用例，模型下拉只决定 MODEL 输入框默认值；
// 期望值随模型变化的部分（thinkingLevel 合法集、不可用参数、默认档）由后端矩阵解析。
function geminiModelOptions(){
  const item=state.cases.find(x=>x.test_kind==='gemini'&&Array.isArray(x.gemini_models)&&x.gemini_models.length);
  return item?item.gemini_models:GEMINI3_MODEL_FALLBACK.models;
}
function renderGeminiVariantConfig(){
  const box=$('#geminiVariantConfig');
  if(box) box.style.display=state.testKind==='gemini'?'flex':'none';
  const select=$('#geminiModelSelect');
  if(!select) return;
  const options=geminiModelOptions();
  const current=$('#model').value.trim();
  // 无通道型号（available=false）置灰：选中它只会得到整批 blocked 结果。
  select.innerHTML=options.map(option=>`<option value="${escapeHtml(option.id)}"${option.available===false?' disabled':''} title="${escapeHtml(option.reason||option.label||'')}">${escapeHtml(option.label||option.id)}</option>`).join('');
  const picked=options.find(option=>option.id===current&&option.available!==false)
    || options.find(option=>option.id===state.geminiModel&&option.available!==false)
    || options.find(option=>option.available!==false);
  if(picked) select.value=picked.id;
}
function applyGeminiModel(id){
  const option=geminiModelOptions().find(x=>x.id===id);
  if(!option||option.available===false) return;
  state.geminiModel=id;
  $('#model').value=id; updateEndpointPreview(); setModelPlaceholder(); renderGeminiVariantConfig(); renderTestModels(); persistForm();
}
$('#geminiModelSelect').onchange=event=>applyGeminiModel(event.target.value);
// 「千问测试」：六款型号共用同一批用例，模型下拉只决定 MODEL 默认值；
// 档位相关判据（思考默认开、上下文区间、图像输入支持面）由后端矩阵与真实探测解析。
function qwenModelOptions(){
  const item=state.cases.find(x=>x.test_kind==='qwen'&&Array.isArray(x.qwen_models)&&x.qwen_models.length);
  return item?item.qwen_models:QWEN_MODEL_FALLBACK.models;
}
function renderQwenVariantConfig(){
  const box=$('#qwenModelConfig');
  if(box) box.style.display=state.testKind==='qwen'?'flex':'none';
  const select=$('#qwenModelSelect');
  if(!select) return;
  const options=qwenModelOptions();
  const current=$('#model').value.trim();
  select.innerHTML=options.map(option=>`<option value="${escapeHtml(option.id)}"${option.available===false?' disabled':''} title="${escapeHtml(option.reason||option.label||'')}">${escapeHtml(option.label||option.id)}</option>`).join('');
  const picked=options.find(option=>option.id===current&&option.available!==false)
    || options.find(option=>option.id===state.qwenModel&&option.available!==false)
    || options.find(option=>option.available!==false);
  if(picked) select.value=picked.id;
}
function applyQwenModel(id){
  const option=qwenModelOptions().find(x=>x.id===id);
  if(!option||option.available===false) return;
  state.qwenModel=id;
  $('#model').value=id; updateEndpointPreview(); setModelPlaceholder(); renderQwenVariantConfig(); renderTestModels(); persistForm();
}
$('#qwenModelSelect').onchange=event=>applyQwenModel(event.target.value);
function updateCustomDefaults(){
  const item=state.cases.find(x=>x.id===state.selected[0]); const box=$('#customLoad');
  const isImage=state.testKind==='gpt-image'||state.testKind==='gemini-image';
  // Seedance 与图片套件同口径：内容参数显式输入，执行规模（视频任务数 / 并发）也走手动输入——
  // 单条视频成本极高，「随机抽取 + 高并发」只会烧钱且结论不可解释。
  const isVideo=state.testKind==='seedance';
  const manualKind=isImage||isVideo;
  if(!box) return; const fixedLoad=Boolean(item?.fixed_parameters); const manualLoad=Boolean(item?.manual_parameters); const enabled=item && ((state.testKind==='load' && !fixedLoad) || (manualKind && !fixedLoad)); box.style.display=enabled?'flex':'none'; const moderationBox=$('#moderationConfig'); if(moderationBox) moderationBox.style.display=state.testKind==='moderation'?'flex':'none';
  const quickToggle=$('#quick').closest('.mode-toggle'); quickToggle.style.display=state.testKind==='load'&&!fixedLoad&&!manualLoad?'flex':'none'; if(fixedLoad||manualLoad)$('#quick').checked=false;
  $('#customEnabled').disabled=manualKind||manualLoad;
  if(manualKind||manualLoad){ $('#customEnabled').checked=true; $('#customEnabled').closest('.custom-toggle').querySelector('span').textContent=isVideo?'视频任务参数':isImage?'图片压测参数':'随机压测参数'; box.querySelector('small').textContent=isVideo?'视频生成严格使用下方手动输入的任务数与并发数：每条任务提交一次生成，随后逐条轮询任务状态直到终态。':isImage?(state.testKind==='gpt-image'?'图片压测严格使用下方手动输入的总请求数和并发数；勾选「压测持续时长」后，总请求数退化为上限，时间到即停止发送新请求。':'图片压测严格使用下方手动输入的总请求数和并发数。'):'随机输入压测严格使用下方手动输入的总请求数和并发数。'; }
  else { $('#customEnabled').checked=state.testKind==='load'?$('#customEnabled').checked:false; $('#customEnabled').closest('.custom-toggle').querySelector('span').textContent='启用自定义性能参数'; box.querySelector('small').textContent='关闭时使用测试项原有默认配置；开启后应用下方参数'; }
  if(enabled && !$('#customRequests').dataset.edited){ $('#customRequests').value=item.requests||1; $('#customConcurrency').value=item.concurrency||1; }
  if(enabled){ $('#customRequests').disabled=!$('#customEnabled').checked; $('#customConcurrency').disabled=!$('#customEnabled').checked; }
  $('#imageHoldField').style.display=enabled&&state.testKind==='gpt-image'?'flex':'none';
  renderImageConfig();
  renderGeminiImageConfig();
  renderImagePromptConfig();
  renderSeedanceConfig();
  renderChatConfig();
  renderSignatureConfig();
  syncImageHoldState();
}

// —— 签名两步校验的输出预算 ——
// 只在选中 PROTOBUF_SIGNATURE_CHECK 时出现：预算给少了思考会把上限吃满、正文为零，
// 被契约读成「文本内容为空」⇒ 假 failed（2026-10-20 实测 sonnet-4-6 在 4096 下必然如此）。
// thinking 档位不在这里选：由后端 fable5/model_profile 按被测模型名解析。
const SIGNATURE_CASE = 'PROTOBUF_SIGNATURE_CHECK';
function renderSignatureConfig(){
  const box=$('#signatureConfig');
  if(!box) return;
  box.style.display=(state.testKind==='fable'&&state.selected.includes(SIGNATURE_CASE))?'flex':'none';
  const item=state.cases.find(x=>x.id===SIGNATURE_CASE);
  const field=$('#signatureMaxTokens');
  if(item&&field){
    if(item.signature_max_tokens_limit) field.max=String(item.signature_max_tokens_limit);
    if(!field.dataset.edited) field.value=String(item.signature_max_tokens_default||field.value||16000);
  }
}

// —— 对话窗口 ——
// 面板只把「用户消息 + 系统提示词 + 输出上限 + 温度」交给后端；执行走的是与其它套件
// 同一条队列与证据链（结果目录 / requests.jsonl / summary.json / response.md），
// 所以面板在提交后轮询记录状态，拿到终态再从 /response 取完整原文回显。
function renderChatConfig(){
  const box=$('#chatConfig');
  if(!box) return;
  box.style.display=state.testKind==='chat'?'block':'none';
  const item=state.cases.find(x=>x.test_kind==='chat');
  if(item){
    const field=$('#chatMaxTokens');
    if(field){
      if(item.chat_max_tokens_limit) field.max=String(item.chat_max_tokens_limit);
      if(!field.dataset.edited) field.value=String(item.chat_default_max_tokens||field.value||8192);
    }
  }
}
function setChatStatus(text, kind){
  const status=$('#chatStatus'); if(!status) return;
  status.textContent=text||'';
  status.className='chat-status'+(kind?` ${kind}`:'');
}
function renderChatResult(text, meta, raw){
  const box=$('#chatResult'); if(!box) return;
  state.chatRawText=text||'';
  state.chatRawView=Boolean(raw);
  box.className='chat-result show'+(raw?' raw':' markdown-body');
  if(raw) box.textContent=text||'（空响应）';
  else box.innerHTML=renderMarkdown(text||'（空响应）');
  if(meta) setChatStatus(meta, 'done');
}
async function pollChatRun(runId){
  const send=$('#chatSend');
  if(send) send.disabled=true;
  setChatStatus('已加入执行队列，等待结果…', 'running');
  try{
    for(let tick=0; tick<900; tick+=1){
      let run=null;
      try{ run=await api(`/api/runs/${encodeURIComponent(runId)}`); }
      catch(e){ setChatStatus(`读取记录失败：${e.message}`,'error'); return; }
      if(['queued','running'].includes(run.status)){
        setChatStatus(run.status==='queued'?'排队中…':`执行中… 已用 ${formatDuration(run)}`, 'running');
        await new Promise(resolve=>setTimeout(resolve,1000));
        continue;
      }
      await showChatResult(run);
      return;
    }
    setChatStatus('等待超时，请到报告页查看该条结果。','error');
  } finally {
    if(send) send.disabled=false;
  }
}
function chatUsageText(summary){
  const usage=summary&&summary.usage||{};
  const parts=[];
  if(summary&&summary.http_status!=null) parts.push(`HTTP ${summary.http_status}`);
  if(summary&&summary.finish_reason) parts.push(`finish ${summary.finish_reason}`);
  if(usage.prompt_tokens!=null||usage.completion_tokens!=null) parts.push(`token ${usage.prompt_tokens??'—'}/${usage.completion_tokens??'—'}`);
  if(summary&&typeof summary.response_chars==='number') parts.push(`正文 ${summary.response_chars} 字符`);
  if(summary&&summary.error_type) parts.push(`错误 ${summary.error_type}`);
  return parts.join(' · ');
}
async function showChatResult(run){
  let summary=null;
  if(run.summary_url){ try{ summary=await api(`/api/runs/${run.id}/summary`); }catch(e){ summary=null; } }
  const detail=(summary&&summary.error_message)?String(summary.error_message).slice(0,300):'';
  const meta=[`${run.model}`, statusLabel(run.status), run.passed===true?'通过':run.passed===false?'未通过':'待判定', formatDuration(run), chatUsageText(summary)].filter(Boolean).join(' · ');
  let text='';
  if(run.response_url){
    try{ text=await (await fetch(run.response_url)).text(); }
    catch(e){ text=''; }
  }
  if(!text && !run.response_url){
    // 没有正文文件（请求根本没发出去 / 未生成产物）：把原因直接摆在面板里。
    setChatStatus(detail?`${meta} · ${detail}`:meta, run.passed===false?'error':'done');
    const box=$('#chatResult');
    if(box){ box.className='chat-result show raw'; box.textContent=detail||'本次对话未生成响应正文，请查看该条记录的日志与请求明细。'; }
    return;
  }
  renderChatResult(text, detail?`${meta} · ${detail}`:meta, false);
}
function updateSelectionUI(){
  updateCustomDefaults();
  renderPelicanConfig();
  renderCaseDetails();
}
function selectCase(id) {
  const item=state.cases.find(x=>x.id===id);
  if(!item||!caseAvailability(item).supported)return;
  state.activeCase=id;
  const hasExclusive=state.cases.some(selected=>state.selected.includes(selected.id)&&TestCasePresets.isExclusive(state.testKind,selected));
  if(state.selected.includes(id))state.selected=state.selected.filter(x=>x!==id);
  else if(TestCasePresets.isExclusive(state.testKind,item)||hasExclusive) state.selected=[id];
  else state.selected=[...state.selected,id];
  renderCases();updateSelectionUI(); if(state.testKind==='gpt-image') renderImageConfig(); if(state.testKind==='gemini-image') renderGeminiImageConfig();
  persistForm();
}
function formatDuration(run) {
  if (!run.started_at) return '—'; const end=run.finished_at ? new Date(run.finished_at) : new Date();
  const sec=Math.max(0,Math.floor((end-new Date(run.started_at))/1000));
  return sec>=60 ? `${Math.floor(sec/60)}m ${sec%60}s` : `${sec}s`;
}
function statusLabel(status){return ({queued:'排队中',running:'执行中',completed:'已完成',failed:'失败',interrupted:'已中断'})[status]||status}
// 结果目录内可直接查看的产物（如鹈鹕骑车用例的 SVG）：按钮按文件类型给短标签，
// 完整说明放 title，点击在新标签页打开（html 直接看渲染，svg 看原图）。
function artifactLabel(name){
  if(name.endsWith('.html'))return '查看产物';
  if(name.endsWith('.svg'))return 'SVG 文件';
  if(name.endsWith('.md'))return '响应原文';
  return name;
}
function artifactLinks(run){
  const images=run.images_url?`<a class="link-btn artifact-btn" href="${escapeHtml(run.images_url)}" target="_blank" rel="noopener noreferrer" title="逐张查看生成的图片">查看图片</a>`:'';
  return images+(run.artifacts||[]).map(a=>`<a class="link-btn artifact-btn" href="${escapeHtml(a.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(a.title||a.name)}">${escapeHtml(artifactLabel(a.name))}</a>`).join('');
}
function scoreBadge(run){
  if(run.score===null||run.score===undefined)return '';
  const score=Number(run.score.toFixed(2));
  return `<span class="score-chip ${run.score_grade?'':'partial'}" title="模型侧 ${run.model_score??'未测'} / 100 · 通道侧 ${run.channel_score??'未测'} / 20"><i>总分</i>${score} / 120<em>${escapeHtml(run.score_grade||'证据不足')}</em></span>`;
}
function renderRuns() { renderWorkspaceRuns(); }
function updateDownloadUI(){
  const button=$('#downloadSelected'), count=state.downloadSelected.size;
  button.disabled=!count; button.textContent=`下载选中（${count}）`;
}
$('#selectAllDownloads').onclick=()=>{
  const downloadable=state.runs.filter(r=>r.download_url).map(r=>r.id);
  const all=downloadable.length && downloadable.every(id=>state.downloadSelected.has(id));
  downloadable.forEach(id=>all?state.downloadSelected.delete(id):state.downloadSelected.add(id));
  renderRuns();
};
$('#downloadSelected').onclick=async()=>{
  const ids=[...state.downloadSelected]; if(!ids.length)return;
  const button=$('#downloadSelected'); button.disabled=true; button.textContent='打包中…';
  try{
    const response=await fetch('/api/runs/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_ids:ids})});
    if(!response.ok){let data={};try{data=await response.json()}catch{};throw new Error(data.error||'打包下载失败')}
    const blob=await response.blob(), url=URL.createObjectURL(blob), link=document.createElement('a');
    link.href=url; link.download=`test-data-selected-${new Date().toISOString().slice(0,10)}.zip`; link.click(); URL.revokeObjectURL(url); toast(`已打包下载 ${ids.length} 条测试记录`);
  }catch(e){toast(e.message)}finally{updateDownloadUI()}
};

async function refreshRuns(silent=true) {
  const request=++workspace.listRequest;
  try {
    const query=new URLSearchParams({page:String(state.runsPage),page_size:String(state.runsPageSize),status:state.filter,conclusion:state.conclusion,q:state.runsSearch});
    const data=await api(`/api/runs?${query}`); if(request!==workspace.listRequest)return; state.runs=data.items; state.runsTotal=data.total;
    const lastPage=Math.max(1,Math.ceil(state.runsTotal/state.runsPageSize));
    if(state.runsPage>lastPage){state.runsPage=lastPage;return refreshRuns(silent)}
    renderRuns();
    if(workspace.page==='reports'&&workspace.report){
      const previous=workspace.report;
      let current=state.runs.find(r=>r.id===previous.id);
      if(!current&&['running','queued'].includes(previous.status)){
        try{current=await api(`/api/runs/${encodeURIComponent(previous.id)}`)}catch(e){return}
      }
      if(request!==workspace.listRequest||workspace.report?.id!==previous.id)return;
      if(current&&reportSignature(current)!==reportSignature(previous))openArtifact(current.id,workspace.view);
    }
  }
  catch(e) { if(!silent) toast(e.message); }
}


window.cancelRun = async id => {
  const run=state.runs.find(x=>x.id===id)||(workspace.report?.id===id?workspace.report:null); if(!run) return;
  const action=run.status==='queued'?'取消排队':'停止测试';
  if(!confirm(`确认${action}？已执行的请求会保留，并生成中断报告。`)) return;
  try { await api(`/api/runs/${id}/cancel`,{method:'POST',body:'{}'}); toast(`${action}请求已提交`); await refreshRuns(false); }
  catch(e) { toast(e.message); }
};

window.deleteRun = async id => {
  const run=state.runs.find(item=>item.id===id)||(workspace.report?.id===id?workspace.report:null); if(!run)return;
  const target=run.is_batch?'整个批次及其子记录和报告文件':'这条记录及其报告文件';
  if(!confirm(`确定删除${target}？此操作不可恢复。`))return;
  try{
    const response=await fetch(`/api/runs/${encodeURIComponent(id)}`,{method:'DELETE'});
    const data=await response.json(); if(!response.ok)throw new Error(data.error||'删除失败');
    state.downloadSelected.delete(id); if(workspace.report?.id===id)clearReport(); if(workspace.latest?.id===id){workspace.latest=null;$('#latestRun').hidden=true;} toast(`已删除 ${data.deleted_records} 条记录`); await refreshRuns(false);
  }catch(e){toast(e.message)}
};

window.copyRecordId = async id => {
  try { await navigator.clipboard.writeText(id); toast(`记录 ID 已复制：${id}`); }
  catch(e) { toast(`记录 ID：${id}`); }
};

const CASE_STATUS_META = { passed:['passed','通过'], failed:['failed','失败'], error:['failed','错误'], warning:['warning','警告'], blocked:['blocked','阻塞'], skipped:['blocked','跳过'], not_evaluable:['unknown','未评估'], interrupted:['warning','已中断'], completed:['passed','已完成'], running:['unknown','执行中'], queued:['unknown','排队中'] };
function caseBadge(status) {
  const meta = CASE_STATUS_META[status] || ['unknown', status || '未知'];
  return `<span class="case-badge ${meta[0]}">${escapeHtml(meta[1])}</span>`;
}
function fmtMetricSeconds(v) { return typeof v==='number' && isFinite(v) ? `${v.toFixed(3)}s` : '—'; }
function fmtMetricRate(v) { return typeof v==='number' && isFinite(v) ? `${(v*100).toFixed(1)}%` : '—'; }
function imageFingerprintDetails(images) {
  if (!images.length) return '<p class="detail-empty">暂无已保存的图片，无法读取图片指纹</p>';
  return `<div class="report-image-fingerprints">${images.map((image, index) => `
    <section class="report-image-fingerprint">
      <header><strong>图片 ${index + 1}</strong><span>${escapeHtml(image.name)}</span>
        <a href="${escapeHtml(image.url)}" target="_blank" rel="noopener noreferrer">查看原图</a></header>
      <dl>${(image.fields || []).map(field => `<div><dt>${escapeHtml(field.label)}</dt><dd>${escapeHtml(field.value)}</dd></div>`).join('')}</dl>
    </section>`).join('')}</div>`;
}
function casePointsHTML(points) {
  if (!points) return '<span class="case-points-muted">不计分</span>';
  if (points.superseded) return '<span class="case-points-muted" title="该用例已有后续复测，本条不重复计入报告总分">复测已替代</span>';
  if (points.veto_failed) return '<span class="case-points-loss">评分无效<small>信任校验失败</small></span>';
  if (!points.weight) return '<span class="case-points-muted">不计分</span>';
  const side = points.side === 'channel' ? '通道侧' : '模型侧';
  if (points.earned == null) return `<span class="case-points-muted">未计分<small>${side} · 无有效计分结果</small></span>`;
  const fmt = value => value > 0 && value < 0.01 ? value.toFixed(4) : value.toFixed(2);
  const earned = points.earned > 0 ? `<span class="case-points-gain">+${fmt(points.earned)}</span>` : '';
  const lost = points.lost > 0 ? `<span class="case-points-loss">−${fmt(points.lost)}</span>` : '';
  return `<span class="case-points" title="该用例满分 ${fmt(points.weight)}；得分 ${fmt(points.earned)}；失分 ${fmt(points.lost)}">${earned}${earned && lost ? ' / ' : ''}${lost}<small>${side}</small></span>`;
}
function reportRoundDetails(details) {
  if (!details) return '';
  const sections = (details.sections || []).map(section => {
    const columns = section.columns || [];
    return `<section class="round-detail-section"><h3>${escapeHtml(section.title)}</h3>
      <div class="round-detail-scroll"><table class="round-detail-table"><thead><tr>${columns.map(col => `<th data-col="${escapeHtml(col.key)}">${escapeHtml(col.label)}</th>`).join('')}</tr></thead>
      <tbody>${(section.rows || []).map(row => `<tr>${columns.map(col => `<td data-col="${escapeHtml(col.key)}">${col.key === 'status' ? caseBadge(row[col.key]) : col.key === 'points' ? casePointsHTML(row[col.key]) : escapeHtml(row[col.key] ?? '—')}</td>`).join('')}</tr>`).join('')}</tbody></table></div></section>`;
  }).join('');
  const notes = (details.notes || []).map(note => `<p class="round-detail-note">${escapeHtml(note)}</p>`).join('');
  const links = (details.links || []).filter(link => /^\/api\/runs\/[^/]+\/(requests|summary|log|download)$/.test(link.url || ''))
    .map(link => `<a href="${escapeHtml(link.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(link.label)}</a>`).join('');
  return sections + notes + (links ? `<nav class="round-detail-links" aria-label="执行证据">${links}</nav>` : '');
}
function batchSuiteTable(manifest) {
  const items = manifest && Array.isArray(manifest.results) ? manifest.results : [];
  if (!items.length) return '';
  // 多模型批次（同一用例 × 多个模型，如签名校验的执行范围多选）：
  // 首列行标签与表头改成按模型展示；普通批次维持按用例/轮次展示。
  const multiModel = Boolean(manifest.model) && items.some(item=>item.model && item.model !== manifest.model);
  const headLabel = multiModel ? '模型' : '轮次';
  const rows = items.map((item) => {
    const m = item.metrics || {}, s = item.summary || {};
    const cases = Array.isArray(s.results) ? s.results : [];
    const hasFailures=item.passed===false||cases.some(row=>['failed','error'].includes(row.case_status||row.status)||row.passed===false);
    const rowLabel = (multiModel && item.model) ? item.model : item.case;
    const caseTotal = m.case_total != null ? m.case_total : (cases.length || null);
    const casePassed = m.case_passed != null ? m.case_passed : cases.filter(r=>r.status==='passed').length;
    const extras = (m.case_warning || m.case_blocked) ? ` <small class="case-extras">⚠${m.case_warning||0} · ⏸${m.case_blocked||0}</small>` : '';
    const caseText = caseTotal ? `${casePassed}/${caseTotal}${extras}` : '—';
    const detailRows = cases.length ? cases.map(r => `<tr>
        <td data-col="id"><code>${escapeHtml(r.case_id || r.id || '—')}</code></td>
        <td data-col="points">${casePointsHTML(null)}</td>
        <td data-col="name">${escapeHtml(r.name || r.title || '')}</td>
        <td data-col="status">${caseBadge(r.case_status || r.status || (r.passed===true?'passed':r.passed===false?'failed':'not_evaluable'))}</td>
        <td data-col="duration" class="num">${typeof r.duration_seconds==='number' && r.duration_seconds>=0 ? r.duration_seconds.toFixed(2) : '—'}</td>
        <td data-col="detail" class="case-detail-text">${escapeHtml(String(r.details || r.detail || r.error || '—').slice(0, 240))}</td>
      </tr>`).join('') : `<tr><td colspan="6" class="detail-empty">${escapeHtml(item.error || s.error || '暂无已保存的执行明细')}</td></tr>`;
    const imageContent = Array.isArray(item.image_fingerprints) ? imageFingerprintDetails(item.image_fingerprints) : '';
    const roundContent = reportRoundDetails(item.report_details);
    const detailContent = imageContent + roundContent || `<div class="round-detail-scroll"><table class="round-detail-table"><thead><tr><th>用例</th><th>分值</th><th>名称</th><th>状态</th><th>耗时(s)</th><th>结论说明</th></tr></thead><tbody>${detailRows}</tbody></table></div>`;
    return `<tr class="suite-row" aria-expanded="false" title="子记录 ${escapeHtml(item.record_id || '')}">
      <td><span class="chev">▸</span> <b>${escapeHtml(rowLabel)}</b></td>
      <td><span class="badge ${escapeHtml(item.status || '')}">${escapeHtml(statusLabel(item.status || ''))}</span></td>
      <td><b class="${hasFailures?'fail':item.passed===true?'pass':'pending'}">${hasFailures?'有失败项':item.passed===true?'无失败项':'待完成'}</b></td>
      <td class="num">${typeof m.requests==='number' ? m.requests : '—'}</td>
      <td class="num">${fmtMetricRate(m.success_rate)}</td>
      <td class="num">${caseText}</td>
      <td class="num">${fmtMetricSeconds(m.ttft_p90_seconds)}</td>
      <td class="num">${fmtMetricSeconds(m.ttfc_p90_seconds)}</td>
      <td class="num">${fmtMetricSeconds(m.e2e_p90_seconds)}</td>
      <td class="num">${typeof m.tpot_p90_ms_per_token==='number' ? m.tpot_p90_ms_per_token.toFixed(1) : '—'}</td>
    </tr>
    <tr class="suite-detail" hidden><td colspan="10"><div class="case-detail"><p class="case-record">子记录 <code>${escapeHtml(item.record_id || '')}</code></p>${detailContent}</div></td></tr>`;
  }).join('');
  const imageClass = items.some(item => Array.isArray(item.image_fingerprints) || item.report_details) ? ' has-image-details' : '';
  return `<div class="batch-suites md-table-wrap${imageClass}"><table><thead><tr>
    <th>${headLabel}</th><th>状态</th><th>结论</th><th>请求</th><th>成功率</th><th>用例通过</th><th>TTFT P90</th><th>TTFC P90</th><th>E2E P90</th><th>TPOT(ms/tok)</th>
  </tr></thead><tbody>${rows}</tbody></table>
  <p class="batch-suites-hint">点击轮次展开 / 收起对应的用例、请求、任务结果或图片指纹。请求数与成功率来自执行证据（requests.jsonl）；TTFT / TTFC P90 仅统计流式请求样本，E2E P90 覆盖全部请求。</p></div>`;
}
function renderBatchReport(text, manifest) {
  // 把报告按 "## " 标题分节；"各轮结果" 一节用可展开/收起的交互表格替代静态 markdown 表格。
  const lines = String(text).replace(/\r/g,'').split('\n');
  const sections = []; let current = { heading: null, lines: [] };
  for (const line of lines) {
    const h = line.match(/^##\s+(.+)$/);
    if (h) { sections.push(current); current = { heading: h[1].trim(), lines: [] }; }
    else current.lines.push(line);
  }
  sections.push(current);
  return sections.map(sec => {
    const body = sec.lines.join('\n');
    // 「各轮结果」（fable5 批次）与「子用例结果」（dsv4 等专项批次）都是套件级清单：
    // 统一替换成可展开/收起的分组套件表，避免静态表把所有用例平铺在一起。
    if ((sec.heading === '各轮结果' || sec.heading === '子用例结果') && manifest) {
      const table = batchSuiteTable(manifest);
      if (table) return table;
    }
    // 用例级结果已在交互表中展示，跳过静态的「用例结果」章节避免重复。
    if (sec.heading === '用例结果' && manifest && batchSuiteTable(manifest)) return '';
    const content = renderMarkdown(body);
    return sec.heading ? `<h2>${inlineMarkdown(sec.heading)}</h2>${content}` : content;
  }).join('');
}

function renderSingleReport(text, manifest, run) {
  // 单项报告在收尾阶段已改写成聚合版式（头部元信息 + 统一指标表 + 原有章节）。
  // 这里把 `## 指标` 的静态表格替换成可展开/收起的交互表；旧报告没有该章节，
  // 则按旧逻辑在头部要点之后补插表格，保证历史记录同样有指标视图。
  const lines = String(text).replace(/\r/g,'').split('\n');
  const sections = []; let current = { heading: null, lines: [] };
  for (const line of lines) {
    const h = line.match(/^##\s+(.+)$/);
    if (h) { sections.push(current); current = { heading: h[1].trim(), lines: [] }; }
    else current.lines.push(line);
  }
  sections.push(current);
  const hasMeta = /^- 记录 ID：/m.test(String(text));
  const metaLines = hasMeta ? [] : [`- 记录 ID：\`${run?.id||'—'}\``, `- 协议：\`${run?.protocol||'—'}\``, `- 模型：\`${run?.model||'—'}\``];
  const table = manifest ? batchSuiteTable(manifest) : '';
  const hasMetrics = sections.some(sec=>sec.heading==='指标');
  let html = '', placed = false;
  sections.forEach(sec => {
    if (sec.heading === null) {
      const index = sec.lines.findIndex(line=>line.startsWith('# '));
      const titleLines = index>=0 ? sec.lines.slice(0,index+1) : [];
      const rest = index>=0 ? sec.lines.slice(index+1) : sec.lines;
      const merged = [...titleLines, '', ...metaLines, ...(metaLines.length?['']:[]), ...rest];
      html += renderMarkdown(merged.join('\n'));
      if (table && !placed && !hasMetrics) { html += table; placed = true; }
      return;
    }
    // 「指标」与「子用例结果」的静态表都交给分组套件表，避免用例平铺。
    if ((sec.heading === '指标' || sec.heading === '子用例结果') && table) { html += table; placed = true; return; }
    // 用例级结果已在交互表中展示，跳过静态的「用例结果」章节避免重复。
    if (sec.heading === '用例结果' && table) return;
    html += `<h2>${inlineMarkdown(sec.heading)}</h2>${renderMarkdown(sec.lines.join('\n'))}`;
  });
  if (table && !placed) html += table;
  return html;
}

function renderScore(rep){
  if(rep?.scores)return rep.scores.map(item=>`<section class="model-score-report"><h2>${escapeHtml(item.model||'模型评分')}</h2>${renderScore(item)}</section>`).join('');
  if(!rep||rep.error)return `<p class="data-note">${escapeHtml(rep?.error||'暂无评分')}</p>`;
  const number=value=>typeof value==='number'?Number(value.toFixed(2)).toString():'—';
  const components=rep.components||{};
  const sideName=side=>side==='channel'?'通道侧':'模型侧';
  const componentText=side=>components[side]?.status==='not_selected'?'本次未测':components[side]?.score==null?'暂无有效证据':`${Math.round((components[side].coverage||0)*100)}% 覆盖`;
  const cards=[[`${number(rep.model_score)} / 100`,'模型侧',componentText('model')],
    [`${number(rep.channel_score)} / 20`,'通道侧',componentText('channel')],
    [`${number(rep.score)} / 120`,'总分',rep.valid===false?'评分无效':rep.partial?'证据覆盖不足':'按两侧得分相加'],
    [rep.grade||'待评分','评分等级',rep.rating_passed===true?'评分通过':rep.rating_passed===false?'评分未达可用标准':'尚不形成评级']];
  const stateText={full:'满分',gap:'有失分',unevaluated:'无证据',untested:'未覆盖',not_selected:'本次未测',veto_passed:'已通过',veto_failed:'有失败项 · 评分无效',evidence:'仅证据',performance:'不计分'};
  const faces=(rep.faces||[]).filter(f=>f.weight||f.status==='veto_passed'||f.status==='veto_failed');
  return `<div class="score-cards">${cards.map(([value,label,detail])=>`<div><span>${label}</span><b>${escapeHtml(value)}</b><small>${escapeHtml(detail)}</small></div>`).join('')}</div>
    <p class="data-note">60–不足 70：可用 · 70–不足 80：一般 · 80–不足 90：良好 · 90–不足 100：优秀 · 100–110：十分优秀 · 超过 110：官方；满分 120。这里的「官方」为评分等级。</p>
    ${rep.valid===false?`<p class="fail">${escapeHtml(rep.invalid_reason||'评分无效')}</p>`:''}
    <p class="report-note">模型侧满分 100，通道侧满分 20。通道失分只扣通道侧；未测项不计得分。通过与否按评分判断，失败用例仍保留原始证据。</p>
    ${(rep.notes||[]).map(note=>`<p class="report-note">${escapeHtml(note)}</p>`).join('')}
    <div class="section-title"><h3>分侧计分明细</h3></div><div class="md-table-wrap"><table><thead><tr><th>侧别</th><th>计分面</th><th>满分</th><th>得分</th><th>状态</th></tr></thead><tbody>${faces.map(face=>`<tr><td>${sideName(face.side)}</td><td>${escapeHtml(face.label)}</td><td>${number(face.weight)}</td><td>${number(face.earned)}</td><td>${escapeHtml(stateText[face.status]||face.status)}</td></tr>`).join('')}</tbody></table></div>
    ${(rep.deductions||[]).length?`<div class="section-title"><h3>失分项</h3></div><div class="md-table-wrap"><table><thead><tr><th>用例</th><th>侧别</th><th>计分面</th><th>状态</th><th>失分</th></tr></thead><tbody>${rep.deductions.map(d=>`<tr><td>${escapeHtml(d.case_id)} ${escapeHtml(d.name||'')}</td><td>${sideName(d.side||d.attribution)}</td><td>${escapeHtml(d.face)}</td><td>${escapeHtml(d.status)}</td><td>${number(d.points)}</td></tr>`).join('')}</tbody></table></div>`:''}
    <p class="report-note">口径 ${escapeHtml(rep.scoring_version||'')}；passed 满权、warning 半权，未测与无法评估的项目保留为证据缺口；身份一致性失败不形成有效评分。</p>`;
}
function scoreSummaryHTML(run){
  const number=value=>typeof value==='number'?Number(value.toFixed(2)).toString():'—';
  const channel=run.score_components?.channel;
  return `<div class="score-cards"><div><span>模型侧 / 100</span><b>${number(run.model_score)}</b></div><div><span>通道侧 / 20</span><b>${number(run.channel_score)}</b><small>${channel?.status==='not_selected'?'本次未测':''}</small></div><div><span>总分 / 120</span><b>${number(run.score)}</b></div><div><span>评分等级</span><b>${escapeHtml(run.score_grade||(run.score_models?'按模型分别评分':'待评分'))}</b><small>${run.rating_passed===true?'评分通过':run.rating_passed===false?'评分未达可用标准':run.score_partial?'证据覆盖不足':''}</small></div></div>`;
}
// Two-page presentation. Test execution and saved payloads remain in the original form handler.
const workspace = { page:'test', report:null, view:'report', request:0, controller:null, listRequest:0, listMarkup:null, scroll:{test:0,reports:0}, latest:null };
const REPORT_VIEWS = [['report','报告'],['requests','请求明细'],['artifacts','产物'],['log','日志']];
const DATA_VIEWS = [['score','一致性评分'],['summary','指标 JSON'],['manifest','聚合数据'],['response','响应原文'],['full','完整报告']];
function setTheme(theme, save=false){
  const value=theme==='dark'?'dark':'light';
  document.documentElement.dataset.theme=value;
  const button=$('#themeToggle');
  button.textContent=value==='dark'?'☀ 浅色':'☾ 深色';
  button.setAttribute('aria-label',value==='dark'?'切换到浅色主题':'切换到深色主题');
  button.title=`当前为${value==='dark'?'深色':'浅色'}主题`;
  if(save){try{localStorage.setItem('ml-test-platform.theme.v1',value)}catch(e){toast('主题已切换，浏览器未允许保存偏好')}}
}
function showPage(page, updateHash=true){
  const nextPage=page==='reports'?'reports':'test';
  const changed=nextPage!==workspace.page;
  if(changed)workspace.scroll[workspace.page]=window.scrollY;
  workspace.page=nextPage;
  for(const name of ['test','reports']){
    $('#'+name+'Page').hidden=workspace.page!==name;
    $('#'+name+'Nav').classList.toggle('active',workspace.page===name);
    $('#'+name+'Nav').setAttribute('aria-current',workspace.page===name?'page':'false');
  }
  if(changed)window.scrollTo({top:workspace.scroll[workspace.page],behavior:'instant'});
  if(updateHash&&location.hash!=='#'+workspace.page)location.hash=workspace.page;
}
function fitTestWorkspaceToViewport(){
  const panel=document.querySelector('#testPage .tests');
  if(!panel)return;
  let pending=false;
  const update=()=>{
    pending=false;
    if($('#testPage').hidden)return;
    const top=panel.getBoundingClientRect().top;
    const headerBottom=document.querySelector('.topbar').getBoundingClientRect().bottom;
    const viewport=window.visualViewport;
    const bottom=viewport?viewport.offsetTop+viewport.height:window.innerHeight;
    // Keep the full-size fallback while below the fold; fit the remaining space once visible.
    if(top>=bottom){panel.style.removeProperty('height');panel.style.removeProperty('margin-bottom');return;}
    const height=Math.max(1,Math.floor(bottom-Math.max(top,headerBottom+12)-12));
    if(panel.style.height!==`${height}px`)panel.style.height=`${height}px`;
    // Preserve enough page travel to move the configuration above the viewport.
    // As the workspace expands, this trailing space shrinks by the same amount.
    const trailing=Math.max(0,Math.ceil(top-headerBottom-12));
    if(panel.style.marginBottom!==`${trailing}px`)panel.style.marginBottom=`${trailing}px`;
  };
  const schedule=()=>{if(!pending){pending=true;requestAnimationFrame(update);}};
  window.addEventListener('resize',schedule);
  window.addEventListener('scroll',schedule,{passive:true});
  window.visualViewport?.addEventListener('resize',schedule);
  window.visualViewport?.addEventListener('scroll',schedule);
  const observer=new ResizeObserver(schedule);
  for(const element of [panel,document.querySelector('.topbar'),document.querySelector('#testPage .connection')])if(element)observer.observe(element);
  schedule();
}
function hasFailureItems(run){
  const checks=run.summary?.results||[];
  return run.has_failure_items===true||run.passed===false||checks.some(row=>['failed','error'].includes(row.case_status||row.status)||row.passed===false);
}
function runVerdict(run){
  // Raw item outcomes remain separate from score-based acceptance.
  return hasFailureItems(run)?'有失败项':run.passed===true?'无失败项':'待完成';
}
function renderWorkspaceRuns(){
  const markup=state.runs.length?state.runs.map(run=>`<tr class="${run.id===workspace.report?.id?'active':''}"><td>
    <div class="record-main"><input type="checkbox" class="download-check" data-run-id="${escapeHtml(run.id)}" ${state.downloadSelected.has(run.id)?'checked':''} ${run.download_url?'':'disabled'} aria-label="选择下载 ${escapeHtml(run.case_name)}">
    <button class="record-button" data-report-id="${escapeHtml(run.id)}" ${run.id===workspace.report?.id?'aria-current="true"':''}><strong>${escapeHtml(kindLabel(run.test_kind))} · ${escapeHtml(run.is_batch?'批量测试':run.case_title||run.case_name)}${run.quick?' · 快速':''}</strong><small>${escapeHtml(run.model)}</small><small>${escapeHtml(run.id)}</small><small>${escapeHtml(new Date(run.created_at).toLocaleString('zh-CN'))}</small></button></div>
    <div class="record-meta"><span class="badge ${escapeHtml(run.status)}">${escapeHtml(statusLabel(run.status))}</span><span class="${hasFailureItems(run)?'fail':run.passed===true?'pass':'pending'}">${runVerdict(run)}</span><span class="time" data-start="${escapeHtml(run.id)}">${formatDuration(run)}</span></div>${scoreBadge(run)}</td></tr>`).join(''):'<tr><td class="empty">暂无匹配的测试记录</td></tr>';
  if(workspace.listMarkup!==markup){$('#runRows').innerHTML=markup;workspace.listMarkup=markup;}
  document.querySelectorAll('.download-check').forEach(el=>el.onchange=()=>{if(el.checked)state.downloadSelected.add(el.dataset.runId);else state.downloadSelected.delete(el.dataset.runId);updateDownloadUI()});
  document.querySelectorAll('[data-report-id]').forEach(el=>el.onclick=()=>openArtifact(el.dataset.reportId,'report'));
  const pages=Math.max(1,Math.ceil(state.runsTotal/state.runsPageSize));
  $('#runsPageInfo').textContent=`${state.runsTotal} 条 · ${state.runsPage}/${pages}`;
  $('#runsPrev').disabled=state.runsPage<=1;$('#runsNext').disabled=state.runsPage>=pages;
  updateDownloadUI();
  if(workspace.latest){const run=state.runs.find(r=>r.id===workspace.latest.id);if(run)showLatestRun(run,false)}
}
function showLatestRun(run, scroll=true){
  workspace.latest=run;
  const box=$('#latestRun');box.hidden=false;
  box.innerHTML=`<span><b>${escapeHtml(statusLabel(run.status))}</b> · ${escapeHtml(run.case_title||run.case_name||run.id)} · ${runVerdict(run)}</span><button class="text-btn" id="latestReportLink">查看报告 →</button>`;
  box.querySelector('#latestReportLink').onclick=()=>openArtifact(run.id,'report');
  if(scroll)box.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function reportSignature(run){
  return JSON.stringify([run.status,run.passed,run.score,run.score_grade,run.model_score,run.channel_score,run.finished_at,run.report_url,run.manifest_url,run.summary_url,run.requests_url,run.response_url,run.score_url,run.images_url,run.artifacts]);
}
function clearReport(){
  workspace.request++;workspace.controller?.abort();workspace.report=null;
  $('#dialogTitle').textContent='选择一条测试记录';$('#dialogEyebrow').textContent='测试报告';
  for(const id of ['reportMeta','reportTabs','reportActions'])$('#'+id).replaceChildren();
  $('#reportBody').className='report-body';$('#reportBody').removeAttribute('aria-busy');
  $('#reportBody').innerHTML='<div class="empty">在左侧选择一条记录，查看报告与原始证据。</div>';
}
function renderReportHeader(run,view){
  $('#dialogEyebrow').textContent='测试报告 / '+kindLabel(run.test_kind);
  $('#dialogTitle').textContent=`${run.model} · ${run.is_batch?'批量测试':run.case_title||run.case_name}`;
  $('#reportMeta').innerHTML=`<div class="report-meta"><span>记录 <code>${escapeHtml(run.id)}</code></span><span>${escapeHtml(run.protocol)}</span><span>${escapeHtml(run.base_url)}${run.path_override?' · '+escapeHtml(run.path_override):''}</span><span>${escapeHtml(new Date(run.created_at).toLocaleString('zh-CN'))}</span><span class="badge ${escapeHtml(run.status)}">${escapeHtml(statusLabel(run.status))}</span><span>耗时 ${formatDuration(run)}</span>${scoreBadge(run)}</div>`;
  $('#reportActions').innerHTML=`<div class="report-actions">${run.download_url?`<a class="button" href="${escapeHtml(run.download_url)}" download>↓ 下载全部</a>`:''}<details class="report-more"><summary aria-label="更多记录操作">···</summary><div class="report-more-menu"><button data-record-action="copy">复制记录 ID</button><button data-record-action="refresh">刷新报告</button>${['queued','running'].includes(run.status)?`<button class="danger" data-record-action="cancel">${run.status==='queued'?'取消':'停止'}测试</button>`:'<button class="danger" data-record-action="delete">删除记录</button>'}</div></details></div>`;
  $('#reportTabs').innerHTML=[...REPORT_VIEWS,...DATA_VIEWS].map(([key,label])=>`<button data-report-view="${key}" class="${view===key?'active':''}" aria-current="${view===key?'page':'false'}">${label}</button>`).join('');
  $('#reportTabs').querySelectorAll('[data-report-view]').forEach(b=>b.onclick=()=>openArtifact(run.id,b.dataset.reportView));
  $('#reportActions').querySelectorAll('[data-record-action]').forEach(b=>b.onclick=()=>{
    const action=b.dataset.recordAction;
    if(action==='copy')copyRecordId(run.id);
    if(action==='refresh')openArtifact(run.id,workspace.view);
    if(action==='cancel')cancelRun(run.id);
    if(action==='delete')deleteRun(run.id);
  });
}
function overviewSuiteTable(manifest,item){
  // The outer disclosure opens directly to evidence; the original report keeps its own row controls.
  return batchSuiteTable({...manifest,results:[item]})
    .replace('class="suite-row" aria-expanded="false"','class="suite-row open" aria-expanded="true"')
    .replace('class="suite-detail" hidden','class="suite-detail"');
}
function reportOverview(run,manifest,fullHTML,notes=[],capabilityScore=null){
  const items=Array.isArray(manifest?.results)?manifest.results:[];
  const ranked=items.map((item,index)=>({item,index})).sort((a,b)=>(hasFailureItems(a.item)?0:a.item.passed===true?2:1)-(hasFailureItems(b.item)?0:b.item.passed===true?2:1));
  const passed=items.filter(item=>item.passed===true&&!hasFailureItems(item)).length;
  const requestCounts=items.map(item=>item.metrics?.requests);
  const totalRequests=items.length&&requestCounts.every(n=>typeof n==='number'&&Number.isFinite(n))?requestCounts.reduce((a,b)=>a+b,0):'—';
  // A batch has no aggregate percentile/rate in its manifest. Show each round's original values.
  const rate=items.length===1?fmtMetricRate(items[0].metrics?.success_rate):items.length?'分轮展示':'—';
  const failureItems=hasFailureItems(run)||items.some(hasFailureItems);
  const verdict=failureItems?'有失败项':run.passed===true?'无失败项':'测试尚未完成';
  const hint='失败项仅描述测试证据，通过与否按评分判断。模型侧满分 100，通道侧满分 20，总分 120。'+(run.quick?'本次为快速验证。':'');
  return `<div class="verdict-box ${failureItems?'fail-box':run.passed===true?'pass-box':'pending-box'}"><div class="verdict-symbol">${failureItems?'!':run.passed===true?'✓':'·'}</div><div><h3>${verdict}</h3><p>${hint}</p></div></div>
    ${scoreSummaryHTML(run)}
    ${renderCapabilityRadar(capabilityScore,manifest)}
    <div class="report-metrics"><div><span>无失败项测试</span><b>${items.length?passed:'—'}<small>${items.length?' / '+items.length:''}</small></b></div><div><span>正式请求数</span><b>${totalRequests}</b></div><div><span>请求成功率</span><b class="metric-rate">${rate}</b></div><div><span>执行耗时</span><b>${formatDuration(run)}</b></div></div>
    ${notes.map(note=>`<p class="data-note">${escapeHtml(note)}</p>`).join('')}
    ${items.length?`<div class="section-title"><h3>测试结果</h3><small>有失败项的测试优先 · 展开查看证据</small></div><div class="result-checks">${ranked.map(({item,index})=>`<details class="result-item"><summary><span><b>${escapeHtml(item.model&&item.model!==run.model?item.model+' · ':'')}${escapeHtml(caseLabel(item.case)||item.record_id||'测试 '+(index+1))}</b><small>${escapeHtml(statusLabel(item.status||''))}</small></span><span class="${hasFailureItems(item)?'fail':item.passed===true?'pass':'pending'}">${runVerdict(item)}</span></summary><div class="result-evidence">${overviewSuiteTable(manifest,item)}</div></details>`).join('')}</div>`:''}
    <details class="original-report" ${!items.length?'open':''}><summary>完整原始报告 · 全部指标、批次和逐轮数据</summary><div class="markdown-body">${fullHTML}</div></details>`;
}
function reportArtifacts(run){
  const files=[...REPORT_VIEWS,...DATA_VIEWS].filter(([key])=>!['artifacts','full'].includes(key)&&run[key+'_url']);
  const media=artifactLinks(run);
  if(!files.length&&!media&&!run.download_url)return '<div class="empty">尚未生成产物。任务完成后可在此查看和下载。</div>';
  return `<p class="data-note">报告、请求、日志和生成文件按原格式保留，下载全部可获取完整归档。</p>${media?`<div class="artifact-media">${media}</div>`:''}<div class="artifact-list">${files.map(([key,label])=>`<div class="artifact-file"><strong>${label}</strong><div><button data-artifact-view="${key}">查看</button> <a class="button" href="${escapeHtml(run[key+'_url'])}" download>下载 ↓</a></div></div>`).join('')}</div>`;
}
async function readReportFile(run,kind,signal){
  const response=await fetch(`/api/runs/${encodeURIComponent(run.id)}/${kind}`,{signal});
  if(!response.ok)throw new Error(response.status===404?'文件尚未生成或已不存在':`读取失败（HTTP ${response.status}）`);
  return response.text();
}
window.openArtifact = async (id, kind='report') => {
  if(![...REPORT_VIEWS,...DATA_VIEWS].some(([key])=>key===kind))return;
  const changedRecord=workspace.report?.id!==id||workspace.page!=='reports';
  const request=++workspace.request;
  workspace.controller?.abort();workspace.controller=new AbortController();
  const signal=workspace.controller.signal;
  workspace.view=kind;showPage('reports');
  if(changedRecord)document.querySelector('.report-panel').scrollIntoView({block:'start',behavior:'instant'});
  const body=$('#reportBody');body.className='report-body';body.setAttribute('aria-busy','true');
  body.innerHTML='<div class="empty" role="status">正在读取报告…</div>';
  // Clear the previous record before fetching, so late responses and failed reads cannot expose stale actions.
  workspace.report=null;$('#dialogEyebrow').textContent='测试报告';$('#dialogTitle').textContent='正在读取记录…';
  for(const key of ['reportMeta','reportActions','reportTabs'])$('#'+key).replaceChildren();
  try{
    const response=await fetch(`/api/runs/${encodeURIComponent(id)}`,{signal});
    if(!response.ok)throw new Error(response.status===404?'记录已不存在':`记录读取失败（HTTP ${response.status}）`);
    const run=await response.json();if(request!==workspace.request)return;
    workspace.report=run;renderReportHeader(run,kind);renderRuns();
    if(kind==='artifacts'){
      body.innerHTML=reportArtifacts(run);
      body.querySelectorAll('[data-artifact-view]').forEach(b=>b.onclick=()=>openArtifact(id,b.dataset.artifactView));
    }else if(kind==='report'||kind==='full'){
      const results=await Promise.allSettled([readReportFile(run,'report',signal),readReportFile(run,'manifest',signal).then(JSON.parse),kind==='report'?readReportFile(run,'score',signal).then(JSON.parse):Promise.resolve(null)]);
      if(request!==workspace.request)return;
      const [reportResult,manifestResult,capabilityResult]=results;
      const manifest=manifestResult.status==='fulfilled'?manifestResult.value:null;
      const text=reportResult.status==='fulfilled'?reportResult.value.replace(/^(\s*-?\s*)总结论[：:]\s*(未通过|通过)\s*$/gm,(_,prefix,value)=>prefix+'测试项结果：'+(value==='未通过'?'有失败项':'无失败项')):'';
      const validManifest=manifest?.record_type==='batch'?manifest:null;
      const fullHTML=reportResult.status==='fulfilled'?(run.is_batch?renderBatchReport(text,validManifest):renderSingleReport(text,validManifest,run)):'<p class="empty">报告文件尚不可用，请查看日志或稍后刷新。</p>';
      const notes=[];
      if(reportResult.status==='rejected')notes.push('报告：'+reportResult.reason.message);
      if(manifestResult.status==='rejected')notes.push('指标与逐项证据暂不可用：'+manifestResult.reason.message+'。可查看完整原始报告或日志。');
      body.className=kind==='full'?'report-body markdown-body':'report-body';
      body.innerHTML=kind==='full'?fullHTML:reportOverview(run,validManifest,fullHTML,notes,capabilityResult.status==='fulfilled'?capabilityResult.value:null);
      bindCapabilityRadars(body);
    }else{
      const text=await readReportFile(run,kind,signal);if(request!==workspace.request)return;
      if(kind==='score'){body.className='report-body markdown-body';body.innerHTML=renderScore(JSON.parse(text))}
      else{$('#reportBody').className='report-body code-body';$('#reportBody').textContent=text}
    }
    // The existing expandable report rows also work from the keyboard.
    body.querySelectorAll('.suite-row').forEach(row=>{row.tabIndex=0;row.setAttribute('role','button')});
  }catch(e){
    if(request!==workspace.request||e.name==='AbortError')return;
    body.className='report-body';body.innerHTML=`<div class="empty"><p>${escapeHtml(e.message)}</p><button id="retryReport">重试</button></div>`;
    body.querySelector('#retryReport').onclick=()=>openArtifact(id,kind);
  }finally{
    if(request===workspace.request){
      body.removeAttribute('aria-busy');
      if(changedRecord&&workspace.page==='reports')document.querySelector('.report-panel').scrollIntoView({block:'start',behavior:'instant'});
    }
  }
};

$('#runForm').addEventListener('submit', async event => {
  event.preventDefault(); const message=$('#formMessage'); message.textContent='';
  if(!state.selected.length){message.textContent='请先选择一个测试场景';return}
  const quick=$('#quick').checked;
  const selectedItem=state.cases.find(x=>x.id===state.selected[0]);
  const effectiveTestKind=state.testKind==='open-source'?(selectedItem?.test_kind||''):state.testKind;
  const isImage=effectiveTestKind==='gpt-image';
  const isGeminiImage=effectiveTestKind==='gemini-image';
  const isVideo=effectiveTestKind==='seedance';
  const isChat=effectiveTestKind==='chat';
  if(effectiveTestKind==='load' && !quick && state.selected.some(id=>['C1','SIMPLE_LOAD','FULL_LOAD'].includes(id)) && !confirm(`${state.selected.join('、')} 包含高成本测试，会产生较高 Token 成本，确认立即排队执行？`)) return;
  if(isChat && !$('#chatPrompt').value.trim()){message.textContent='请先填写要发送的用户消息';return}
  if(isVideo){
    // 视频单条成本极高：明显不合法的输入在提交前先拦一次，避免把废任务排进队列。
    // 取值域来自 /api/cases 下发的白名单，服务端仍会按同一份白名单二次校验。
    const video=seedanceCase();
    if(!$('#seedancePrompt').value.trim()){message.textContent='请先填写视频提示词';$('#seedancePrompt').focus();return}
    const duration=$('#seedanceDuration'), seconds=Number(duration.value);
    if(!Number.isInteger(seconds)||seconds<Number(duration.min)||seconds>Number(duration.max)){message.textContent=`时长必须填写 ${duration.min}～${duration.max} 的整数秒`;return}
    const maxTasks=video&&video.seedance_max_requests?video.seedance_max_requests:100;
    const maxConcurrency=video&&video.seedance_max_concurrency?video.seedance_max_concurrency:16;
    const tasks=Number($('#customRequests').value), concurrent=Number($('#customConcurrency').value);
    if(!Number.isInteger(tasks)||tasks<1||tasks>maxTasks){message.textContent=`视频任务数必须是 1～${maxTasks} 的整数`;return}
    if(!Number.isInteger(concurrent)||concurrent<1||concurrent>maxConcurrency){message.textContent=`并发数必须是 1～${maxConcurrency} 的整数`;return}
    if(state.seedanceUploading){message.textContent='图片还在上传中，请稍候再提交';return}
    if(state.seedanceVideoUploading){message.textContent='参考视频还在上传中，请稍候再提交';return}
    // 生成方式决定哪些输入真的会进请求体：必需的那一项缺失就在前端拦掉
    // （每条视频都按次计费，废任务不该排进队列）。服务端按同一口径二次校验。
    const intent=seedanceModeFlags();
    if(intent.image&&!state.seedanceImagePath&&!$('#seedanceImageUrl').value.trim()){
      message.textContent='生成方式需要图片：请上传参考图或填写图片地址（或改成「仅文生视频」/「带参考视频」）';return}
    if(intent.video&&!$('#seedanceVideoUrl').value.trim()){
      message.textContent='生成方式需要参考视频地址：本地上传的视频只归档留痕、不会提交给上游（该接口只接受 URL）';return}
    if(tasks>1 && !confirm(`本次将提交 ${tasks} 条视频生成任务（每条一次生成，按条计费），确认立即排队执行？`)) return;
  }
  // 生成方式与提交体共用同一判定（seedanceModeFlags）：界面上被隐藏的输入，请求里也不会带。
  const videoMode=isVideo?seedanceModeFlags():{mode:null,image:false,video:false};
  // 产物类用例（鹈鹕骑车 SVG）：请求轮数 / 并发数走专门的输入框，只在选中该用例时生效。
  const isPelican=effectiveTestKind==='gpt'&&!!selectedItem?.multi_round;
  const pelicanRounds=isPelican?Math.max(1,Number($('#gptRequests').value)||1):null;
  const pelicanConcurrency=isPelican?Math.max(1,Number($('#gptConcurrency').value)||1):null;
  if(isPelican&&(pelicanRounds>(selectedItem.multi_round.max_requests||1)||pelicanConcurrency>(selectedItem.multi_round.max_concurrency||1))){
    message.textContent=`请求轮数最多 ${selectedItem.multi_round.max_requests}，并发数最多 ${selectedItem.multi_round.max_concurrency}`;return;
  }
  if(((isImage||isGeminiImage) && !selectedItem?.fixed_parameters || selectedItem?.manual_parameters) && (!Number($('#customRequests').value) || !Number($('#customConcurrency').value))){message.textContent='请填写总请求数和并发数';return}
  const imageLoadOptions={};
  if((isImage||isGeminiImage)&&!selectedItem?.fixed_parameters) imageLoadOptions.image_prompt=$('#imagePrompt').value;
  if(isImage&&!selectedItem?.fixed_parameters){
    // 压测持续时长**默认不启用**：勾选后才提交该参数；不勾选 = 按总请求数跑满为止。
    if($('#imageHoldEnabled').checked){
      const hold=Number($('#imageHoldSeconds').value);
      if(!Number.isFinite(hold)||hold<=0){message.textContent='压测持续时长必须是大于 0 的秒数';return}
      imageLoadOptions.image_hold_seconds=hold;
    }
    if($('#imageAspectEnabled').checked) imageLoadOptions.image_aspect_ratio=$('#imageAspectRatio').value;
  }
  const button=$('#submitBtn'); button.disabled=true;
  // 执行范围只在这一批**全部**都声明了 model_scope 时提交：混选普通卡片会让服务端忽略
  // case_models（按单模型跑），提前判定可避免"以为多模型其实只跑了一个"。
  const scopeAllDeclared=state.selected.length>0&&state.cases.filter(x=>state.selected.includes(x.id)).every(x=>x.model_scope);
  // 测试面是**批次级**开关：任一所选卡片勾了「含通道面」就整批带上；
  // 都不勾则只跑模型能力与验真（默认，也是服务端缺省）。
  const selectedFaces=state.selected.some(id=>state.faceChannel[id])?'all':'model,verify';
  try {
    const run=await api('/api/runs',{method:'POST',body:JSON.stringify({...imageLoadOptions,protocol:state.protocol,path_override:state.customPath||null,test_kind:effectiveTestKind,claude_variant:effectiveTestKind==='fable'?state.claudeVariant:null,base_url:$('#baseUrl').value.trim(),api_key:$('#apiKey').value.trim(),model:$('#model').value.trim(),cases:state.selected,temporary_case:(selectedItem?.case_options&&!selectedItem.billing_probe)?state.temporaryCase:null,billing_probe:(effectiveTestKind==='fable'&&selectedItem?.billing_probe)?(state.billingProbe||'BILLING-ALL'):null,case_models:(scopeAllDeclared&&state.caseModels.length>1)?state.caseModels:null,faces:selectedFaces,quick,requests:(effectiveTestKind==='load'||isImage||isGeminiImage||isVideo)&&$('#customEnabled').checked?Number($('#customRequests').value):pelicanRounds,concurrency:effectiveTestKind==='moderation'?Number($('#moderationConcurrency').value):((state.testKind==='load'||isImage||isGeminiImage||isVideo)&&$('#customEnabled').checked?Number($('#customConcurrency').value):pelicanConcurrency),image_quality:isImage?selectedValues('#imageQuality'):null,image_models:isImage?selectedValues('#imageModels'):null,image_sizes:isImage?selectedValues('#imageSizes'):null,gemini_quality:isGeminiImage?selectedValues('#geminiQuality'):null,gemini_models:isGeminiImage?selectedValues('#geminiModels'):null,gemini_aspects:isGeminiImage?selectedValues('#geminiAspects'):null,gemini_counts:isGeminiImage?selectedValues('#geminiCounts').map(Number):null,chat_prompt:isChat?$('#chatPrompt').value:null,chat_system:isChat?$('#chatSystem').value:null,chat_max_tokens:isChat?Number($('#chatMaxTokens').value)||8192:null,signature_max_tokens:(effectiveTestKind==='fable'&&state.selected.includes(SIGNATURE_CASE))?(Number($('#signatureMaxTokens').value)||16000):null,chat_temperature:isChat&&$('#chatTemperature').value!==''?Number($('#chatTemperature').value):null,seedance_prompt:isVideo?$('#seedancePrompt').value:null,seedance_mode:videoMode.mode,seedance_resolution:isVideo?$('#seedanceResolution').value:null,seedance_ratio:isVideo?$('#seedanceRatio').value:null,seedance_duration:isVideo?Number($('#seedanceDuration').value):null,seedance_image_role:videoMode.image?$('#seedanceImageRole').value:null,seedance_image_path:videoMode.image?(state.seedanceImagePath||null):null,seedance_image_url:videoMode.image?($('#seedanceImageUrl').value.trim()||null):null,seedance_video_role:videoMode.video?$('#seedanceVideoRole').value:null,seedance_video_path:videoMode.video?(state.seedanceVideoPath||null):null,seedance_video_url:videoMode.video?($('#seedanceVideoUrl').value.trim()||null):null,seedance_audio:isVideo?$('#seedanceAudio').checked:null,seedance_watermark:isVideo?$('#seedanceWatermark').checked:null})});
    state.runsPage=1; await refreshRuns(); toast(`${state.selected.join('、')} 已加入执行队列`);
    showLatestRun(run);
    if(isChat){state.chatRunId=run.id; pollChatRun(run.id).catch(e=>setChatStatus(e.message,'error'));}
  } catch(e){message.textContent=e.message} finally {button.disabled=false}
});
$('#connectivityBtn').onclick=async()=>{
  const button=$('#connectivityBtn'), result=$('#connectivityResult');
  const base_url=$('#baseUrl').value.trim(), api_key=$('#apiKey').value.trim(), model=$('#model').value.trim();
  result.className='connectivity-result show';
  if(!base_url || !api_key || !model){result.textContent='请先填写 Base URL、API Key 和 Model';result.classList.add('error');return}
  button.disabled=true;button.querySelector('span').textContent='测试中…';result.textContent='正在发送一个小请求检查连接…';
  try{
    const response=await fetch('/api/connectivity/check',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({protocol:state.protocol,path_override:state.customPath||null,base_url,api_key,model})});
    const data=await response.json();
    if(!response.ok || !data.passed){
      const details=[data.error||'连通性测试失败',data.diagnosis,data.endpoint&&`请求地址：${data.endpoint}`,data.provider_request_id&&`请求 ID：${data.provider_request_id}`].filter(Boolean);
      throw new Error(details.join('；'));
    }
    result.textContent=`连接成功 · HTTP ${data.http_status} · ${Number(data.duration_ms).toFixed(0)} ms`;
    result.classList.add('success');toast('连通性测试通过');
  }catch(e){result.textContent=`连接失败：${e.message}`;result.classList.add('error')}
  finally{button.disabled=false;button.querySelector('span').textContent='测试连通性'}
};

$('#queryModels').onclick=async()=>{
  const button=$('#queryModels'), hint=$('#modelHint'); hint.className='field-hint';
  const base_url=$('#baseUrl').value.trim(), api_key=$('#apiKey').value.trim();
  if(!base_url || !api_key){hint.textContent='请先填写 Base URL 和 API Key';hint.classList.add('error');return}
  button.disabled=true; button.textContent='查询中…'; hint.textContent='正在请求模型列表…';
  try{
    const result=await api('/api/models/query',{method:'POST',body:JSON.stringify({protocol:state.protocol,path_override:state.customPath||null,base_url,api_key})});
    state.models=result.models||[];
    renderCases();
    hint.textContent=result.count?`已查询到 ${result.count} 个模型；输入名称或 ID 可模糊筛选`:'接口返回成功，但未发现可用于生成内容的模型';
    hint.classList.add(result.count?'success':'error');
    if(result.count===1)$('#model').value=result.models[0].id;
    renderModelSuggestions($('#model').value);renderTestModels();
  }catch(e){hint.textContent=e.message;hint.classList.add('error')}finally{button.disabled=false;button.textContent='查询模型'}
};
function fuzzyModelMatch(model, query){
  const haystack=`${model.id} ${model.name}`.toLowerCase(), needle=query.trim().toLowerCase();
  if(!needle)return true;if(haystack.includes(needle))return true;
  let index=0;for(const char of haystack){if(char===needle[index])index++;if(index===needle.length)return true}return false;
}
function hideModelSuggestions(){const box=$('#modelSuggestions');box.classList.remove('show');box.innerHTML='';$('#model').setAttribute('aria-expanded','false')}
function selectSuggestedModel(id){state.caseModels=[];$('#model').value=id;hideModelSuggestions();$('#model').focus();updateEndpointPreview();renderCases();persistForm()}
function renderModelSuggestions(query=''){
  const box=$('#modelSuggestions');if(!state.models.length){hideModelSuggestions();return}
  const matches=state.models.filter(model=>fuzzyModelMatch(model,query)).slice(0,50);
  box.innerHTML=matches.length?matches.map(model=>`<button type="button" role="option" data-model-id="${escapeHtml(model.id)}"><b>${escapeHtml(model.name)}</b><small>${escapeHtml(model.id)}</small></button>`).join(''):'<div class="no-model-match">没有匹配模型，可继续使用手工输入值</div>';
  box.classList.add('show');$('#model').setAttribute('aria-expanded','true');
  box.querySelectorAll('[data-model-id]').forEach(button=>button.onmousedown=event=>{event.preventDefault();selectSuggestedModel(button.dataset.modelId)});
}
$('#testModelSearch').oninput=()=>renderTestModels();
$('#model').oninput=()=>{state.caseModels=[];renderModelSuggestions($('#model').value);updateEndpointPreview();renderCases();persistForm()};
$('#apiKey').oninput=()=>persistForm();
$('#baseUrl').oninput=()=>{updateEndpointPreview();persistForm()};
$('#customPath').oninput=()=>{state.customPath=$('#customPath').value.trim();updateEndpointPreview();persistForm()};
$('#baseUrl').onblur=()=>{const input=$('#baseUrl');input.value=input.value.trim().replace(/\/+$/,'');updateEndpointPreview();persistForm()};
$('#model').onfocus=()=>{if(state.models.length)renderModelSuggestions($('#model').value)};
$('#model').onkeydown=event=>{if(event.key==='Escape')hideModelSuggestions()};
document.addEventListener('mousedown',event=>{if(!event.target.closest('.model-combobox'))hideModelSuggestions()});
$('#toggleKey').onclick=()=>{const input=$('#apiKey');const masked=input.classList.toggle('secret-masked');$('#toggleKey').textContent=masked?'显示':'隐藏'};
$('#customRequests').oninput=()=>{$('#customRequests').dataset.edited='1';persistForm()}; $('#customConcurrency').oninput=()=>{$('#customConcurrency').dataset.edited='1';persistForm()};
$('#customEnabled').onchange=()=>{ $('#customRequests').disabled=!$('#customEnabled').checked; $('#customConcurrency').disabled=!$('#customEnabled').checked; persistForm(); };
$('#imageAspectEnabled').onchange=()=>{$('#imageAspectRatio').disabled=!$('#imageAspectEnabled').checked;};
$('#imageHoldEnabled').onchange=()=>{$('#imageHoldEnabled').dataset.edited='1';syncImageHoldState();persistForm();};
$('#imageHoldSeconds').oninput=()=>{$('#imageHoldSeconds').dataset.edited='1';persistForm();};
$('#imagePrompt').oninput=()=>persistForm();
$('#resetCustom').onclick=()=>{delete $('#customRequests').dataset.edited; delete $('#customConcurrency').dataset.edited; if(state.testKind==='gpt-image'){delete $('#imageHoldEnabled').dataset.edited; delete $('#imageHoldSeconds').dataset.edited; $('#imageHoldEnabled').checked=false; $('#imageAspectEnabled').checked=false;$('#imageAspectRatio').value='1:1';} updateCustomDefaults();persistForm()};
$('#refreshBtn').onclick=()=>refreshRuns(false);
// —— 对话窗口：发送 / 复制 / 原文视图 ——
$('#chatSend').onclick=()=>{
  if(state.testKind!=='chat')return;
  const message=$('#formMessage');
  if(!$('#chatPrompt').value.trim()){message.textContent='请先填写要发送的用户消息';$('#chatPrompt').focus();return}
  message.textContent='';
  $('#runForm').requestSubmit();
};
$('#chatCopy').onclick=async()=>{
  const text=state.chatRawText||'';
  if(!text){toast('暂无可复制的响应内容');return}
  try{await navigator.clipboard.writeText(text);toast(`已复制响应原文（${text.length} 字符）`)}
  catch(e){toast('复制失败，可切到「原文视图」手动选择')}
};
$('#chatRaw').onclick=()=>{
  const box=$('#chatResult');
  if(!box||!box.classList.contains('show')){toast('暂无响应内容');return}
  const status=$('#chatStatus'), statusClass=status?status.className:'';
  renderChatResult(state.chatRawText,null,!state.chatRawView);
  if(status) status.className=statusClass;
  $('#chatRaw').textContent=state.chatRawView?'渲染视图':'原文视图';
};
$('#chatPrompt').oninput=()=>persistForm();
$('#chatSystem').oninput=()=>persistForm();
$('#chatMaxTokens').oninput=()=>{$('#chatMaxTokens').dataset.edited='1';persistForm()};
$('#chatTemperature').oninput=()=>persistForm();
if($('#signatureMaxTokens')) $('#signatureMaxTokens').oninput=()=>{$('#signatureMaxTokens').dataset.edited='1';persistForm()};
// —— Seedance 视频生成：上传图片 + 参数持久化 ——
// 上传成功后清空文件输入，这样重复选同一个文件仍会触发 change；移除图片用「清除」按钮。
$('#seedanceImageFile').onchange=async event=>{
  const file=event.target.files&&event.target.files[0];
  await uploadSeedanceImage(file||null);
  event.target.value='';
};
$('#seedanceImageClear').onclick=()=>{
  if(!state.seedanceImagePath&&!state.seedanceUploadError){toast('当前没有已上传的图片');return}
  state.seedanceImagePath='';state.seedanceImageName='';state.seedanceUploadError='';
  renderSeedanceConfig();toast('已移除本地上传的图片（图片地址不受影响）');
};
// 参考视频同理：上传成功后清空文件输入，重复选同一个文件仍会触发 change。
$('#seedanceVideoFile').onchange=async event=>{
  const file=event.target.files&&event.target.files[0];
  await uploadSeedanceVideo(file||null);
  event.target.value='';
};
$('#seedanceVideoClear').onclick=()=>{
  if(!state.seedanceVideoPath&&!state.seedanceVideoError){toast('当前没有已上传的参考视频');return}
  state.seedanceVideoPath='';state.seedanceVideoName='';state.seedanceVideoError='';
  renderSeedanceConfig();toast('已移除本地上传的参考视频（视频地址不受影响）');
};
$('#seedancePrompt').oninput=()=>persistForm();
$('#seedanceImageUrl').oninput=()=>persistForm();
$('#seedanceVideoUrl').oninput=()=>persistForm();
$('#seedanceDuration').oninput=()=>persistForm();
$('#seedanceResolution').onchange=()=>persistForm();
// 生成方式：切换立刻重排面板（隐藏 / 显示对应的输入与位置参数），无需等下一次整面板重渲染。
$('#seedanceMode').onchange=()=>{applySeedanceModeVisibility();persistForm()};
$('#seedanceRatio').onchange=()=>persistForm();
$('#seedanceImageRole').onchange=()=>persistForm();
$('#seedanceVideoRole').onchange=()=>persistForm();
$('#seedanceAudio').onchange=()=>persistForm();
$('#seedanceWatermark').onchange=()=>persistForm();

// 批量报告：点击套件行展开 / 收起该集合下的用例级结果（事件委托，表格由 renderBatchReport 注入）。
$('#reportBody').addEventListener('click',event=>{
  const row=event.target.closest('.suite-row'); if(!row) return;
  const detail=row.nextElementSibling;
  if(!detail || !detail.classList.contains('suite-detail')) return;
  const open=detail.hidden;
  detail.hidden=!open; row.classList.toggle('open',open); row.setAttribute('aria-expanded',String(open));
});
document.querySelectorAll('#runForm input, #runForm select').forEach(input=>input.addEventListener('change',persistForm));
function applyKindTabsState(){
  $('#kindTabs').classList.remove('collapsed');
  document.querySelectorAll('.kind-tab').forEach(tab=>tab.classList.remove('extra'));
  $('#kindTabsToggle').hidden=true;
}
document.querySelectorAll('.kind-tab').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('.kind-tab').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.testKind=button.dataset.kind;state.selected=[];state.activeCase='';if(state.testKind!=='kvv')state.temporaryCase='';if(state.testKind!=='fable')state.billingProbe=''; delete $('#customRequests').dataset.edited; delete $('#customConcurrency').dataset.edited; $('#customEnabled').checked=false;
  $('#quick').closest('.mode-toggle').style.display=state.testKind==='load'?'flex':'none';
  if(state.testKind==='fable' && !$('#model').value.trim()) $('#model').value=CLAUDE_VARIANT_MODELS[state.claudeVariant];
  setModelPlaceholder();
  renderGptVariantConfig();
  renderClaudeVariantConfig();
  renderGeminiVariantConfig();
  renderQwenVariantConfig();
  renderSeedanceConfig();
  renderChatConfig();
  renderCases();
  let first=state.cases.find(x=>kindMatches(x)&&protocolSupports(x));
  if(!first){const matchingProtocol=state.protocols.find(protocol=>state.cases.some(x=>kindMatches(x)&&protocolSupports(x,protocol.id)));if(matchingProtocol){selectProtocol(matchingProtocol.id);first=state.cases.find(x=>kindMatches(x)&&protocolSupports(x));}}
  if(first)state.selected=[first.id];renderCases();updateSelectionUI();updateEndpointPreview();
  if((state.testKind==='gpt-image'||state.testKind==='gemini-image') && first && !$('#model').value) $('#model').value=first.image_models?.[0]||first.title;
  if(state.testKind==='gpt' && !$('#model').value) $('#model').value=GPT_VARIANT_MODELS[state.gptVariant];
  if(state.testKind==='gemini' && !$('#model').value) $('#model').value=state.geminiModel;
  if(state.testKind==='qwen' && !$('#model').value) $('#model').value=state.qwenModel;
  applyKindTabsState();
  persistForm();
});
document.querySelectorAll('.filter[data-filter]').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('.filter[data-filter]').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.filter=button.dataset.filter;state.runsPage=1;refreshRuns();
});
document.querySelectorAll('.cfilter').forEach(button=>button.onclick=()=>{
  document.querySelectorAll('.cfilter').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.conclusion=button.dataset.conclusion;state.runsPage=1;refreshRuns();
});
$('#runsPrev').onclick=()=>{if(state.runsPage>1){state.runsPage--;refreshRuns()}};
$('#runsNext').onclick=()=>{state.runsPage++;refreshRuns()};
$('#runsPageSize').onchange=event=>{state.runsPageSize=Number(event.target.value);state.runsPage=1;refreshRuns()};
let runsSearchTimer;
$('#runsSearch').oninput=event=>{
  state.runsSearch=event.target.value.trim();state.runsPage=1;clearTimeout(runsSearchTimer);
  runsSearchTimer=setTimeout(()=>refreshRuns(),250);
};

setTheme(document.documentElement.dataset.theme);
$('#themeToggle').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark',true);
const themePreference=window.matchMedia('(prefers-color-scheme: dark)');
themePreference.addEventListener('change',event=>{
  try{const saved=localStorage.getItem('ml-test-platform.theme.v1');if(saved==='dark'||saved==='light')return}catch(e){}
  setTheme(event.matches?'dark':'light');
});
window.addEventListener('storage',event=>{if(event.key==='ml-test-platform.theme.v1')setTheme(event.newValue==='dark'||event.newValue==='light'?event.newValue:themePreference.matches?'dark':'light')});
document.querySelectorAll('[data-page]').forEach(button=>button.onclick=()=>showPage(button.dataset.page));
window.addEventListener('hashchange',()=>showPage(location.hash==='#reports'?'reports':'test',false));
$('#reportBody').addEventListener('keydown',event=>{if(event.target.matches('.suite-row')&&['Enter',' '].includes(event.key)){event.preventDefault();event.target.click()}});
$('#refreshBtn').onclick=()=>{refreshRuns(false);if(workspace.page==='reports'&&workspace.report)openArtifact(workspace.report.id,workspace.view)};
showPage(location.hash==='#reports'?'reports':'test',false);
fitTestWorkspaceToViewport();

(async function init(){
  try {
    const saved=readPersistedForm();
    state.testKind=['load','kvv','deepseek','glm','qwen','gpt','fable','gemini','gpt-image','gemini-image','seedance','open-source','moderation','chat','forensics'].includes(saved.testKind)?saved.testKind:'load';
    if(saved.claudeVariant in CLAUDE_VARIANT_MODELS)state.claudeVariant=saved.claudeVariant;
    if(saved.gptVariant in GPT_VARIANT_MODELS)state.gptVariant=saved.gptVariant;
    if(typeof saved.geminiModel==='string'&&saved.geminiModel.trim())state.geminiModel=saved.geminiModel.trim();
    if(typeof saved.qwenModel==='string'&&saved.qwenModel.trim())state.qwenModel=saved.qwenModel.trim();
    document.querySelectorAll('.kind-tab').forEach(button=>button.classList.toggle('active',button.dataset.kind===state.testKind));
    applyKindTabsState();
    $('#quick').closest('.mode-toggle').style.display=state.testKind==='load'?'flex':'none';
    state.protocols=await api('/api/protocols');
    state.protocol=state.protocols.some(item=>item.id===saved.protocol)?saved.protocol:'openai';
    renderProtocols(); selectProtocol(state.protocol);
    if(typeof saved.customPath==='string'&&saved.customPath.trim()){state.customPath=saved.customPath.trim();syncCustomPathField();updateEndpointPreview();}
    state.cases=await api('/api/cases');
    if(saved.baseUrl!=null)$('#baseUrl').value=String(saved.baseUrl);
    if(saved.apiKey!=null)$('#apiKey').value=String(saved.apiKey);
    if(saved.model!=null)$('#model').value=String(saved.model);
    if(saved.temporaryCase!=null)state.temporaryCase=String(saved.temporaryCase);
    if(saved.billingProbe!=null)state.billingProbe=String(saved.billingProbe);
    if(Array.isArray(saved.caseModels))state.caseModels=saved.caseModels.map(String).filter(Boolean);
    if(saved.faceChannel&&typeof saved.faceChannel==='object'&&!Array.isArray(saved.faceChannel)){
      state.faceChannel=Object.fromEntries(Object.entries(saved.faceChannel).filter(([,v])=>v===true));
    }
    if(typeof saved.quick==='boolean')$('#quick').checked=saved.quick;
    if(typeof saved.customEnabled==='boolean')$('#customEnabled').checked=saved.customEnabled;
    if(saved.customRequests!=null){$('#customRequests').value=String(saved.customRequests);$('#customRequests').dataset.edited='1'}
    if(saved.imageHoldEnabled!=null){$('#imageHoldEnabled').checked=Boolean(saved.imageHoldEnabled);$('#imageHoldEnabled').dataset.edited='1'}
    if(saved.imageHoldSeconds!=null){$('#imageHoldSeconds').value=String(saved.imageHoldSeconds);$('#imageHoldSeconds').dataset.edited='1'}
    if(saved.imagePrompt!=null)$('#imagePrompt').value=String(saved.imagePrompt);
    if(saved.customConcurrency!=null){$('#customConcurrency').value=String(saved.customConcurrency);$('#customConcurrency').dataset.edited='1'}
    if(saved.gptRequests!=null&&String(saved.gptRequests).trim()!=='')$('#gptRequests').value=String(saved.gptRequests);
    if(saved.gptConcurrency!=null&&String(saved.gptConcurrency).trim()!=='')$('#gptConcurrency').value=String(saved.gptConcurrency);
    if(saved.moderationConcurrency!=null)$('#moderationConcurrency').value=String(saved.moderationConcurrency);
    if(saved.chatPrompt!=null)$('#chatPrompt').value=String(saved.chatPrompt);
    if(saved.chatSystem!=null)$('#chatSystem').value=String(saved.chatSystem);
    if(saved.chatMaxTokens!=null&&String(saved.chatMaxTokens).trim()!==''){$('#chatMaxTokens').value=String(saved.chatMaxTokens);$('#chatMaxTokens').dataset.edited='1'}
    if(saved.chatTemperature!=null)$('#chatTemperature').value=String(saved.chatTemperature);
    if(saved.signatureMaxTokens!=null&&String(saved.signatureMaxTokens).trim()!==''&&$('#signatureMaxTokens')){$('#signatureMaxTokens').value=String(saved.signatureMaxTokens);$('#signatureMaxTokens').dataset.edited='1'}
    if(saved.seedancePrompt!=null)$('#seedancePrompt').value=String(saved.seedancePrompt);
    if(saved.seedanceImageUrl!=null)$('#seedanceImageUrl').value=String(saved.seedanceImageUrl);
    if(saved.seedanceVideoUrl!=null)$('#seedanceVideoUrl').value=String(saved.seedanceVideoUrl);
    if(typeof saved.seedanceAudio==='boolean')$('#seedanceAudio').checked=saved.seedanceAudio;
    if(typeof saved.seedanceWatermark==='boolean')$('#seedanceWatermark').checked=saved.seedanceWatermark;
    restoreCaseSelection(saved.selected);
    renderCases(); updateSelectionUI(); renderGptVariantConfig(); renderClaudeVariantConfig(); renderGeminiVariantConfig(); renderQwenVariantConfig(); renderSeedanceConfig(); renderChatConfig();
    [['#imageQuality',saved.imageQuality],['#imageModels',saved.imageModels],['#imageSizes',saved.imageSizes],['#geminiQuality',saved.geminiQuality],['#geminiModels',saved.geminiModels],['#geminiAspects',saved.geminiAspects],['#geminiCounts',saved.geminiCounts]].forEach(([selector,values])=>{
      if(!Array.isArray(values))return;
      const element=$(selector); if(!element)return;
      [...element.options].forEach(option=>option.selected=values.includes(option.value));
    });
    renderMultiSelects();
    // Seedance 的单选下拉由 renderSeedanceConfig() 现场填选项，上面的单选回填必须在它之后，
    // 否则用户上次选的档位会被套件的默认值冲掉。
    [['#seedanceMode',saved.seedanceMode],['#seedanceResolution',saved.seedanceResolution],['#seedanceRatio',saved.seedanceRatio],['#seedanceImageRole',saved.seedanceImageRole],['#seedanceVideoRole',saved.seedanceVideoRole]].forEach(([selector,value])=>{
      if(value==null)return;
      const element=$(selector); if(!element)return;
      if([...element.options].some(option=>option.value===String(value))) element.value=String(value);
    });
    if(saved.seedanceDuration!=null&&String(saved.seedanceDuration).trim()!=='')$('#seedanceDuration').value=String(saved.seedanceDuration);
    renderTestModels();
    applySeedanceModeVisibility();
    renderSeedanceUploadState();
    renderSeedanceVideoState();
    updateEndpointPreview(); persistForm(); await refreshRuns();
  }
  catch(e){$('#serviceState').innerHTML='<i style="background:#fb7185"></i>服务异常';toast(e.message)}
  setInterval(()=>refreshRuns(),3000); setInterval(()=>{state.runs.filter(r=>['queued','running'].includes(r.status)).forEach(r=>{const el=document.querySelector(`[data-start="${r.id}"]`);if(el)el.textContent=formatDuration(r)})},1000);
})();
