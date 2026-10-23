// Read-only capability projection, with a legacy score-face fallback.
const CAPABILITY_AXES = [
  {key:'protocol',label:'协议合规',faces:['core','params','envelope'],description:'核心对话、参数与错误契约'},
  {key:'tools',label:'工具调用',faces:['tools'],description:'工具选择、调用参数与 Agent 闭环'},
  {key:'thinking',label:'推理能力',faces:['thinking'],description:'思考模式、推理输出与控制'},
  {key:'context',label:'上下文',faces:['context'],description:'上下文处理、长文本与缓存'},
  {key:'vision',label:'多模态',faces:['vision'],description:'视觉输入与图像理解'},
  {key:'quality',label:'能力质量',faces:['quality'],description:'专项题目的回答质量；工具、推理、上下文等能力测试分别展示在对应维度'},
];
const radarEscape = value => String(value ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const radarNumber = value => value.toFixed(1);
function capabilityDimensions(report) {
  if(Array.isArray(report?.capabilities?.axes))return CAPABILITY_AXES.map(axis=>{
    const item=report.capabilities.axes.find(d=>d.key===axis.key);
    if(!item)return {...axis,parts:[],cases:[],weight:0,measured:0,earned:0,coverage:0,score:null,rating:'未评估',state:'暂无用例',evaluated:0,case_count:0};
    const score=typeof item.score==='number'&&Number.isFinite(item.score)?Math.max(0,Math.min(100,item.score)):null;
    return {...axis,...item,parts:[],score,rating:score===null?'未评估':score>=90?'优秀':score>=80?'良好':score>=70?'一般':score>=60?'可用':'待提升'};
  });
  const faces = Array.isArray(report?.faces) ? report.faces : [];
  const nonnegative = value => typeof value==='number' && Number.isFinite(value) ? Math.max(0,value) : 0;
  return CAPABILITY_AXES.map(axis=>{
    const parts=faces.filter(face=>axis.faces.includes(face.face)&&nonnegative(face.weight)>0);
    const weight=parts.reduce((sum,f)=>sum+nonnegative(f.weight),0);
    const measured=parts.reduce((sum,f)=>sum+Math.min(nonnegative(f.weight),nonnegative(f.measured_weight)),0);
    const earned=parts.reduce((sum,f)=>sum+Math.min(nonnegative(f.weight),nonnegative(f.earned)),0);
    const coverage=weight?Math.min(1,measured/weight):0;
    const score=weight&&measured>0&&coverage>=0.8-1e-6?Math.min(100,earned/weight*100):null;
    const rating=score===null?'未评估':score>=90?'优秀':score>=80?'良好':score>=70?'一般':score>=60?'可用':'待提升';
    return {...axis,parts,weight,measured,earned,coverage,score,rating,
      state:score!==null?'已评估':!weight?'无计分项':measured>0?'证据不足':'未评估'};
  });
}
function capabilityChart(dimensions, model) {
  const cx=250,cy=205,radius=132;
  const point=(i,r)=>[cx+Math.sin(i*Math.PI/3)*r,cy-Math.cos(i*Math.PI/3)*r];
  const coordinates=points=>points.map(p=>p.map(n=>n.toFixed(2)).join(',')).join(' ');
  const vertices=dimensions.map((d,i)=>d.score===null?null:point(i,radius*d.score/100));
  const grid=[25,50,75,100].map(level=>`<polygon class="${level===100?'radar-target':''}" points="${coordinates(dimensions.map((_,i)=>point(i,radius*level/100)))}"/>`).join('');
  const spokes=dimensions.map((_,i)=>`<line x1="${cx}" y1="${cy}" x2="${point(i,radius)[0]}" y2="${point(i,radius)[1]}"/>`).join('');
  const area=vertices.every(Boolean)?`<polygon class="radar-area" points="${coordinates(vertices)}"/>`:
    vertices.map((p,i)=>p&&vertices[(i+1)%6]?`<line class="radar-segment" x1="${p[0]}" y1="${p[1]}" x2="${vertices[(i+1)%6][0]}" y2="${vertices[(i+1)%6][1]}"/>`:'').join('');
  const labels=dimensions.map((d,i)=>{
    const [x,y]=point(i,radius+38),align=i===0||i===3?'middle':i<3?'start':'end';
    const tooltip=`${d.label}：${d.score===null?d.state:radarNumber(d.score)+' / 100，'+d.rating}${d.weight?'；证据覆盖 '+Math.round(d.coverage*100)+'%':'；不代表没有执行能力测试'}`;
    const dot=vertices[i]?`<circle class="radar-dot" cx="${vertices[i][0]}" cy="${vertices[i][1]}" r="4"/>`:'';
    return `<g class="radar-axis ${d.score===null?'radar-unassessed':''}" tabindex="0" role="img" aria-label="${radarEscape(tooltip)}"><title>${radarEscape(tooltip)}</title>${dot}<text x="${x}" y="${y-3}" text-anchor="${align}" class="radar-label">${radarEscape(d.label)}</text><text x="${x}" y="${y+16}" text-anchor="${align}" class="radar-value">${d.score===null?d.state:radarNumber(d.score)}</text></g>`;
  }).join('');
  return `<svg class="capability-svg" viewBox="0 0 500 420" role="img" aria-label="${radarEscape(model||'当前模型')}六维能力雷达图，每轴满分100，未评估维度不绘制得分"><g class="radar-grid">${grid}${spokes}</g>${area}<g class="radar-ticks">${[25,50,75,100].map(n=>`<text x="${cx+5}" y="${cy-radius*n/100+4}">${n}</text>`).join('')}</g>${labels}</svg>`;
}
function capabilityEvaluation(report, dimensions) {
  if(report.valid===false)return `<p class="radar-evaluation radar-invalid"><b>总评分无效</b><span>${radarEscape(report.invalid_reason||'信任校验存在失败项')}。图中仅展示已有能力证据。</span></p>`;
  const evaluated=dimensions.filter(d=>d.score!==null);
  if(!evaluated.length)return '<p class="radar-evaluation"><b>等待更多测试证据</b><span>各维度有效证据覆盖达到 80% 后展示分值。</span></p>';
  const strongest=[...evaluated].sort((a,b)=>b.score-a.score)[0];
  const weakest=[...evaluated].sort((a,b)=>a.score-b.score)[0];
  const strength=strongest.score>=80?`优势维度：${strongest.label} ${radarNumber(strongest.score)}`:'已测维度仍有提升空间';
  const focus=weakest.score<80?`建议关注${weakest.label}（${radarNumber(weakest.score)}），展开下方查看依据。`:'已评估维度均达到良好及以上。';
  const missing=dimensions.filter(d=>d.score===null).map(d=>d.label);
  return `<p class="radar-evaluation"><b>${radarEscape(strength)}</b><span>${radarEscape(focus)}${missing.length?' 未形成分值：'+radarEscape(missing.join('、'))+'（见下方依据）。':''}</span></p>`;
}
function capabilityUnscoredEvidence(report, manifest) {
  // Use authoritative, deduplicated case verdicts. The manifest only supplies names.
  // In particular, a zero scoring weight is not a failed capability test.
  const names=new Map();
  const collect=(node,inheritedModel='')=>{
    if(!node||typeof node!=='object')return;
    const model=node.model||inheritedModel;
    const id=node.case_id||node.id;
    if(id&&node.name&&model===report.model)names.set(id,node.name);
    if(node.summary)collect(node.summary,model);
    if(Array.isArray(node.results))node.results.forEach(row=>collect(row,model));
  };
  collect(manifest);
  const radarCases=new Set((report.capabilities?.axes||[]).flatMap(axis=>(axis.cases||[]).map(c=>c.case_id)));
  const rows=Object.values(report.case_scores||{}).filter(row=>row&&row.weight===0&&!radarCases.has(row.case_id));
  if(!rows.length)return '';
  const labels={passed:'通过',warning:'警告',failed:'失败',error:'错误',blocked:'受阻',not_evaluable:'无法评价',interrupted:'已中断',skipped:'未执行'};
  const passed=rows.filter(row=>row.status==='passed').length;
  return `<details class="radar-dimension radar-unscored"><summary><span>未计分测试证据 · ${rows.length} 项</span><small>${passed} 项通过</small></summary><div class="radar-evidence"><p>这些用例的计分权重为 0，仍保留原始裁定；身份、汇总或未登记用例不能直接换算为能力分。</p>${rows.map(row=>`<p>${radarEscape(row.case_id)} · ${radarEscape(names.get(row.case_id)||'测试用例')}：<b>${labels[row.status]||'无法评价'}</b></p>`).join('')}</div></details>`;
}
function capabilityEvidence(d) {
  if(!Array.isArray(d.cases))return d.parts.length?d.parts.map(p=>`<p>${p.side==='channel'?'通道侧':'模型侧'} · ${radarEscape(p.label||p.face)}：${radarNumber(p.earned||0)} / ${radarNumber(p.weight)} 分</p>`).join(''):'<p>当前评分没有该维度的适用计分项。</p>';
  if(!d.cases.length)return `<p>${d.state==='不适用'?'该型号的能力声明不支持此维度，未计为失败。':'当前套件尚无此维度的适用用例。'}</p>`;
  const labels={passed:'通过',warning:'警告',failed:'失败',error:'错误',blocked:'受阻',not_evaluable:'无法评价',interrupted:'已中断',skipped:'跳过',unobserved:'未测试',not_selected:'本次未选'};
  return `<div class="radar-case-table"><table><thead><tr><th>测试用例</th><th>裁定</th><th>维度得失分</th></tr></thead><tbody>${d.cases.map(c=>`<tr><td><b>${radarEscape(c.name||c.case_id)}</b><small>${radarEscape(c.case_id)} · ${c.side==='channel'?'通道侧':'模型侧'} · 满分 ${radarNumber(c.weight||0)}</small></td><td>${labels[c.status]||'无法评价'}</td><td>${c.earned===null||c.earned===undefined?'—':`<span class="case-points-gain">+${radarNumber(c.earned)}</span>${c.lost>0?` <span class="case-points-loss">−${radarNumber(c.lost)}</span>`:''}`}</td></tr>`).join('')}</tbody></table></div>`;
}
function capabilityPanel(report, manifest) {
  const dimensions=capabilityDimensions(report);
  const unscored=capabilityUnscoredEvidence(report,manifest);
  if(!dimensions.some(d=>d.weight>0))return '<p class="radar-empty">暂无可评估的能力数据。当前评分没有适用计分项或评分数据不可用，不代表测试未执行。</p>'+unscored;
  const projected=Array.isArray(report.capabilities?.axes);
  const evaluated=dimensions.filter(d=>d.score!==null).length;
  const caseCount=dimensions.reduce((n,d)=>n+(d.case_count||0),0);
  return `<div class="radar-score-strip"><div><strong>${evaluated} / 6</strong><span>能力面已评分</span></div>${projected?`<div><strong>${caseCount}</strong><span>适用测试用例</span></div>`:''}<p>每个能力面独立满分 100<br>展开右侧维度查看测试依据</p></div><div class="radar-layout"><div class="radar-visual">${capabilityChart(dimensions,report.model)}<p class="radar-legend"><i></i> 实测得分 <span>外圈为 100 分 · 未测维度留空</span></p></div><div class="radar-assessment">${capabilityEvaluation(report,dimensions)}<div class="radar-dimensions">${dimensions.map(d=>`<details class="radar-dimension"><summary><span>${radarEscape(d.label)}</span><span class="radar-meter" aria-hidden="true"><i style="width:${d.score??0}%"></i></span><b>${d.score===null?'—':radarNumber(d.score)}</b><small class="${d.score===null?'is-unknown':d.score<60?'is-low':''}">${d.score===null?radarEscape(d.state):d.rating}</small></summary><div class="radar-evidence"><p>${radarEscape(d.description)}${d.weight?' · 证据覆盖 '+Math.round(d.coverage*100)+'%':''}${projected?' · 已评估 '+d.evaluated+' / '+d.case_count+' 项':''}</p>${capabilityEvidence(d)}${d.score===null&&d.weight?'<p>有效证据覆盖不足 80%，暂不评价；未测项不会补分。</p>':''}</div></details>`).join('')}</div></div></div>${unscored}<p class="radar-footnote">${projected?'六个能力面分别按适用用例的优先级归一为 100 分；通过获全分、警告获半分、失败不得分，已有连续评分沿用其比例。每条用例只归入一个能力面，未测项保留在分母中。':'维度分＝该维度实际得分 ÷ 适用满分 × 100。'} 雷达分不相加、不替代模型 100＋通道 20 的总评；身份声明和汇总不作为能力分。分值只代表本次用例覆盖的能力。</p>`;
}
function renderCapabilityRadar(payload, manifest=null) {
  const reports=Array.isArray(payload?.scores)?payload.scores.filter(r=>r&&typeof r==='object'):[payload];
  if(!reports.length)reports.push(null);
  const multiple=reports.length>1;
  return `<section class="capability-radar" aria-label="能力评价"><header class="radar-heading"><div><h3>能力评价</h3><p>六个能力面 · 逐项测试评分</p></div><span>基于本次测试证据</span></header>${multiple?`<div class="radar-models" role="tablist" aria-label="选择能力图模型">${reports.map((r,i)=>`<button type="button" role="tab" data-radar-model="${i}" aria-selected="${i===0}" tabindex="${i===0?0:-1}">${radarEscape(r.model||'模型 '+(i+1))}</button>`).join('')}</div>`:''}${reports.map((r,i)=>`<div data-radar-panel="${i}" ${multiple?'role="tabpanel" aria-label="'+radarEscape(r?.model||'模型 '+(i+1))+'"':''} ${i?'hidden':''}>${capabilityPanel(r||{},manifest)}</div>`).join('')}</section>`;
}
function bindCapabilityRadars(root) {
  root.querySelectorAll('.capability-radar').forEach(card=>{
    const buttons=Array.from(card.querySelectorAll('[data-radar-model]'));
    const select=button=>{
      buttons.forEach(b=>{b.setAttribute('aria-selected',String(b===button));b.tabIndex=b===button?0:-1});
      card.querySelectorAll('[data-radar-panel]').forEach(panel=>{panel.hidden=panel.dataset.radarPanel!==button.dataset.radarModel});
    };
    buttons.forEach((button,index)=>{
      button.onclick=()=>select(button);
      button.onkeydown=event=>{
        if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
        event.preventDefault();
        const next=event.key==='Home'?0:event.key==='End'?buttons.length-1:(index+(event.key==='ArrowRight'?1:buttons.length-1))%buttons.length;
        select(buttons[next]);buttons[next].focus();
      };
    });
  });
}
