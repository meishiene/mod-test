/* Explicit combinations of existing tests. Adding a preset never starts a run. */
(function(root){
  const groups={
    load:[
      ['basic','基础验收',['T0','B1']],
      ['simple','简单压测',['SIMPLE_LOAD']],
      ['full','完整压测',['FULL_LOAD']],
      ['capacity','容量专项',['C0','C1']],
    ],
    glm:[
      ['basic','基础验收',['GLM_53_CORE','GLM_53_VERIFY']],
      ['agent','Agent 专项',['GLM_53_CORE','GLM_53_AGENT']],
      ['cache','缓存专项',['GLM_53_CORE','GLM_53_CACHE']],
      // FULL already includes core/agent/fingerprint/capability/verify; avoid duplicate runs.
      ['full','综合验收',['GLM_53_FULL','GLM_53_CACHE']],
    ],
    qwen:[
      ['basic','基础验收',['QWEN_CORE','QWEN_VERIFY']],
      ['agent','Agent 专项',['QWEN_CORE','QWEN_THINKING','QWEN_AGENT']],
      ['vision','视觉专项',['QWEN_CORE','QWEN_VISION']],
      ['full','综合验收',['QWEN_FULL']],
    ],
    kvv:[
      ['basic','基础验收',['KVV_K3_FEATURES']],
      ['agent','Agent 专项',['KVV_K3_FEATURES','KVV_K3_DYNAMIC_TOOLS']],
      ['thinking','思考专项',['KVV_K3_FEATURES','KVV_K3_THINKING_EFFORT']],
      ['temporary','供应商自测',['K3_TEMP_SUITE']],
    ],
    deepseek:[
      ['basic','基础验收',['DS_V4_P0']],
      ['enhanced','增强验收',['DS_V4_P0','DS_V4_P1']],
      ['contract','契约检查',['DS_V4_P2','DS_CONTRACT']],
    ],
    gemini:[
      ['basic','基础验收',['GEMINI3_BASIC','GEMINI3_P0']],
      ['enhanced','增强验收',['GEMINI3_BASIC','GEMINI3_P0','GEMINI3_P1']],
      ['verify','能力与验真',['GEMINI3_BASIC','GEMINI3_FINGERPRINT']],
    ],
    moderation:[
      ['basic','基础验收',['MOD_PARTNER_SMOKE']],
      ['general','通用审核',['MOD_GENERAL_COMMON']],
      ['partner','伴侣审核',['MOD_PARTNER_COMMON']],
    ],
    forensics:[
      ['basic','身份与错误取证',['IMPLICIT_IDENTITY','ERROR_ENVELOPE_LEAKAGE']],
      ['full','全部取证',['IMPLICIT_IDENTITY','IMAGE_UNDERSTANDING','ERROR_ENVELOPE_LEAKAGE']],
    ],
  };
  function forCases(kind,cases){
    let definitions=groups[kind]||[];
    if(kind==='gpt'||kind==='fable'){
      const pattern=kind==='gpt'?/^(GPT56|GPT61|GPT6)_P0$/:/^(FABLE5|OPUS5|SONNET5|HAIKU45|SONNET46)_BASIC$/;
      const prefix=cases.map(item=>item.id.match(pattern)).find(Boolean)?.[1];
      if(prefix){
        const basic=kind==='gpt'?`${prefix}_P0`:`${prefix}_BASIC`;
        definitions=[
          ['basic','基础验收',[basic]],
          ['enhanced','增强验收',[basic,`${prefix}_${kind==='gpt'?'P1':'P0'}`]],
          ['verify','能力与验真',[basic,`${prefix}_FINGERPRINT`]],
        ];
        if(kind==='fable')definitions.push(['relay','中转验真',['PROTOBUF_SIGNATURE_CHECK','ZERO_INJECTION_CHECK']]);
      }
    }
    const ids=new Set(cases.map(item=>item.id));
    const economy=cases.find(item=>item.economy_validation);
    if(economy)definitions=[['economy','经济验证',[economy.id]],...definitions];
    return definitions.filter(([, ,members])=>members.every(id=>ids.has(id)))
      .map(([id,label,caseIds])=>({id,label,caseIds:[...caseIds]}));
  }
  function isExclusive(kind,item){
    return ['open-source','gpt-image','seedance'].includes(kind)||Boolean(item.fixed_parameters||item.manual_parameters)||item.id==='K3_TEMP_SUITE';
  }
  const api={forCases,isExclusive};
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.TestCasePresets=api;
})(globalThis);
