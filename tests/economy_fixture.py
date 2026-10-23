"""Local oracle for deterministic economy prompts; never contacts a model."""
import json
import re

CODE = ('def solve(nums):\n    result = []\n    for n in nums:\n'
        '        if n >= 0 and nums.count(n) >= 2 and n not in result:\n'
        '            result.append(n)\n    return sorted(result)')


def request_history(payload, transport):
    if transport == 'chat': return payload['messages']
    if transport == 'responses':
        history = []
        if payload.get('instructions'): history.append({'role': 'system', 'content': payload['instructions']})
        for item in payload['input']:
            if item.get('type') == 'function_call_output':
                history.append({'role': 'tool', 'content': item['output']})
            elif item.get('role'):
                history.append(item)
        return history
    history = []
    if payload.get('system'): history.append({'role': 'system', 'content': payload['system']})
    for item in payload['messages']:
        for block in item['content']:
            if block['type'] == 'tool_result': history.append({'role': 'tool', 'content': block['content']})
            if block['type'] == 'text': history.append({'role': item['role'], 'content': block['text']})
    return history


def solve_fixture(history):
    prompts = '\n'.join(m.get('content') or '' for m in history)
    if '只能输出字符串' in prompts:
        return re.search(r'只能输出字符串([a-f0-9]+)', prompts)[1], []
    if 'arithmetic=' in prompts:
        a,b,c=map(int,re.search(r'arithmetic=\((\d+)\*(\d+)\)-(\d+)',prompts).groups())
        ca,cb,units=map(int,re.search(r'A包3件(\d+)元、B包5件(\d+)元，至少买(\d+)件',prompts).groups())
        value={'arithmetic':a*b-c,'minimum_cost':min(x*ca+y*cb for x in range(units+1) for y in range(units+1) if 3*x+5*y>=units),'logic':4096-1728}
    elif '定义 solve(nums)' in prompts: return CODE, []
    elif '订单ORD-' in prompts and '单价' in prompts:
        oid,q,p=re.search(r'订单(ORD-\d+)：A1买(\d+)件，单价(\d+)元',prompts).groups()
        value={'order_id':oid,'items':[{'sku':'A1','quantity':int(q),'unit_price':int(p)}],'total':int(q)*int(p),'paid':True,'note':None}
    elif 'HEAD=' in prompts:
        value={key:re.search(key+r'=([a-f0-9]+)',prompts)[1] for key in ('HEAD','MIDDLE','TAIL')}
    elif '高级约束题' in prompts:
        orders=json.loads(prompts.split('数据：',1)[1]); eligible=[o for o in orders if not o['cancelled']]
        ranked=sorted(eligible,key=lambda o:(-o['qty']*o['price'],o['id']))
        value={'ids':[o['id'] for o in ranked[:3]],'total':sum(o['qty']*o['price'] for o in eligible),'count':len(eligible),'checksum':sum(o['qty'] for o in ranked[:3])}
    elif '同一回复中调用两个工具' in prompts:
        order,sku=re.search(r'订单(ORD-\d+)和库存(SKU-\d+)',prompts).groups()
        return '', [('lookup_order',{'order_id':order}),('get_stock',{'sku':sku})]
    elif '工具暂时失败' in prompts:
        results=[json.loads(m['content']) for m in history if m['role']=='tool']
        if results and 'verification_code' in results[-1]: return json.dumps(results[-1],ensure_ascii=False), []
        return '', [('lookup_order',{'order_id':re.search(r'订单(ORD-\d+)',prompts)[1]})]
    else: raise AssertionError(prompts[:100])
    return json.dumps(value,ensure_ascii=False), []


def response_fixture(model, transport, text, calls):
    if transport == 'chat':
        msg={'role':'assistant','content':text}
        if calls: msg['tool_calls']=[{'id':f'call-{i}','type':'function','function':{'name':n,'arguments':json.dumps(a)}} for i,(n,a) in enumerate(calls)]
        return {'model':model,'choices':[{'message':msg,'finish_reason':'tool_calls' if calls else 'stop'}]}
    if transport == 'responses':
        output=[{'type':'function_call','call_id':f'call-{i}','name':n,'arguments':json.dumps(a)} for i,(n,a) in enumerate(calls)]
        if text: output.append({'type':'message','role':'assistant','content':[{'type':'output_text','text':text}]})
        return {'model':model,'status':'completed','output':output}
    content=[{'type':'tool_use','id':f'call-{i}','name':n,'input':a} for i,(n,a) in enumerate(calls)]
    if text: content.append({'type':'text','text':text})
    return {'type':'message','model':model,'content':content,'stop_reason':'tool_use' if calls else 'end_turn'}
