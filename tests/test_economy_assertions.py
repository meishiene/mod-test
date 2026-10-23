import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from common.probe_assertions import arithmetic_function
from dsv4.core import DeepSeekConfig, DeepSeekV4Runner
from fable5.core import Fable5Config, Fable5Runner
from gpt56.core import GPT56Config, GPT56Runner


def test_completion_evaluates_answers_without_executing_python():
    assert arithmetic_function('def add(a,b):\n return a+b', 'add', [((2,3),5),((-4,2),-2)])
    assert not arithmetic_function('def add(a,b):\n return a-b', 'add', [((2,3),5)])
    assert not arithmetic_function('def add(a,b):\n import os\n return a+b', 'add', [((2,3),5)])
    assert not arithmetic_function('def add(a,b):\n return add(a,b)', 'add', [((2,3),5)])
    assert arithmetic_function('def fib(n):\n if n<=1:\n  return n\n return fib(n-1)+fib(n-2)',
                               'fib', [((0,),0),((1,),1),((8,),21)])


def test_deepseek_reasoning_text_with_wrong_answer_fails():
    runner=DeepSeekV4Runner(DeepSeekConfig('key'))
    runner.chat=AsyncMock(return_value=SimpleNamespace(status=200,body={'choices':[{'message':{'content':'999','reasoning_content':'thinking'}}]}))
    result=asyncio.run(runner.case_default_thinking('deepseek-v4-pro'))
    assert result[0] is False


@pytest.mark.parametrize('method',['case_json_schema_output','case_g61_json_schema_output','case_auth_random_json','case_g61_auth_random_json'])
def test_gpt_schema_extra_fields_fail(method):
    runner=GPT56Runner(GPT56Config('key'))
    runner.responses=AsyncMock(return_value=SimpleNamespace(status=200,body={'output':[{'type':'message','content':[{'type':'output_text','text':'{"value":"FIXED","extra":1}'}]}]}))
    with patch('gpt56.core._rng_token',return_value='FIXED'):
        result=asyncio.run(getattr(runner,method)('gpt-6.1-sol'))
    assert result[0] is False


def test_claude_repeated_numbers_do_not_pass_sequence_check():
    runner=Fable5Runner(Fable5Config('key'))
    runner.messages_api=AsyncMock(return_value=SimpleNamespace(status=200,stream={
        'text':', '.join(['1']*1000),'usage':{'output_tokens':3000},'stop_reasons':['end_turn']}))
    result=asyncio.run(runner.case_long_output_stream('claude-fable-5'))
    assert result[0] is False
    assert runner.messages_api.call_args.kwargs['max_tokens']<=4096
