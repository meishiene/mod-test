"""Strict assertions shared by Qwen and GLM; no provider inference."""
from __future__ import annotations

import json
import re
import ast
import operator
from typing import Any

from jsonschema import Draft202012Validator


def strict_json(raw: str, schema: dict) -> bool:
    try:
        value = json.loads(raw)
        return Draft202012Validator(schema).is_valid(value)
    except (ValueError, TypeError):
        return False


def normalized_model(value: Any) -> str:
    if not isinstance(value, str):
        return ''
    value = value.strip().lower().rsplit('/', 1)[-1]
    aliases = {'glm5.3': 'glm-5.3', 'glm5.3flash': 'glm-5.3-flash',
               'glm5.3flashx': 'glm-5.3-flashx', 'glm-5.3-flash-x': 'glm-5.3-flashx'}
    return aliases.get(value, value)


def identity_match(requested: str, returned: Any) -> bool:
    return bool(normalized_model(returned)) and normalized_model(requested) == normalized_model(returned)


def valid_usage(body: dict | None) -> bool:
    usage = (body or {}).get('usage') or {}
    values = [usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
    return all(type(v) is int and v >= 0 for v in values) and values[0] + values[1] == values[2]


def valid_tools(calls: Any, tools: list[dict], expected: dict | None = None) -> bool:
    if not isinstance(calls, list) or not calls:
        return False
    schemas = {t['function']['name']: t['function']['parameters'] for t in tools}
    ids, names = set(), set()
    for call in calls:
        if not isinstance(call, dict) or call.get('type') != 'function':
            return False
        cid, function = call.get('id'), call.get('function')
        if not isinstance(cid, str) or not cid or cid in ids or not isinstance(function, dict):
            return False
        ids.add(cid)
        name, raw = function.get('name'), function.get('arguments')
        if name not in schemas or not isinstance(raw, str) or not strict_json(raw, schemas[name]):
            return False
        args = json.loads(raw)
        if expected is not None and (name not in expected or args != expected[name]):
            return False
        names.add(name)
    return expected is None or (names == set(expected) and len(calls) == len(expected))


def stream_tools(events: list[dict]) -> list[dict]:
    """Reassemble OpenAI deltas by call index; malformed fragments cannot pass."""
    assembled: dict[int, dict] = {}
    for event in events:
        for choice in event.get('choices') or []:
            if choice.get('index', 0) != 0:
                continue
            for delta in (choice.get('delta') or {}).get('tool_calls') or []:
                index = delta.get('index')
                if type(index) is not int or index < 0:
                    return []
                row = assembled.setdefault(index, {'id': '', 'type': '', 'function': {'name': '', 'arguments': ''}})
                for key in ('id', 'type'):
                    if key in delta:
                        if not isinstance(delta[key], str) or (row[key] and row[key] != delta[key]):
                            return []
                        row[key] = delta[key]
                function = delta.get('function') or {}
                for key in ('name', 'arguments'):
                    if key in function:
                        if not isinstance(function[key], str):
                            return []
                        row['function'][key] += function[key]
    return [assembled[i] for i in sorted(assembled)]


def exact_answer(raw: str, expected: tuple[str, ...]) -> bool:
    value = raw.strip().lower().strip('。.!！')
    colors = {'黄色': '黄', 'yellow': '黄', '蓝色': '蓝', 'blue': '蓝', '绿色': '绿', 'green': '绿'}
    value = colors.get(value, value)
    return any(value == colors.get(token.lower(), token.lower()) for token in expected)


def integer_answer(raw: str, expected: int) -> bool:
    return isinstance(raw, str) and bool(re.fullmatch(r'[+-]?\d+', raw.strip())) and int(raw.strip()) == expected


def arithmetic_function(source: str, name: str, samples: list[tuple[tuple[int, ...], int]]) -> bool:
    """Interpret a tiny arithmetic-function subset with a shared step budget.

    Used for add/Fibonacci completions. No eval/exec, imports, globals or I/O.
    """
    if not isinstance(source, str) or len(source) > 4000: return False
    try:
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef): return False
        fn = tree.body[0]
        if (fn.name != name or fn.decorator_list or fn.args.defaults or fn.args.kw_defaults
                or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs
                or fn.returns or any(a.annotation for a in fn.args.args)): return False
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.If,
                   ast.IfExp, ast.Compare, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant,
                   ast.Call, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.USub, ast.Lt,
                   ast.LtE, ast.Eq, ast.Gt, ast.GtE)
        nodes = list(ast.walk(tree))
        if len(nodes) > 128 or any(not isinstance(n, allowed) for n in nodes): return False
        remaining = 10000
        def expr(node, env, depth):
            nonlocal remaining
            remaining -= 1
            if remaining < 0 or depth > 32: raise ValueError('budget')
            if isinstance(node, ast.Constant) and type(node.value) is int: value = node.value
            elif isinstance(node, ast.Name): value = env[node.id]
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): value = -expr(node.operand, env, depth)
            elif isinstance(node, ast.BinOp):
                op = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}[type(node.op)]
                value = op(expr(node.left, env, depth), expr(node.right, env, depth))
            elif isinstance(node, ast.Compare) and len(node.ops) == 1:
                op = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Eq: operator.eq,
                      ast.Gt: operator.gt, ast.GtE: operator.ge}[type(node.ops[0])]
                value = op(expr(node.left, env, depth), expr(node.comparators[0], env, depth))
            elif isinstance(node, ast.IfExp):
                value = expr(node.body if expr(node.test, env, depth) else node.orelse, env, depth)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name and not node.keywords:
                value = call(tuple(expr(arg, env, depth) for arg in node.args), depth+1)
            else: raise ValueError('unsupported expression')
            if type(value) not in (int, bool) or abs(value).bit_length() > 128: raise ValueError('value budget')
            return value
        def block(statements, env, depth):
            for stmt in statements:
                if isinstance(stmt, ast.Return): return True, expr(stmt.value, env, depth)
                if not isinstance(stmt, ast.If): raise ValueError('unsupported statement')
                found, value = block(stmt.body if expr(stmt.test, env, depth) else stmt.orelse, env, depth)
                if found: return True, value
            return False, None
        def call(args, depth=0):
            if depth > 32 or len(args) != len(fn.args.args): raise ValueError('call budget')
            found, value = block(fn.body, dict(zip((a.arg for a in fn.args.args), args)), depth)
            if not found: raise ValueError('missing return')
            return value
        return bool(samples) and all(type(value := call(args)) is int and value == expected for args, expected in samples)
    except (SyntaxError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
        return False
