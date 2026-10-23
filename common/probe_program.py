"""Small bounded AST interpreter for generated code probes. Never eval/exec Python.

Only the documented loop/list subset is supported; no imports, attributes other
than list append/count, I/O, recursion, or access to Python globals.
"""
from __future__ import annotations

import ast
import operator


class ProgramRejected(ValueError):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class Program:
    def __init__(self, source: str):
        if len(source) > 12000:
            raise ProgramRejected('source too long')
        try:
            tree = ast.parse(source)
        except (SyntaxError, RecursionError) as exc:
            raise ProgramRejected('invalid syntax') from exc
        if len(list(ast.walk(tree))) > 500 or len(tree.body) != 1:
            raise ProgramRejected('one bounded function required')
        fn = tree.body[0]
        if (not isinstance(fn, ast.FunctionDef) or fn.name != 'solve' or fn.decorator_list
                or len(fn.args.args) != 1 or fn.args.vararg or fn.args.kwarg or fn.args.defaults
                or fn.args.kwonlyargs or fn.args.posonlyargs):
            raise ProgramRejected('expected solve(nums)')
        self.body, self.arg = fn.body, fn.args.args[0].arg

    def run(self, values: list[int]):
        self.steps = 0
        self.env = {self.arg: list(values)}
        try:
            self.block(self.body)
        except _Return as result:
            if self.env.get(self.arg) != values:
                raise ProgramRejected("input mutation is forbidden")
            return result.value
        return None

    def tick(self):
        self.steps += 1
        if self.steps > 10000:
            raise ProgramRejected('step budget exceeded')

    def bound(self, value):
        if isinstance(value, int) and value.bit_length() > 256:
            raise ProgramRejected('integer budget exceeded')
        if isinstance(value, (str, list, tuple, set, dict)) and len(value) > 2048:
            raise ProgramRejected('collection budget exceeded')
        return value

    def assign(self, target, value):
        if not isinstance(target, ast.Name) or target.id.startswith('_'):
            raise ProgramRejected('assignment target not allowed')
        self.env[target.id] = self.bound(value)

    def block(self, statements):
        for node in statements:
            self.tick()
            if isinstance(node, ast.Return):
                raise _Return(self.expr(node.value))
            if isinstance(node, ast.Assign):
                value = self.expr(node.value)
                for target in node.targets:
                    self.assign(target, value)
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                self.assign(node.target, self.bound(self.expr(node.target) + self.expr(node.value)))
            elif isinstance(node, ast.For):
                values = self.expr(node.iter)
                if not isinstance(values, (list, tuple, set, range)) or len(values) > 2048:
                    raise ProgramRejected('invalid iterator')
                for value in list(values):
                    self.tick()
                    self.assign(node.target, value)
                    self.block(node.body)
                self.block(node.orelse)
            elif isinstance(node, ast.If):
                self.block(node.body if self.expr(node.test) else node.orelse)
            elif isinstance(node, ast.Expr):
                self.expr(node.value)
            elif isinstance(node, ast.Pass):
                pass
            else:
                raise ProgramRejected(f'unsupported statement: {type(node).__name__}')

    def expr(self, node):
        self.tick()
        if isinstance(node, ast.Constant) and type(node.value) in (int, str, bool, type(None)):
            return self.bound(node.value)
        if isinstance(node, ast.Name) and node.id in self.env:
            return self.env[node.id]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self.expr(x) for x in node.elts]
            return self.bound(set(values) if isinstance(node, ast.Set) else tuple(values) if isinstance(node, ast.Tuple) else values)
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand)
            if isinstance(node.op, ast.Not): return not value
            if isinstance(node.op, ast.USub) and type(value) is int: return -value
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left), self.expr(node.right)
            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mod: operator.mod,
                   ast.FloorDiv: operator.floordiv}
            if type(node.op) in ops and type(left) is int and type(right) is int:
                return self.bound(ops[type(node.op)](left, right))
        if isinstance(node, ast.BoolOp):
            for part in node.values:
                value = self.expr(part)
                if isinstance(node.op, ast.And) and not value: return value
                if isinstance(node.op, ast.Or) and value: return value
            return value
        if isinstance(node, ast.Compare):
            left = self.expr(node.left)
            ops = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                   ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                   ast.In: lambda a,b: a in b, ast.NotIn: lambda a,b: a not in b}
            for op, operand in zip(node.ops, node.comparators):
                right = self.expr(operand)
                if type(op) not in ops: raise ProgramRejected('comparison not allowed')
                if not ops[type(op)](left, right): return False
                left = right
            return True
        if isinstance(node, ast.Call) and not node.keywords:
            args = [self.expr(a) for a in node.args]
            if isinstance(node.func, ast.Name):
                funcs = {'sorted': sorted, 'len': len, 'set': set, 'list': list, 'abs': abs}
                if node.func.id in funcs:
                    return self.bound(funcs[node.func.id](*args))
            if isinstance(node.func, ast.Attribute) and node.func.attr in {'append', 'count'}:
                owner = self.expr(node.func.value)
                if isinstance(owner, list) and len(args) == 1:
                    if node.func.attr == 'count': return owner.count(args[0])
                    if len(owner) >= 2048: raise ProgramRejected('collection budget exceeded')
                    owner.append(args[0]); return None
        raise ProgramRejected(f'unsupported expression: {type(node).__name__}')
