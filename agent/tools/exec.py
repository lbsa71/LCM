"""Restricted Python AST interpreter for safe computation (PRD Section 25.3)."""

import ast
import operator
from typing import Any, Dict, Optional


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}

SAFE_BUILTINS = {
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sorted": sorted,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "range": range,
    "zip": zip,
    "all": all,
    "any": any,
}


class RestrictedASTEvaluator:
    """Safely evaluates a constrained subset of Python expressions without eval/exec."""

    def __init__(self, max_operations: int = 10000):
        self.max_operations = max_operations
        self.op_count = 0

    def evaluate(self, code_str: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Parses and evaluates a safe Python expression."""
        self.op_count = 0
        context = dict(SAFE_BUILTINS)
        if inputs:
            for k, v in inputs.items():
                if not k.startswith("_"):
                    context[k] = v

        try:
            tree = ast.parse(code_str.strip(), mode="eval")
        except Exception as e:
            # Check if it's a simple statement/number
            try:
                tree = ast.parse(code_str.strip(), mode="exec")
                if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                    tree = ast.Expression(body=tree.body[0].value)
                else:
                    return {
                        "status": "error",
                        "error_type": "SYNTAX_ERROR",
                        "message": f"Code must be a single evaluatable expression: {str(e)}"
                    }
            except Exception as e2:
                return {
                    "status": "error",
                    "error_type": "SYNTAX_ERROR",
                    "message": f"Failed to parse expression: {str(e2)}"
                }

        try:
            result = self._eval_node(tree.body, context)
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": "RUNTIME_ERROR",
                "message": f"Execution error: {str(e)}"
            }

    def _eval_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        self.op_count += 1
        if self.op_count > self.max_operations:
            raise RuntimeError("Operation limit exceeded in sandboxed code.")

        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            raise NameError(f"Name '{node.id}' is not defined or permitted.")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            op_func = ALLOWED_OPERATORS.get(type(node.op))
            if not op_func:
                raise ValueError(f"Operator {type(node.op).__name__} is not permitted.")
            return op_func(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context)
            op_func = ALLOWED_OPERATORS.get(type(node.op))
            if not op_func:
                raise ValueError(f"Unary operator {type(node.op).__name__} is not permitted.")
            return op_func(operand)

        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context)
                op_func = ALLOWED_OPERATORS.get(type(op))
                if not op_func or not op_func(left, right):
                    return False
                left = right
            return True

        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for val in node.values:
                    res = self._eval_node(val, context)
                    if not res:
                        return res
                return res
            elif isinstance(node.op, ast.Or):
                for val in node.values:
                    res = self._eval_node(val, context)
                    if res:
                        return res
                return res

        elif isinstance(node, ast.List):
            return [self._eval_node(elem, context) for elem in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elem, context) for elem in node.elts)

        elif isinstance(node, ast.Dict):
            return {
                self._eval_node(k, context): self._eval_node(v, context)
                for k, v in zip(node.keys, node.values)
            }

        elif isinstance(node, ast.Subscript):
            target = self._eval_node(node.value, context)
            slice_val = self._eval_node(node.slice, context)
            return target[slice_val]

        elif isinstance(node, ast.Slice):
            lower = self._eval_node(node.lower, context) if node.lower else None
            upper = self._eval_node(node.upper, context) if node.upper else None
            step = self._eval_node(node.step, context) if node.step else None
            return slice(lower, upper, step)

        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func, context)
            if func not in SAFE_BUILTINS.values() and not callable(func):
                raise PermissionError("Calling unauthorized function or method.")
            args = [self._eval_node(arg, context) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value, context) for kw in node.keywords}
            return func(*args, **kwargs)

        elif isinstance(node, ast.ListComp):
            # Safe single-generator list comprehension
            if len(node.generators) != 1:
                raise ValueError("Only single-generator list comprehensions are permitted.")
            gen = node.generators[0]
            if not isinstance(gen.target, ast.Name):
                raise ValueError("Target in list comprehension must be a simple variable name.")
            iter_val = self._eval_node(gen.iter, context)
            res = []
            var_name = gen.target.id
            for item in iter_val:
                local_ctx = dict(context)
                local_ctx[var_name] = item
                # Check ifs
                passes = True
                for if_clause in gen.ifs:
                    if not self._eval_node(if_clause, local_ctx):
                        passes = False
                        break
                if passes:
                    res.append(self._eval_node(node.elt, local_ctx))
            return res

        else:
            raise PermissionError(f"AST node '{type(node).__name__}' is strictly prohibited in sandbox.")
