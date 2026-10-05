"""Tools for agents.

A tool is a function decorated with @tool. The LLM does NOT run it: it reads
its name, docstring and arguments, and ASKS to use it; your code runs it. That
is why the docstring is written for the model: what it does and when to use it
(in Spanish here, like the rest of the text the model reads).

Only general-purpose tools live here. Tools of one application (upload a
video, publish a post...) go in that application, and so does the list of
the ones that need human approval (LLMHub.agent(..., approval={...})).
"""
import ast
import operator
from datetime import date

import requests
from langchain_core.tools import tool

from llmkit import config

# --------------------------------------------------------------- calculator
_OPERATIONS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
# Brakes against expressions that would hang the process (e.g. 9**9**9**9)
MAX_EXPONENT = 1000
MAX_RESULT_BITS = 10_000          # ~3000 digits


def calculate(expression: str):
    """Safe calculator: only numbers, + - * / % ** and parentheses.

    NEVER eval(): it would run any code the LLM writes. Here the expression
    tree is walked and anything that is not arithmetic is rejected, as is any
    power or result too big to compute in reasonable time.
    """
    def evaluate(node):
        if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATIONS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > MAX_EXPONENT:
                    raise ValueError(f"Exponent too big (max {MAX_EXPONENT}): {expression!r}")
                if isinstance(left, int) and isinstance(right, int) and left.bit_length() * right > MAX_RESULT_BITS:
                    raise ValueError(f"Result too big: {expression!r}")   # checked BEFORE computing it
            result = _OPERATIONS[type(node.op)](left, right)
            if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
                raise ValueError(f"Result too big: {expression!r}")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATIONS:
            return _OPERATIONS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"Expression not allowed: {expression!r}")

    return evaluate(ast.parse(expression.strip(), mode="eval").body)


@tool
def calculator(expression: str) -> str:
    """Hace cálculos aritméticos: + - * / % ** y paréntesis. Ej.: '(45 + 6) * 0.21'.
    Úsala siempre para operaciones con números en vez de calcular de cabeza."""
    return str(calculate(expression))


# ------------------------------------------------------------------- others
@tool
def today_date() -> str:
    """Devuelve la fecha de hoy (AAAA-MM-DD). Úsala en cualquier pregunta que
    dependa de la fecha actual: plazos, días que faltan, 'hoy', 'mañana'."""
    return date.today().isoformat()


@tool
def wikipedia(query: str, language: str = "es") -> str:
    """Busca en Wikipedia y devuelve el resumen de los artículos más relevantes.
    Para datos enciclopédicos: personas, lugares, historia, ciencia, libros.
    language: 'es' (por defecto) o 'en' (tiene muchos más artículos)."""
    r = requests.get(
        f"https://{language}.wikipedia.org/w/api.php",
        headers={"User-Agent": config.USER_AGENT},  # Wikipedia requires it
        timeout=15,
        params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrlimit": 3,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "exchars": 1200,
        },
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    if not pages:
        return f"No hay resultados en Wikipedia ({language}) para: {query}"
    ordered = sorted(pages.values(), key=lambda p: p["index"])
    return "\n\n".join(f"Página: {p['title']}\nResumen: {p.get('extract', '')}" for p in ordered)
