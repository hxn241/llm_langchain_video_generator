"""Evaluation: check with test cases that a system answers well.

    cases (data/eval_*.json, written by hand)
        -> system(case)          runs the product with the case's input
        -> checks                each one says ok / fail / (None = could not check)
        -> report()              summary per check + every result in SQLite
                                 (.state/evaluations.sqlite, one row per case and check,
                                  so runs can be compared with SQL)

Kinds of check:
- Deterministic (no LLM): "did it classify 'complaint'?", "did it use tool X?".
  Cheap and exact: use them whenever you can.
- Judge with a REFERENCE: compares with an expected answer.
- Judge with a CRITERION: there is no single answer; it judges whether a rule
  is met ("it is polite and addresses the specific problem").
- Judge with a RUBRIC: scores several criteria from 1 to 5 (long texts).

The judge should be a model AS CAPABLE AS OR MORE than the evaluated one: a
small one may reason "it is wrong" and still mark it correct. It is set
separately with JUDGE_MODEL_NAME in the .env (config.create_judge_model()).

Case fields used here: "input" (always), "expected_answer", "criterion".
"""
import json
from contextlib import closing
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llmkit import config, storage


# ------------------------------------------------------------------- judges
class Verdict(BaseModel):
    # 'reason' goes BEFORE 'correct': the model reasons first and decides afterwards
    reason: str = Field(description="Una frase explicando la decisión.")
    correct: bool = Field(description="True si cumple; False si no.")


REFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Eres un profesor corrigiendo. Compara la respuesta del alumno con la "
               "correcta. Ignora estilo y longitud: solo importa si el CONTENIDO es "
               "correcto y no contradice a la respuesta correcta."),
    ("human", "Pregunta: {question}\nRespuesta correcta: {expected}\nRespuesta del alumno: {actual}"),
])

CRITERION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Eres un revisor de calidad. Decide si la salida cumple el criterio."),
    ("human", "Entrada: {input}\n\nSalida: {output}\n\nCriterio: {criterion}"),
])


class CriterionScore(BaseModel):
    # FLAT schema and one criterion per call: more reliable than asking for a
    # list of objects at once (small models return it empty).
    reason: str = Field(description="Por qué esta puntuación, en una frase.")
    score: int = Field(ge=1, le=5, description="1 = muy mal, 5 = excelente.")


RUBRIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Eres un evaluador exigente. Puntúa el texto de 1 a 5 SOLO en el criterio indicado."),
    ("human", "Tarea que se pidió: {input}\n\nTexto:\n{output}\n\nCriterio: {criterion}"),
])


# Each judge accepts its own prompt (another language, a stricter judge...)
# as long as it keeps the same variables.

def reference_judge(model, prompt=REFERENCE_PROMPT):
    """(question, expected, actual) -> Verdict"""
    return prompt | model.with_structured_output(Verdict)


def criterion_judge(model, prompt=CRITERION_PROMPT):
    """(input, output, criterion) -> Verdict"""
    return prompt | model.with_structured_output(Verdict)


def rubric_judge(model, prompt=RUBRIC_PROMPT):
    """(input, output, criterion) -> CriterionScore  (called once per criterion)"""
    return prompt | model.with_structured_output(CriterionScore)


# ------------------------------------------------------------------- checks
# A check is a function (case, result) -> (ok, reason), where ok is
# True / False, or None if it does not apply to that case.

def check_with_reference(judge, output_field="answer"):
    def check(case, result):
        if "expected_answer" not in case:
            return None, "no expected answer"
        v = judge.invoke({"question": case["input"], "expected": case["expected_answer"],
                          "actual": result[output_field]})
        return v.correct, v.reason
    check.__name__ = "reference_judge"
    return check


def check_with_criterion(judge, output_field="answer"):
    def check(case, result):
        if "criterion" not in case:
            return None, "no criterion"
        v = judge.invoke({"input": case["input"], "output": result[output_field],
                          "criterion": case["criterion"]})
        return v.correct, v.reason
    check.__name__ = "criterion_judge"
    return check


def check_with_rubric(judge, criteria, pass_mark=3.5, output_field="answer"):
    def check(case, result):
        scores = {
            criterion: judge.invoke({"input": case["input"], "output": result[output_field],
                                     "criterion": criterion}).score
            for criterion in criteria
        }
        mean = sum(scores.values()) / len(scores)
        detail = ", ".join(f"{c}={s}" for c, s in scores.items())
        return mean >= pass_mark, f"mean {mean:.1f} ({detail})"
    check.__name__ = "rubric_judge"
    return check


# ---------------------------------------------------------------------- run
def load_cases(file_name):
    return json.loads((config.DATA_DIR / file_name).read_text(encoding="utf-8"))


def evaluate(cases, system, checks, verbose=True):
    """Runs every case through the system and through every check.

    An error (e.g. a 429 from a free model) does not stop the evaluation: that
    case is left as None ("not evaluated") so the score is not distorted.
    """
    rows = []
    for i, case in enumerate(cases, start=1):
        row = {"case": case, "result": None, "checks": {}}
        try:
            row["result"] = system(case)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        for check in checks:
            name = check.__name__
            if row["result"] is None:
                row["checks"][name] = (None, "the system failed")
                continue
            try:
                row["checks"][name] = check(case, row["result"])
            except Exception as e:
                row["checks"][name] = (None, f"ERROR {type(e).__name__}: {e}"[:200])
        if verbose:
            icons = " ".join({True: "✔", False: "✘", None: "·"}[ok] for ok, _ in row["checks"].values())
            print(f"  [{i}/{len(cases)}] {icons}  {case['input'][:70]}")
        rows.append(row)
    return rows


RESULTS_DB = "evaluations"              # -> .state/evaluations.sqlite


def open_results():
    """The database of evaluation results: one row per case and check."""
    db = storage.connect(RESULTS_DB)
    db.execute("""
        CREATE TABLE IF NOT EXISTS checks (
            run         TEXT NOT NULL,      -- "<name>_<date>": one evaluation run
            name        TEXT NOT NULL,      -- what was evaluated (e.g. write_script)
            created_at  TEXT NOT NULL,
            case_number INTEGER NOT NULL,
            input       TEXT,
            answer      TEXT,
            error       TEXT,               -- the system failed on this case
            check_name  TEXT,
            ok          INTEGER,            -- 1 correct, 0 wrong, NULL not evaluated
            reason      TEXT
        )
    """)
    return db


def save_results(rows, name, created_at=None):
    """Saves an evaluation's rows and returns the run id. Compare runs with SQL, e.g.
    SELECT run, check_name, AVG(ok) FROM checks WHERE name = ? GROUP BY run, check_name"""
    created_at = created_at or datetime.now()
    run = f"{name}_{created_at:%Y%m%d_%H%M%S}"
    with closing(open_results()) as db:
        for number, r in enumerate(rows, start=1):
            result = r.get("result")
            answer = result.get("answer", result) if isinstance(result, dict) else result
            base = {"run": run, "name": name, "created_at": created_at.isoformat(timespec="seconds"),
                    "case_number": number, "input": str(r["case"].get("input")),
                    "answer": None if answer is None else str(answer), "error": r.get("error")}
            for check_name, (ok, reason) in (r["checks"] or {None: (None, None)}).items():
                storage.insert(db, "checks", {**base, "check_name": check_name,
                                              "ok": None if ok is None else int(ok), "reason": reason})
    return run


def report(rows, name):
    """Prints the summary, saves every result to .state/evaluations.sqlite and returns the run id."""
    print(f"\n===== REPORT: {name} =====")
    names = list(rows[0]["checks"]) if rows else []
    for n in names:
        values = [r["checks"][n][0] for r in rows]
        evaluated = [v for v in values if v is not None]
        print(f"  {n:<22} {sum(evaluated)}/{len(evaluated)} correct  (not evaluated: {len(values) - len(evaluated)})")

    for i, r in enumerate(rows, start=1):
        failures = {n: reason for n, (ok, reason) in r["checks"].items() if ok is False}
        if failures or r.get("error"):
            print(f"\n  ✘ Case {i}: {r['case']['input'][:80]}")
            if r.get("error"):
                print(f"     error: {r['error']}")
            for n, reason in failures.items():
                print(f"     {n}: {reason}")

    run = save_results(rows, name)
    print(f"\n  Details saved to {config.readable_path(storage.path(RESULTS_DB))} (run {run})")
    return run
