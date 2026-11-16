"""
FinQA 평가 데이터 가공 스크립트 (Adaptive FinQA Agent, 9조)

하는 일
  1. FinQA 원본(test/dev/train.json)을 불러와 데이터 형식을 검증한다.
  2. 문항마다 라벨(단일/복합, 정답 형식, 퍼센트 여부 등)을 붙인다.
  3. 모델 입력(질문·본문·표)을 원본 ID를 보존한 채로 정리한다.
  4. 평가 세트(단일 40 + 복합 40 + 정보 부족 20)와 개발 세트(30)를 고정 시드로 뽑는다.
  5. 질문 파일과 정답 파일을 분리해 저장하고, manifest.json에 해시를 기록한다.
  6. 정답 누출, 개수, 재현성을 검증한다.

사용법 (Colab)
  !git clone https://github.com/czyssrs/FinQA.git
  !python scripts/prepare_data.py --finqa_dir FinQA --out_dir data_processed

규칙
  - LLM을 호출하지 않는다. 모든 처리는 규칙 기반이다.
  - 원본 json은 수정하지 않는다.
  - 같은 설정이면 출력 파일의 SHA-256이 항상 같다.
"""

import argparse
import copy
import csv
import datetime
import hashlib
import io
import json
import os
import random
import re
import subprocess
from collections import Counter

# ─────────────────────────────────────────────
# 설정값 (팀이 바꿀 수 있음. 바꾸면 평가 세트가 바뀌므로 확정 후에는 고정)
# ─────────────────────────────────────────────
SEED = 42
N_SINGLE = 40      # test에서 뽑을 단일 연산 문항 수
N_MULTI = 40       # test에서 뽑을 복합 연산 문항 수
N_UNANS = 20       # test에서 만들 정보 부족 문항 수 (원래 단일/복합 반반)
N_DEV = 30         # dev에서 뽑을 개발용 문항 수 (단일/복합 반반)
N_FEWSHOT = 0      # 프롬프트 예시 수. train에서만 뽑는다 (0이면 생략)

CONFIG = dict(SEED=SEED, N_SINGLE=N_SINGLE, N_MULTI=N_MULTI, N_UNANS=N_UNANS,
              N_DEV=N_DEV, N_FEWSHOT=N_FEWSHOT)

# 원본 데이터에 대해 미리 확인한 사실 (test.json 기준). 다르면 멈춘다.
EXPECTED_TEST = dict(n=1147, yes=13, no=7, dash_cells=322, dot_sents=2565)

TOP_KEYS = {"id", "filename", "pre_text", "post_text", "table", "table_ori", "qa",
            "table_retrieved", "text_retrieved", "table_retrieved_all", "text_retrieved_all"}
QA_KEYS = {"question", "answer", "explanation", "ann_table_rows", "ann_text_rows", "steps",
           "program", "gold_inds", "exe_ans", "tfidftopn", "program_re", "model_input"}
QUESTION_KEYS = {"qid", "question", "text", "table"}

PUNCT_ONLY = re.compile(r"^[\s\.,;:\-]*$")          # 구두점만 있는 문장
CHAR_FIXES = [                                       # 깨진 특수문자 복원 (본문)
    (re.compile(r"(?<![0-9a-z])2019(?=[st]\b)"), "’"),   # 2019s → ’s
    (re.compile(r"(?<![0-9a-z])201c"), "“"),             # 201c → “
    (re.compile(r"(?<![0-9a-z])201d"), "”"),             # 201d → ”
]
DASH_CELLS = {"2014", "$ 2014"}                      # 표에서 "—"가 깨진 값
NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def rng(tag: str) -> random.Random:
    """단계마다 독립된 고정 난수열 (문자열 시드는 파이썬 버전과 무관하게 결정적)."""
    return random.Random(f"{SEED}-{tag}")


def sample_sorted(items, k, tag):
    items = sorted(items, key=lambda e: e["id"])
    if len(items) < k:
        raise ValueError(f"[{tag}] 후보 {len(items)}개 < 필요 {k}개")
    return rng(tag).sample(items, k)


def to_jsonl(rows) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                   for r in rows).encode("utf-8")


def norm_num(s: str):
    """'$ 5,735' / '( 32 )' / '14.5%' → 절댓값 float. 숫자가 아니면 None."""
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return round(abs(float(s)), 6)
    except ValueError:
        return None


def label_consistent(e) -> bool:
    """사람이 쓴 답(qa.answer)과 정답값(exe_ans)이 서로 맞는지 확인한다.
    FinQA에는 두 값이 어긋나는 라벨 오류가 있다(예: answer '10%', exe_ans 52336).
    퍼센트 표기 차이(×100, ÷100)와 반올림은 허용한다.
    answer가 비었거나 yes/no 문항이면 확인할 수 없으므로 통과로 본다."""
    q = e["qa"]
    if isinstance(q["exe_ans"], str) or not q["answer"].strip():
        return True
    m = re.search(r"-?\d[\d,]*\.?\d*", q["answer"].replace("$", ""))
    if not m:
        return True
    t = float(m.group().replace(",", ""))
    dec = len(m.group().split(".")[1]) if "." in m.group() else 0
    tol = max(0.05 * abs(t), 0.5 * 10 ** (-dec))
    x = float(q["exe_ans"])
    return any(abs(abs(t) - abs(c)) <= tol for c in (x, x * 100, x / 100))


def nums_in(s: str):
    out = set()
    for m in NUM_RE.findall(s):
        v = norm_num(m)
        if v is not None:
            out.add(v)
    return out


# ─────────────────────────────────────────────
# 1. 불러오기와 검증
# ─────────────────────────────────────────────
def load_split(finqa_dir, name):
    path = os.path.join(finqa_dir, "dataset", f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def verify_split(data, name):
    for e in data:
        missing = TOP_KEYS - set(e)
        if missing:
            raise SystemExit(f"[{name}] {e.get('id')} 최상위 키 누락: {missing}")
        if QA_KEYS - set(e["qa"]):
            raise SystemExit(f"[{name}] {e['id']} qa 키 누락: {QA_KEYS - set(e['qa'])}")
        if not e["table"] or len({len(r) for r in e["table"]}) != 1:
            raise SystemExit(f"[{name}] {e['id']} 표 형식 이상")
        ea = e["qa"]["exe_ans"]
        if isinstance(ea, str) and ea not in ("yes", "no"):
            raise SystemExit(f"[{name}] {e['id']} 예상 못한 exe_ans: {ea!r}")
        n_text = len(e["pre_text"]) + len(e["post_text"])
        for k in e["qa"]["gold_inds"]:
            kind, idx = k.split("_")
            idx = int(idx)
            if kind == "table" and not (0 <= idx < len(e["table"])):
                raise SystemExit(f"[{name}] {e['id']} gold {k} 범위 밖")
            if kind == "text" and not (0 <= idx < n_text):
                raise SystemExit(f"[{name}] {e['id']} gold {k} 범위 밖")

    ans = Counter(e["qa"]["exe_ans"] if isinstance(e["qa"]["exe_ans"], str) else "number"
                  for e in data)
    steps = Counter(min(len(e["qa"]["steps"]), 3) for e in data)
    dash = sum(c.strip() in DASH_CELLS for e in data for r in e["table"][1:] for c in r[1:])
    dots = sum(s.strip() == "." for e in data for s in e["pre_text"] + e["post_text"])
    print(f"[{name}] 문항 {len(data)} | 정답 형식 {dict(ans)} | 단계 수(3=3+) {dict(sorted(steps.items()))}"
          f" | 표 '2014' 칸 {dash} | '.' 문장 {dots}")

    if name == "test":
        got = dict(n=len(data), yes=ans["yes"], no=ans["no"], dash_cells=dash, dot_sents=dots)
        diff = {k: (EXPECTED_TEST[k], got[k]) for k in EXPECTED_TEST if EXPECTED_TEST[k] != got[k]}
        if diff:
            raise SystemExit(f"[test] 데이터 사실이 다릅니다 (예상, 실제): {diff}")


# ─────────────────────────────────────────────
# 2. 라벨
# ─────────────────────────────────────────────
def labels(e):
    q = e["qa"]
    n_steps = len(q["steps"])
    kinds = {k.split("_")[0] for k in q["gold_inds"]}
    return dict(
        type="single" if n_steps == 1 else "multi",
        answer_type="yes_no" if isinstance(q["exe_ans"], str) else "number",
        exe_ans=q["exe_ans"],
        is_percent=q["answer"].strip().endswith("%"),
        answer_text=q["answer"],
        program=q["program"],
        n_steps=n_steps,
        gold_inds=sorted(q["gold_inds"]),
        evidence_loc="both" if kinds == {"table", "text"} else kinds.pop(),
        uses_const="const_" in q["program"],
    )


# ─────────────────────────────────────────────
# 3. 모델 입력 (원본 ID 보존)
# ─────────────────────────────────────────────
def program_uses_2014(e):
    return any("2014" in (s["arg1"], s["arg2"]) for s in e["qa"]["steps"])


def build_input(e, stats):
    text = []
    sents = [("pre", s) for s in e["pre_text"]] + [("post", s) for s in e["post_text"]]
    for i, (part, s) in enumerate(sents):
        if PUNCT_ONLY.match(s):
            stats["punct_removed"] += 1
            continue
        for pat, rep in CHAR_FIXES:
            s = pat.sub(rep, s)
        text.append({"id": f"text_{i}", "part": part, "sent": s})

    # 표: 모든 행에 table_0부터 원본 번호를 붙인다.
    # table[0]은 보통 헤더지만, 헤더 없이 첫 행부터 데이터인 표도 있고
    # table_0이 정답 근거인 문항도 있다(test 13, dev 4). 그래서 행을 나누지 않는다.
    keep_2014 = program_uses_2014(e)
    if keep_2014:
        stats["dash_exception_ids"].add(e["id"])
    rows = []
    for n, r in enumerate(e["table"]):
        cells = list(r)
        if not keep_2014:
            for j in range(1, len(cells)):
                c = cells[j].strip()
                # 0행의 "2014"는 대부분 실제 연도(열 이름)라서 "$ 2014"만 바꾼다
                if (n >= 1 and c in DASH_CELLS) or (n == 0 and c == "$ 2014"):
                    cells[j] = "—"
                    stats["dash_replaced"] += 1
        rows.append({"id": f"table_{n}", "cells": cells})
    return {"question": e["qa"]["question"], "text": text, "table": rows}


def input_ids(inp):
    return {t["id"] for t in inp["text"]} | {r["id"] for r in inp["table"]}


# ─────────────────────────────────────────────
# 5. 정보 부족 문항
# ─────────────────────────────────────────────
def remove_evidence(inp, gold):
    out = copy.deepcopy(inp)
    out["text"] = [t for t in out["text"] if t["id"] not in gold]
    out["table"] = [r for r in out["table"] if r["id"] not in gold]
    return out


def unans_reason(e, inp):
    """후보 탈락 사유. 통과하면 None."""
    gold = set(e["qa"]["gold_inds"])
    rest = remove_evidence(inp, gold)
    if not rest["text"] and len(rest["table"]) <= 1:
        return "c_empty"
    remain = set()
    for t in rest["text"]:
        remain |= nums_in(t["sent"])
    for r in rest["table"]:
        for c in r["cells"]:
            remain |= nums_in(c)
    args = set()
    for s in e["qa"]["steps"]:
        for a in (s["arg1"], s["arg2"]):
            if a.startswith("#") or a.startswith("const_"):
                continue
            v = norm_num(a)
            if v is not None:
                args.add(v)
    if args and args <= remain:
        return "b_numbers_remain"
    return None


def table_summary(table):
    return "\n".join(f"[{r['id']}] {r['cells'][0]}: " + " | ".join(r["cells"][1:]) for r in table)


# ─────────────────────────────────────────────
# 전체 가공 (파일 바이트를 메모리에 만든다 → 재현성 비교 가능)
# ─────────────────────────────────────────────
def build_all(splits):
    test, dev, train = splits["test"], splits["dev"], splits.get("train")
    stats = {"punct_removed": 0, "dash_replaced": 0, "dash_exception_ids": set()}

    test_in = {e["id"]: build_input(e, stats) for e in test}
    dev_in = {e["id"]: build_input(e, stats) for e in dev}

    # 4. 평가 세트 (정답 라벨이 어긋난 문항은 후보에서 뺀다)
    bad_test = {e["id"] for e in test if not label_consistent(e)}
    bad_dev = {e["id"] for e in dev if not label_consistent(e)}
    singles = [e for e in test if len(e["qa"]["steps"]) == 1 and e["id"] not in bad_test]
    multis = [e for e in test if len(e["qa"]["steps"]) >= 2 and e["id"] not in bad_test]
    eval_src = sample_sorted(singles, N_SINGLE, "eval-single") + \
        sample_sorted(multis, N_MULTI, "eval-multi")
    eval_src_ids = {e["id"] for e in eval_src}

    # 5. 정보 부족
    reasons = Counter()
    cand = {"single": [], "multi": []}
    for e in test:
        if e["id"] in eval_src_ids:
            continue
        r = unans_reason(e, test_in[e["id"]])
        if r:
            reasons[r] += 1
            continue
        cand[labels(e)["type"]].append(e)
    unans_src = sample_sorted(cand["single"], N_UNANS // 2, "unans-single") + \
        sample_sorted(cand["multi"], N_UNANS - N_UNANS // 2, "unans-multi")

    # 6. 개발 세트
    dsingles = [e for e in dev if len(e["qa"]["steps"]) == 1 and e["id"] not in bad_dev]
    dmultis = [e for e in dev if len(e["qa"]["steps"]) >= 2 and e["id"] not in bad_dev]
    dev_src = sample_sorted(dsingles, N_DEV // 2, "dev-single") + \
        sample_sorted(dmultis, N_DEV - N_DEV // 2, "dev-multi")

    # 7. qid 부여 + 질문/정답 분리
    eval_items = [("normal", e) for e in eval_src] + [("unans", e) for e in unans_src]
    eval_items.sort(key=lambda x: (x[0], x[1]["id"]))
    rng("eval-qid").shuffle(eval_items)

    eval_q, eval_a, review = [], [], []
    for i, (kind, e) in enumerate(eval_items, start=1):
        qid = f"E{i:03d}"
        lab = labels(e)
        if kind == "normal":
            eval_q.append({"qid": qid, **test_in[e["id"]]})
            eval_a.append({"qid": qid, "finqa_id": e["id"], **lab, "removed_evidence": []})
        else:
            gold = set(lab["gold_inds"])
            inp = remove_evidence(test_in[e["id"]], gold)
            eval_q.append({"qid": qid, **inp})
            eval_a.append({
                "qid": qid, "finqa_id": e["id"], "type": "unanswerable",
                "answer_type": "unanswerable", "exe_ans": None, "is_percent": False,
                "answer_text": "", "program": None, "n_steps": None, "gold_inds": [],
                "evidence_loc": None, "uses_const": None,
                "removed_evidence": sorted(gold),
                "source": {k: lab[k] for k in
                           ("type", "exe_ans", "program", "n_steps", "gold_inds", "evidence_loc")},
            })
            orig = test_in[e["id"]]
            removed_text = [f"[{t['id']}] {t['sent']}" for t in orig["text"] if t["id"] in gold] + \
                [f"[{r['id']}] " + " | ".join(r["cells"]) for r in orig["table"] if r["id"] in gold]
            review.append({"qid": qid, "finqa_id": e["id"], "question": e["qa"]["question"],
                           "removed_text": "\n".join(removed_text),
                           "remaining_table": table_summary(inp["table"]),
                           "reviewer1": "", "reviewer2": ""})

    dev_items = sorted(dev_src, key=lambda e: e["id"])
    rng("dev-qid").shuffle(dev_items)
    dev_q, dev_a = [], []
    for i, e in enumerate(dev_items, start=1):
        qid = f"D{i:03d}"
        dev_q.append({"qid": qid, **dev_in[e["id"]]})
        dev_a.append({"qid": qid, "finqa_id": e["id"], **labels(e), "removed_evidence": []})

    key = lambda r: r["qid"]
    files = {
        "eval_questions.jsonl": to_jsonl(sorted(eval_q, key=key)),
        "eval_answers.jsonl": to_jsonl(sorted(eval_a, key=key)),
        "dev_questions.jsonl": to_jsonl(sorted(dev_q, key=key)),
        "dev_answers.jsonl": to_jsonl(sorted(dev_a, key=key)),
    }
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=["qid", "finqa_id", "question", "removed_text",
                                        "remaining_table", "reviewer1", "reviewer2"])
    w.writeheader()
    for r in sorted(review, key=key):
        w.writerow(r)
    files["unans_review.csv"] = buf.getvalue().encode("utf-8-sig")

    if N_FEWSHOT > 0 and train:
        fs_in = {"punct_removed": 0, "dash_replaced": 0, "dash_exception_ids": set()}
        fs = sample_sorted(train, N_FEWSHOT, "fewshot")
        files["fewshot.jsonl"] = to_jsonl(
            [{"qid": f"F{i:03d}", **build_input(e, fs_in),
              "exe_ans": e["qa"]["exe_ans"], "program": e["qa"]["program"]}
             for i, e in enumerate(sorted(fs, key=lambda e: e["id"]), start=1)])

    meta = {
        "id_map": {"eval": {a["qid"]: a["finqa_id"] for a in eval_a},
                   "dev": {a["qid"]: a["finqa_id"] for a in dev_a}},
        "cleaning_stats": {
            "label_mismatch_excluded": {"test": len(bad_test), "dev": len(bad_dev)},
            "punct_sentences_removed": stats["punct_removed"],
            "dash_cells_replaced": stats["dash_replaced"],
            "dash_exception_ids": sorted(stats["dash_exception_ids"]),
            "unans_excluded_b_numbers_remain": reasons["b_numbers_remain"],
            "unans_excluded_c_empty": reasons["c_empty"],
            "unans_candidates": {k: len(v) for k, v in cand.items()},
        },
    }
    parsed = dict(eval_q=eval_q, eval_a=eval_a, dev_q=dev_q, dev_a=dev_a,
                  eval_src_ids=eval_src_ids, unans_src_ids={e["id"] for e in unans_src},
                  test_in=test_in)
    return files, meta, parsed


# ─────────────────────────────────────────────
# 8. 검증
# ─────────────────────────────────────────────
def validate(files, parsed, splits):
    eval_q, eval_a, dev_q, dev_a = (parsed[k] for k in ("eval_q", "eval_a", "dev_q", "dev_a"))

    # 질문 파일 키와 누출 문자열
    finqa_ids = {a["finqa_id"] for a in eval_a + dev_a}
    for fname in ("eval_questions.jsonl", "dev_questions.jsonl"):
        raw = files[fname].decode("utf-8")
        for line in raw.splitlines():
            assert set(json.loads(line)) == QUESTION_KEYS, f"{fname}: 키 이상"
        assert "#unans" not in raw and "unanswerable" not in raw, f"{fname}: 유형 누출"
        leaked = [fid for fid in finqa_ids if fid in raw]
        assert not leaked, f"{fname}: finqa_id 누출 {leaked[:3]}"

    # qid 일치
    assert {q["qid"] for q in eval_q} == {a["qid"] for a in eval_a}
    assert {q["qid"] for q in dev_q} == {a["qid"] for a in dev_a}

    # 개수
    c = Counter(a["type"] for a in eval_a)
    assert c == Counter(single=N_SINGLE, multi=N_MULTI, unanswerable=N_UNANS), c
    assert len(dev_a) == N_DEV
    dc = Counter(a["type"] for a in dev_a)
    assert dc == Counter(single=N_DEV // 2, multi=N_DEV - N_DEV // 2), dc

    # 일반 문항과 정보 부족 문항의 원본이 겹치지 않음
    assert not (parsed["eval_src_ids"] & parsed["unans_src_ids"])

    # 근거 ID 확인
    qmap = {q["qid"]: q for q in eval_q + dev_q}
    for a in eval_a + dev_a:
        ids = input_ids(qmap[a["qid"]])
        if a["type"] == "unanswerable":
            assert a["removed_evidence"] and not (set(a["removed_evidence"]) & ids), a["qid"]
        else:
            missing = set(a["gold_inds"]) - ids
            assert not missing, f"{a['qid']}: 근거가 입력에서 사라짐 {missing}"

    # 질문 파일에 정답 숫자 문자열이 그대로 들어 있는지는 검사하지 않는다
    # (FinQA는 정답이 표에 그대로 있는 문항도 있어서 정상일 수 있음)
    print("검증 통과: 키 구성, 누출 문자열, qid 일치, 개수, 원본 겹침, 근거 ID")


def summarize(parsed):
    eval_a, eval_q = parsed["eval_a"], parsed["eval_q"]
    print("\n[평가 세트 요약]")
    print(" 유형:", dict(Counter(a["type"] for a in eval_a)))
    print(" 정답 형식:", dict(Counter(a["answer_type"] for a in eval_a)))
    print(" 퍼센트 문항:", sum(a["is_percent"] for a in eval_a))
    print(" 근거 위치:", dict(Counter(a["evidence_loc"] for a in eval_a if a["evidence_loc"])))
    print(" 단계 수:", dict(sorted(Counter(a["n_steps"] for a in eval_a if a["n_steps"]).items())))
    qmap = {q["qid"]: q for q in eval_q}
    for t in ("single", "multi", "unanswerable"):
        a = next(x for x in sorted(eval_a, key=lambda x: x["qid"]) if x["type"] == t)
        q = qmap[a["qid"]]
        print(f"\n--- 예시 ({t}) {a['qid']}")
        print(" 질문:", q["question"])
        print(" 본문 문장 수:", len(q["text"]), "| 표 행 수:", len(q["table"]))
        print(" 정답:", json.dumps({k: a[k] for k in ("exe_ans", "is_percent", "program",
                                                      "gold_inds", "removed_evidence")},
                                    ensure_ascii=False))


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finqa_dir", default="FinQA")
    ap.add_argument("--out_dir", default="data_processed")
    args, _ = ap.parse_known_args()   # Colab에서 셀로 실행해도 오류가 나지 않게

    splits, src_hash = {}, {}
    names = ["test", "dev"] + (["train"] if N_FEWSHOT > 0 else [])
    for name in names:
        data, path = load_split(args.finqa_dir, name)
        verify_split(data, name)
        splits[name] = data
        src_hash[f"{name}.json"] = sha256_file(path)

    files, meta, parsed = build_all(splits)
    validate(files, parsed, splits)

    # 재현성: 한 번 더 만들어 해시 비교
    files2, _, _ = build_all(splits)
    for k in files:
        assert sha256_bytes(files[k]) == sha256_bytes(files2[k]), f"재현성 실패: {k}"
    print("재현성 통과: 두 번 실행한 출력 해시가 같습니다")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, b in files.items():
        with open(os.path.join(args.out_dir, name), "wb") as f:
            f.write(b)

    manifest = {
        "finqa_commit": git_commit(args.finqa_dir),
        "source_sha256": src_hash,
        "config": CONFIG,
        "id_map": meta["id_map"],
        "output_sha256": {k: sha256_bytes(v) for k, v in sorted(files.items())},
        "cleaning_stats": meta["cleaning_stats"],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    print("\n[정리 통계]", json.dumps(meta["cleaning_stats"], ensure_ascii=False)[:400])
    summarize(parsed)
    print("\n[저장 완료]", args.out_dir)
    for k, v in manifest["output_sha256"].items():
        print(f" {k:24s} {v[:16]}…")


if __name__ == "__main__":
    main()
