# Adaptive FinQA Agent (9조)

## 디렉토리 구조

```
finqa-agent/
├── FinQA/                      # 원본 데이터 (git clone, 수정 금지)
│   └── dataset/
│       ├── train.json          # few-shot 예시용 (선택)
│       ├── dev.json            # 개발 세트 원본
│       ├── test.json           # 평가 세트 원본
│       └── private_test.json   # 사용 안 함 (정답 없음)
│
├── scripts/
│   └── prepare_data.py         # 데이터 가공 (이 단계)
│
├── data_processed/             # prepare_data.py 출력. 확정 후 수정 금지
│   ├── eval_questions.jsonl    # 에이전트 입력 100문항 (qid, question, text, table)
│   ├── eval_answers.jsonl      # 채점용 정답·라벨 100문항
│   ├── dev_questions.jsonl     # 개발용 입력 30문항
│   ├── dev_answers.jsonl       # 개발용 정답 30문항
│   ├── unans_review.csv        # 정보 부족 20문항 검수표 (팀원 2명)
│   └── manifest.json           # 시드, ID 대응표, 원본·출력 SHA-256
│
├── agents/                     # (다음 단계) 공통 도구 + P0, P1, M−B, M−A, M
├── scoring/                    # (다음 단계) score.py
├── results/                    # (실험 후) {조건}_run{n}.jsonl
└── reports/                    # (분석 후) 표, 그래프
```

에이전트 코드는 `*_questions.jsonl`만 읽고, 채점 코드는 `*_answers.jsonl`과 `results/`만 읽는다.

## 실행 (Colab)

```python
%cd /content
!git clone https://github.com/czyssrs/FinQA.git finqa-agent/FinQA
%cd /content/finqa-agent
# scripts/prepare_data.py 를 업로드한 뒤
!python scripts/prepare_data.py --finqa_dir FinQA --out_dir data_processed
```

같은 설정이면 `manifest.json`의 `output_sha256`이 항상 같다. 팀원끼리 해시를 비교해 같은 평가 세트인지 확인한다.

## 가공 규칙 요약

| 단계 | 규칙 |
|---|---|
| 본문 | 구두점만 있는 문장 제거(원본 번호 유지). 깨진 문자 복원: 2019s→’s, 201c/201d→“ ” |
| 표 | 모든 행에 table_0부터 원본 번호. 데이터 칸의 "2014"/"$ 2014"는 "—"(값 없음)으로 치환. 계산식이 2014를 쓰는 문항은 예외 |
| 라벨 오류 | 사람이 쓴 답(answer)과 정답값(exe_ans)이 어긋나는 문항은 추출 후보에서 제외 (×100, ÷100, 반올림 허용) |
| 평가 세트 | test에서 단일 40 + 복합 40 + 정보 부족 20, 시드 42 |
| 정보 부족 | 평가 80문항과 겹치지 않는 test 문항에서 정답 근거(gold_inds) 행·문장 삭제. 지워도 계산 숫자가 문서에 모두 남는 문항은 제외 |
| 개발 세트 | dev에서 단일 15 + 복합 15 |
| ID | 질문 파일은 익명 qid(E001~, D001~)만. 원본 FinQA id는 정답 파일과 manifest에만 |

## 파일 형식

### `*_questions.jsonl` (에이전트 입력)
```json
{"qid": "E001",
 "question": "what was total rent expense for fiscal years 2010 to 2012 , in millions?",
 "text": [{"id": "text_0", "part": "pre", "sent": "abiomed , inc ."}],
 "table": [{"id": "table_0", "cells": ["fiscal year ending march 31,", "operating leases ( in $ 000s )"]},
           {"id": "table_1", "cells": ["2013", "1473"]}]}
```
- `text[].id`, `table[].id`가 에이전트가 인용하는 출처 ID다. FinQA `gold_inds`와 같은 체계다.
- `table_0`은 보통 열 이름(헤더)이지만, 헤더 없이 첫 행부터 데이터인 표도 있다.

### `*_answers.jsonl` (채점·분석)
| 키 | 내용 |
|---|---|
| qid / finqa_id | 익명 번호 / FinQA 원본 id |
| type | single, multi, unanswerable |
| answer_type | number, yes_no, unanswerable |
| exe_ans | 정답값 (정보 부족은 null) |
| is_percent | 퍼센트 문항. FinQA는 퍼센트를 0.25로도 25로도 저장하므로 채점 때 ×100, ÷100 모두 허용 |
| answer_text | 사람이 쓴 반올림 정답 |
| program, n_steps | 정답 계산식, 단계 수 |
| gold_inds, evidence_loc | 정답 근거 ID, 근거 위치(table/text/both) |
| uses_const | 계산식에 문서에 없는 상수(const_100 등) 사용 여부 |
| removed_evidence, source | 정보 부족 문항에서 지운 근거, 원본 정답 정보 |
