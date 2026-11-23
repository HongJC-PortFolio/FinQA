def build_p0_prompt(item):
    # 표 데이터를 원본 출처 ID와 함께 문자열로 변환
    table_lines = [
        f'{row["id"]}: ' + " | ".join(map(str, row["cells"]))
        for row in item["table"]
    ]

    # 본문 데이터를 원본 출처 ID와 함께 문자열로 변환
    text_lines = [
        f'{sentence["id"]}: {sentence["sent"]}'
        for sentence in item["text"]
    ]

    table_content = "\n".join(table_lines)
    text_content = "\n".join(text_lines)

    prompt = f"""You are a financial numerical reasoning assistant.

Answer the question using only the provided financial report.
Perform any necessary calculations yourself without external tools.
If the information is insufficient, respond with INSUFFICIENT_INFORMATION.

Financial Table:
{table_content}

Financial Text:
{text_content}

Question:
{item["question"]}

Return only the final numerical answer or INSUFFICIENT_INFORMATION.
"""
    return prompt

import time
import torch


def run_p0(item, tokenizer, model):
    # 1. 질문 데이터로 P0 프롬프트 생성
    prompt = build_p0_prompt(item)

    # 2. Qwen 채팅 템플릿 적용
    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = inputs["input_ids"].shape[-1]

    # 3. 추론 시간 측정
    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start

    # 4. 새로 생성된 토큰만 디코딩
    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    # 5. 추론 결과 반환 (정답 데이터 사용 안 함)
    return {
        "qid": item["qid"],
        "prediction": answer,
        "latency_sec": elapsed,
        "input_tokens": input_length,
        "output_tokens": len(generated_tokens),
    }
