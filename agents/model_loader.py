import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def load_model():
    # 4-bit 양자화 설정
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 모델 로딩
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16,
    )

    model.eval()

    return tokenizer, model
