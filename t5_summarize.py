# -*- coding: utf-8 -*-
"""
ko_t5_summarizer.py
- 모델: eenzeenee/t5-base-korean-summarization
- 특징:
  1) Hugging Face pipeline 기반
  2) 샘플링 옵션(temperature, top_p) 적용 → 자연스러움 ↑
  3) 후처리: 중복 문장, 불완전 문장 제거
"""

import re
import time
from transformers import pipeline

MODEL_ID = "eenzeenee/t5-base-korean-summarization"

# ---------------- 모델 로딩 ----------------
def load_summarizer():
    print(f"[LOAD] {MODEL_ID}")
    return pipeline(
        "summarization",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device=-1  # GPU 있으면 0
    )

# ---------------- 후처리 ----------------
def clean_summary(text: str) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    out = []
    seen = set()
    for s in sents:
        s = s.strip()
        if not s:
            continue
        # 중복 제거
        if s in seen:
            continue
        seen.add(s)
        # 문장 끝 보정
        if not re.search(r'[.!?]$', s):
            continue
        out.append(s)
    return " ".join(out).strip()

# ---------------- 요약 함수 ----------------
def summarize(text: str, summarizer) -> str:
    result = summarizer(
        text,
        min_length=60,
        max_length=180,
        do_sample=True,       # 샘플링 활성화
        temperature=0.7,      # 다양성 ↑ 안정성 유지
        top_p=0.9,            # 상위 90% 확률 분포에서만 샘플링
        num_return_sequences=1
    )
    raw = result[0]['summary_text']
    return clean_summary(raw)

# ---------------- 실행 ----------------
if __name__ == "__main__":
    REVIEWS = [
       """올여우가 연차를 가지고 증평 인삼골축제를 방문한 후, 다양한 경험과 감상을 공유했습니다. 이 축제는 2024년 10월 3일부터 6일까지 진행되는 행사이며, 가장 특별한 점은 19:30에 열리는 행복증평불꽃놀이였습니다.  축제 장소는 보강천 미루나무숲에 위치해 있어 편리하게 접근할 수 있었고, 주차 정보를 통해 제1주차장인 보강천 하상에서 차를 주차하고 이동하며 즐길 수 있었습니다. 평일 낮에도 불구하고 축제는 사람들을 끊임없이 유치해, 시끌벅적한 노랫소리와 함께 신기함을 느낄 수 있었습니다.  축제 장소가 넓어 발이 닿지 않는 곳까지 여행하는 것이 가능했으며, 어린이 및 가족 체험존에서도 다양한 활동을 즐길 수 있었습니다. 비롯부터는 무료로 이용할 수 있는 부스가 있었지만, 일부 유료로 운영되는 부스도 존재했습니다.  행사 중에서는 증평 좌구산 천문대 태양관측이나 꽃신 및 고무신 등 다양한 체험을 할 수 있었습니다. 또한 식음료를 제공하는 장소나 쇼핑, 편의점 등을 이용할 수 있었고, 행사장을 방문하는 동안에는 현금 인출기가 필요하던 상황도 경험했습니다.  어린이와 가족뿐만 아니라 청년 세대에도 적합한 다양한 체험존들이 설치되어 있었으며, 즐겁게 놀며 시간을 보낼 수 있었습니다. 특히 증평의 관광 명소들에 대해 배울 수 있는 부스가 있어 역사적 배경을 이해하는 데 도움이 되었고, 일부 참여 이벤트를 통해 소소한 선물을 받았습니다.  행사에서 제공된 음식들은 다양하며 가격도 다양했으며, 한우의 할인 행사를 즐기면서 증평 인삼과 다양한 농특산물도 구매할 수 있었습니다. 또한 NH농협은행의 금융 어플을 통해 쌀 등을 받는 이벤트를 경험하기도 했습니다.  마지막으로, 축제는 사람들의 참여와 에너지를 느낄 수 있는 다양한 활동과 높이 비추는 불꽃놀이로 가득 찼으며, 즐거운 시간을 보낼 수 있었습니다. 이 축제를 통해 증평의 아름다움과 지역 문화를 배우고 경험할 수 있었고 추천해드리는 행사는 '증평 인삼골축제'입니다."""
    ]

    summarizer = load_summarizer()
    t0 = time.time()
    print("=== 요약 ===")
    print(summarize(" ".join(REVIEWS), summarizer))
    print(f"(elapsed {time.time()-t0:.2f}s)")