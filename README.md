# SOYO_Summarize · Summarize Model (Korean Reviews)

여행/행사 후기 요약 파이프라인.
Hugging Face 모델 seoseo99/qwen2-1_5b-sum_lk_gemini(Qwen2-1.5B-Instruct 미세조정)를 기반으로, Firestore에 쌓이는 리뷰를 10개 단위로 모아 1–3문장 요약을 생성하고 DB에 저장합니다. Docker로 한 줄 배포/실행.

- 모델 카드: https://huggingface.co/seoseo99/qwen2-1_5b-sum_lk_gemini

---

## 주요 기능
- 요약 모델: **Qwen2-1.5B-Instruct** 미세조정(finetuned)
- 환각 억제
  - **복사 편향(CopyBias)** 로짓 프로세서 → 원문에 나온 토큰/조사/구두점 우대
  - **3문장 강제**(EOS 금지 + 문장 수 기준 정지)
  - **추출식 폴백** → 이상 출력 시 원문에서 문장 3개 보수적 발췌
- 실시간 워커: `places/*/reviews/*` 구독 → 장소별 10개 모이면 합쳐 요약  
  → `places/{placeId}/meta/summary` 저장 + 각 리뷰 `summaryProcessed=True`
- CPU 기본 동작: 도커/로컬 모두 GPU 없이 실행 가능
---

## 구조
```text
├─ Dockerfile                # 요약 워커 컨테이너
├─ docker-compose.yml        # summarizer 서비스 정의
├─ requirements.txt          # 파이썬 의존성
├─ realtime_summarizer.py    # Firestore 워커 + 요약 로직
├─ cpu_qwen2.py              # (옵션) 모델 단독 실행 예제
├─ *.ipynb                   # 학습/실험 노트북(LoRA 등)
└─ .dockerignore / .gitignore
```

---

## 파이프라인 개요

1) **수집**: Firestore `places/{placeId}/reviews/{reviewId}` 에 리뷰 문서가 들어옴  
2) **버퍼링**: 장소별로 신규 리뷰를 버퍼에 쌓아 **10개**가 되면 합치기(최대 4,000자)  
3) **요약 생성**:
   - 프롬프트에서 제목/지역 출력 금지, **원문 어휘 위주** 사용
   - CopyBias 로 원문 토큰/기능 토큰(조사, 구두점) 가중치 ↑, 그 외 ↓  
   - **3문장 강제**: EOS 금지 + 문장 수 스톱핑
4) **저장**: 합친 원문을 1–3문장으로 생성하여 아래 페이로드로 저장  
   경로: `places/{placeId}/meta/summary`
   ```json
   {
     "text": "<요약문>",
     "model": "seoseo99/qwen2-1_5b-sum_lk_gemini",
     "reviewCount": 10,
     "batchSize": 10,
     "updatedAt": "<server timestamp>",
     "title": "<옵션: 장소명>",
     "region": "<옵션: 지역>",
     "sourceReviewIds": ["..."]
   }


## 모델 요약 방법
- **빔서치(decoding)**: `do_sample=False`, `num_beams=5` → 샘플링 없이 **가장 확률 높은 문장 경로** 선택(재현성 높음)
- **복사 편향(CopyBias)**: 원문에서 본 토큰들에 **로짓 가산(+bias)**, 그 외엔 **감산(-bias)** → 원문 바깥 단어 생성 억제
- **3문장 제어**: EOS 금지 프로세서 + 문장 수 스토핑 기준 → **정확히 3문장**에서 멈춤
- **폴백**: 이상 출력 시 **원문 문장 3개 추출**로 안전 대체

---

## 로컬 실행
```bash
# 가상환경 준비
pip install -r requirements.txt

# Firestore 실서버 사용
python realtime_summarizer.py --batch-size 10 --write --project <PROJECT_ID>
```

## 도커로 실행
```bash
# 1) 이미지 빌드
docker compose build summarizer
# 2) 백그라운드 실행
docker compose up -d summarizer
# 3) 실시간 로그
docker compose logs -f summarizer
# 4) 중지 / 정리
docker compose stop summarizer
docker compose rm -f summarizer  # 컨테이너 제거(이미지는 보존)
```
