# -*- coding: utf-8 -*-
"""
Firestore 실시간 요약 워커 (LLM 요약: 복사 편향 + 3문장 강제)
- 모델: seoseo99/qwen2-1_5b-sum_lk_gemini (Transformers)
- places/{placeId}/reviews/* 가 place별로 10개 모이면 합쳐 요약 → places/{placeId}/meta/summary 에 저장
- 각 review 문서에 summaryProcessed=True 표시
- CPU 기본(도커/로컬 모두 동작)
"""
import os, time, threading, json, re, unicodedata, platform, warnings, sys
from pathlib import Path
from typing import Optional, Set
warnings.filterwarnings("ignore")

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    LogitsProcessor, LogitsProcessorList,
    StoppingCriteria, StoppingCriteriaList
)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import firebase_admin
from firebase_admin import credentials, firestore as admin_fs
from google.cloud import firestore as gcf

print("Python :", platform.python_version())
print("CUDA   :", torch.cuda.is_available())

# ======================== 모델/요약 설정 ========================
MODEL_ID = "seoseo99/qwen2-1_5b-sum_lk_gemini"
MAX_CTX_TOKENS   = 1024
MAX_JOIN_CHARS   = 4000   # 리뷰 합칠 때 최대 길이
BATCH_DEFAULT    = 10     # 장소별 리뷰가 이 수에 도달하면 요약
TOK: Optional[AutoTokenizer] = None
MDL: Optional[AutoModelForCausalLM] = None

SYS_PROMPT = (
    "다음 한국어 리뷰 본문을 1~3문장으로 간결하게 요약하세요. "
    "제목/지역/날짜는 출력하지 말고, 원문에 등장한 어휘를 위주로 사용하세요. "
    "과장/광고 톤은 금지하며 자연스러운 한국어 종결어미로 끝내세요."
)

def load_model_once():
    global TOK, MDL
    if TOK is not None and MDL is not None:
        return TOK, MDL
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    TOK, MDL = tok, mdl
    print("=== summarizer model ready ===")
    print(f"device: {next(mdl.parameters()).device} | dtype: {next(mdl.parameters()).dtype}")
    return tok, mdl

def make_ids(tok, mdl, review_text: str, max_ctx=MAX_CTX_TOKENS):
    body = unicodedata.normalize("NFKC", (review_text or "").replace("\n", " "))
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user",   "content": "【리뷰 본문】\n" + body.strip()}
    ]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if ids.shape[-1] > max_ctx:
        keep = int(len(body) * max_ctx / ids.shape[-1])
        cut  = body[:max(600, keep)]
        msgs[1]["content"] = "【리뷰 본문】\n" + cut
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return ids.to(mdl.device)

class CopyBias(LogitsProcessor):
    def __init__(self, allow_ids: Set[int], func_ids: Set[int], bias_in=2.0, bias_out=-1.0, eos_id: Optional[int]=None):
        self.allow_ids = set(allow_ids) | set(func_ids)
        self.func_ids  = set(func_ids)
        self.bias_in   = float(bias_in)
        self.bias_out  = float(bias_out)
        self.eos_id    = eos_id
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        add = torch.full_like(scores, self.bias_out)
        if self.allow_ids:
            idxs = torch.tensor(sorted(self.allow_ids), device=scores.device, dtype=torch.long)
            add.index_fill_(1, idxs, self.bias_in)
        if self.eos_id is not None:
            add[:, self.eos_id] = 0.0
        return scores + add

def build_allowed_ids(tok, text: str):
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    allow = set(ids)
    glue_words = [
        "은","는","이","가","을","를","에","에서","으로","와","과","도","만","까지","부터","보다",
        "처럼","하며","지만","면서","했다","있었다","없었다","좋았다","아쉬웠다",".",",","!","?"," "
    ]
    func_ids: Set[int] = set()
    for w in glue_words:
        func_ids |= set(tok(w, add_special_tokens=False)["input_ids"])
    return allow, func_ids

class EOSUntilNSentences(LogitsProcessor):
    def __init__(self, tok, prompt_len: int, n: int, eos_id: int):
        self.tok = tok
        self.prompt_len = prompt_len
        self.n = int(n)
        self.eos_id = int(eos_id)
        self._re_end = re.compile(r"[\.!?]")
    def __call__(self, input_ids, scores):
        new_tokens = input_ids[0, self.prompt_len:].tolist()
        text = self.tok.decode(new_tokens, skip_special_tokens=True)
        text = unicodedata.normalize("NFKC", text).replace("。",".").replace("！","!").replace("？","?")
        sent_cnt = len(self._re_end.findall(text))
        if sent_cnt < self.n:
            scores[:, self.eos_id] = -float("inf")
        return scores

class NSentenceStop(StoppingCriteria):
    def __init__(self, tok, prompt_len: int, limit: int):
        self.tok = tok
        self.prompt_len = prompt_len
        self.limit = int(limit)
        self._re_end = re.compile(r"[\.!?]")
    def __call__(self, input_ids, scores, **kwargs):
        new_tokens = input_ids[0, self.prompt_len:].tolist()
        text = self.tok.decode(new_tokens, skip_special_tokens=True)
        text = unicodedata.normalize("NFKC", text).replace("。",".").replace("！","!").replace("？","?")
        return len(self._re_end.findall(text)) >= self.limit

_RE_KO = re.compile(r"[가-힣]{2,}")
def looks_hallucinated(out: str, src: str) -> bool:
    src_set = set(re.findall(_RE_KO, unicodedata.normalize("NFKC", src)))
    out_set = set(re.findall(_RE_KO, unicodedata.normalize("NFKC", out)))
    extra = [w for w in out_set if w not in src_set]
    return len(out_set) > 0 and (len(extra)/max(1,len(out_set)) > 0.25)

def extractive_fallback(src: str, n=3) -> str:
    s = unicodedata.normalize("NFKC", (src or "")).replace("\n"," ")
    parts = re.split(r'(?<=[\.!?])\s+', s.strip())
    parts = [p.strip() for p in parts if p.strip()]
    out = " ".join(parts[:n]).strip()
    if out and not re.search(r"[\.!?]$", out): out += "."
    return out

@torch.inference_mode()
def summarize_copy_favored(review_text: str) -> str:
    tok, mdl = load_model_once()
    input_ids = make_ids(tok, mdl, review_text, max_ctx=MAX_CTX_TOKENS)
    prompt_len = input_ids.shape[-1]
    allow, func = build_allowed_ids(tok, review_text)
    processors = LogitsProcessorList([
        CopyBias(allow, func, bias_in=2.0, bias_out=-1.0, eos_id=tok.eos_token_id),
        EOSUntilNSentences(tok, prompt_len, n=3, eos_id=tok.eos_token_id),
    ])
    stops = StoppingCriteriaList([NSentenceStop(tok, prompt_len, limit=3)])

    gen = mdl.generate(
        input_ids=input_ids,
        max_new_tokens=220,
        do_sample=False, num_beams=5,
        no_repeat_ngram_size=5, repetition_penalty=1.1,
        length_penalty=1.0, early_stopping=True,
        logits_processor=processors, stopping_criteria=stops,
        eos_token_id=tok.eos_token_id,
    )
    out = tok.decode(gen[0, prompt_len:], skip_special_tokens=True)
    out = unicodedata.normalize("NFKC", out)
    out = out.replace("\n", " ").replace("。",".").replace("！","!").replace("？","?")
    out = re.sub(r"\s+([\.!?])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if looks_hallucinated(out, review_text):
        out = extractive_fallback(review_text, n=3)
    if out and not re.search(r"[\.!?]$", out):
        out += "."
    return out

# ======================== Firestore ========================
def init_db(use_emulator: bool, project_id: Optional[str]):
    if use_emulator or os.environ.get("FIRESTORE_EMULATOR_HOST"):
        os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:8080")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(options={"projectId": project_id or "local-sim"})
        return admin_fs.client()

    # 서비스 계정(firebase.json) 우선
    cred_path = Path(__file__).parent / "firebase.json"
    if cred_path.exists():
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(cred_path))
            firebase_admin.initialize_app(cred, {"projectId": project_id} if project_id else None)
    else:
        # GAC 환경변수로도 지원
        if not firebase_admin._apps:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {"projectId": project_id} if project_id else None)
    return admin_fs.client()

def join_reviews(docs, limit=10, max_chars=MAX_JOIN_CHARS):
    texts = []
    for d in docs[:limit]:
        data = d.to_dict() or {}
        t = (data.get("review") or data.get("text") or "").strip()
        if t:
            texts.append(t)
    joined = " ".join(texts)
    return joined[:max_chars]

def run_worker(batch_size: int = BATCH_DEFAULT, write: bool = True, emulator: bool = False, project: Optional[str]=None):
    db = init_db(emulator, project)
    buffers = {}
    lock = threading.Lock()

    def on_snapshot(col_snapshot, changes, read_time):
        with lock:
            for ch in changes:
                doc = ch.document
                ref = doc.reference

                # places/{placeId}/reviews/{reviewId} 만
                if not (ref.parent.id == "reviews"
                        and ref.parent.parent is not None
                        and ref.parent.parent.parent is not None
                        and ref.parent.parent.parent.id == "places"):
                    continue

                data = doc.to_dict() or {}
                if data.get("summaryProcessed"):
                    continue

                place_id = ref.parent.parent.id
                review_text = (data.get("review") or data.get("text") or "").strip()
                if not review_text:
                    continue

                key = str(place_id)
                buffers.setdefault(key, [])
                buffers[key].append(doc)
                print(f"[{key}] buffered {len(buffers[key])}/{batch_size}")

                if len(buffers[key]) >= batch_size:
                    # 최신 순으로 정렬(선택사항): 여기서는 단순 수집 순서 사용
                    docs = buffers[key]
                    joined = join_reviews(docs, limit=batch_size, max_chars=MAX_JOIN_CHARS)
                    place_doc = db.collection("places").document(key).get()
                    title = ""
                    region = ""
                    if place_doc.exists:
                        pd = place_doc.to_dict() or {}
                        title = pd.get("name") or pd.get("title") or ""
                        region = pd.get("region") or pd.get("addr") or ""

                    summary = summarize_copy_favored(joined)

                    payload = {
                        "text": summary,
                        "model": MODEL_ID,
                        "reviewCount": len(docs),
                        "batchSize": batch_size,
                        "updatedAt": gcf.SERVER_TIMESTAMP,
                        "title": title,
                        "region": region,
                        "sourceReviewIds": [d.id for d in docs],
                    }
                    target = db.collection("places").document(key).collection("meta").document("summary")

                    if write:
                        target.set(payload, merge=True)
                        for d in docs:
                            d.reference.set({"summaryProcessed": True}, merge=True)
                        print(f"▶ summarized {key} ({len(docs)}) -> places/{key}/meta/summary")
                    else:
                        print(f"[DRY-RUN] would write to places/{key}/meta/summary:")
                        print(json.dumps({k: v for k, v in payload.items() if k != "updatedAt"}, ensure_ascii=False, indent=2))

                    buffers[key] = []

    # 새 리뷰 변화 구독
    unsubscribe = db.collection_group("reviews").on_snapshot(on_snapshot)
    print(f"Realtime worker started. Listening to places/*/reviews/* (batch={batch_size}, write={write}) …")

    def stop():
        try:
            unsubscribe()
        except Exception:
            pass
    return stop

# ======================== CLI ========================
if __name__ == "__main__":
    import argparse, atexit
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--write", action="store_true", default=True)
    ap.add_argument("--emulator", action="store_true")
    ap.add_argument("--project", type=str, default=None)
    args = ap.parse_args()

    stopper = run_worker(batch_size=args.batch_size, write=args.write, emulator=args.emulator, project=args.project)
    atexit.register(stopper)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stopper()
        print("\nstopped.")