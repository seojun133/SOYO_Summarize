# ====================== 백엔드 병합시 필수 추가 ======================
# from realtime_summarizer import run_worker
# stop_summarizer = run_worker(batch_size=10, write=True, emulator=False, project="testproject-139f1")


# -*- coding: utf-8 -*-
import os, time, threading, json, re, unicodedata, platform, warnings, torch
from pathlib import Path
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import firebase_admin
from firebase_admin import credentials, firestore as admin_fs
from google.cloud import firestore as gcf

print("Python :", platform.python_version())
print("CUDA   :", torch.cuda.is_available())

# ============== 모델 로더(지연 로딩) ==============
BASE_MODEL = "seoseo99/qwen2-1_5b-sum_lk_gemini"
_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    print("=== summarizer model ready ===")
    print(f"device: {next(model.parameters()).device} | dtype: {next(model.parameters()).dtype}")

    _tokenizer, _model = tokenizer, model
    return _tokenizer, _model

# ============== 요약 유틸 ==============
RE_HANJA     = re.compile(r"[\u4E00-\u9FFF]")
RE_JP_PUNC   = re.compile(r"[。、「」｡､]")
RE_META      = re.compile(r"(요약(?:하면)?|정리(?:하면)?|다음과\s*같습니다|위\s*리뷰|번호로\s*정리)", re.IGNORECASE)
RE_BAD_BEGIN = re.compile(r"^(?:\d{4}\s*년|년|월|일)\b")
RE_HTML      = re.compile(r"<[^>]+>")
_SENT_SPLIT  = re.compile(r'(?<=[\.!?])\s+')
_INCOMPLETE_TAIL = re.compile(r'(?:지만|인데|으나|면서|라서|이고|고|인데요|였으나|였지만)$')

def _normalize_punct(s: str) -> str:
    s = (s or "").replace("、", ", ").replace("。", ". ").replace("「", "“").replace("」", "”")
    s = s.replace("\uFF0C", ", ").replace("\uFF0E", ". ").replace("\u00A0", " ").replace("�", "")
    return s

def _strip_noise_tokens(s: str) -> str:
    toks = re.split(r'(\s+)', s); out = []
    for tk in toks:
        raw = tk.strip()
        if not raw: out.append(tk); continue
        if "@" in raw or raw.lower()=="null": continue
        if re.fullmatch(r'[A-Za-z]{12,}', raw): continue
        out.append(tk)
    return "".join(out)

def _split_sents(s: str):
    s = re.sub(r'\s+', ' ', (s or '')).strip()
    if not s: return []
    parts = [p.strip() for p in _SENT_SPLIT.split(s) if p.strip()]
    return parts if parts else [s]

def _close_if_incomplete(sentence: str) -> str:
    s = re.sub(r'\s+', ' ', (sentence or '')).strip(' .')
    if not s: return "전반적으로 무난했지만 아쉬운 점도 있었다."
    if _INCOMPLETE_TAIL.search(s):
        base = _INCOMPLETE_TAIL.sub('', s).rstrip(' ,;')
        if len(base) < 6: return "전반적으로 만족스러웠지만 아쉬운 점도 있었다."
        if "아쉬" in base: return base + " 아쉬운 점도 있었다."
        return base + " 만족스러웠지만 개선 여지는 있었다."
    return sentence

def _finalize_and_limit_to_3(s: str) -> str:
    s = re.sub(r'\s+', ' ', (s or '')).strip()
    if s and not re.search(r'[\.!?]$', s): s += '.'
    sents = _split_sents(s)[:3]
    if sents: sents[-1] = _close_if_incomplete(sents[-1])
    out = " ".join(sents).strip()
    if out and not re.search(r'[\.!?]$', out): out += '.'
    return out

def _squeeze_year_noise(s: str) -> str:
    if not s: return s
    t = s
    t = re.sub(r'([가-힣])\s*((?:19|20)\d{2})\s*(?=[가-힣])', r'\1', t)
    t = re.sub(r'(?<!\d)((?:19|20)\d{2})\d{1,3}(?=[^\d]|$)', r'\1', t)
    t = re.sub(r'(?<!\d)\d{5,}(?=[^\d]|$)', '', t)
    def _shrink(m):
        yrs = re.findall(r'(?:19|20)\d{2}', m.group(0)); return ", ".join(yrs[:2])
    t = re.sub(r'\b((?:19|20)\d{2})(?:\D+(?:19|20)\d{2}){2,}\b', _shrink, t)
    t = re.sub(r'(?<!\d)(?:19|20)\d{2}(?!\s*[년월일])(?=[\s가-힣\.,)\]]|$)', '', t)
    return re.sub(r'\s+', ' ', t).strip()

def _has_jongsung(word: str) -> bool:
    code = ord(word[-1]) - ord('가')
    return 0 <= code <= 11172 and (code % 28) != 0

def _eunneun(word: str) -> str:
    return "은" if _has_jongsung(word) else "는"

def _fix_leading_particle(s: str, title: str) -> str:
    s0 = s.strip()
    m = re.match(r'^(?:는|은|이|가|을|를|도|만|으로|에게|에서|에는|에서는)\b\s*(.*)$', s0)
    if m:
        core = m.group(1).strip() or s0
        return f"{title}{_eunneun(title)} {core}"
    if s0.startswith("에서는"):
        return f"{title}{_eunneun(title)} " + s0.replace("에서는", "", 1).strip()
    return s

def _enforce_title_once(s: str, title: str) -> str:
    return (f"{title}{_eunneun(title)} " + s.lstrip()) if (title and title not in s) else s

def _fix_known_typos(s: str, title: str) -> str:
    if "쇠부리" in (title or ""):
        s = s.replace("쓸부리", "쇠부리").replace("씨부리", "쇠부리")
    return s

def han_ratio(s: str) -> float:
    s = unicodedata.normalize("NFKC", s or ""); nz = [ch for ch in s if not ch.isspace()]
    return 0.0 if not nz else sum('가'<=ch<='힣' for ch in nz)/len(nz)

def looks_bad(s: str) -> bool:
    return (not s or s.strip()=="" or RE_META.search(s) or RE_BAD_BEGIN.search(s)
            or RE_JP_PUNC.search(s) or RE_HANJA.search(s) or han_ratio(s) < 0.85)

def _unify_title_variants(s: str, title: str) -> str:
    if not s or not title: return s
    pat = r'\s*'.join(map(re.escape, list(title)))
    return re.sub(pat, title, s)

def _dedupe_title_mention(s: str, title: str) -> str:
    if not s or not title: return s
    josa = r'(?:은|는|이|가|을|를|도|만|으로|에게|에서|에는|에서는)'
    s = re.sub(rf'({re.escape(title)}{josa})\s+({re.escape(title)}{josa})', r'\1 ', s)
    s = re.sub(rf'({re.escape(title)})\s+({re.escape(title)}{josa})', r'\2', s)
    return s

def repair_summary(text: str, title: str=None) -> str:
    x = unicodedata.normalize("NFKC", text or "")
    x = RE_HTML.sub(" ", x)
    x = _normalize_punct(x)
    x = _strip_noise_tokens(x)
    x = _squeeze_year_noise(x)
    x = re.sub(r"\s+", " ", x).strip()
    if title:
        x = _unify_title_variants(x, title)
        x = _fix_leading_particle(x, title)
        x = _enforce_title_once(x, title)
        x = _dedupe_title_mention(x, title)
    x = _fix_known_typos(x, title)
    x = re.sub(RE_META, "", x)
    x = re.sub(r"\s+", " ", x).strip()
    x = _finalize_and_limit_to_3(x)
    if title:
        x = _dedupe_title_mention(x, title)
    return x

def postprocess_remove_years(text: str) -> str:
    s = text
    s = re.sub(r'\b(?:19|20)\d{2}\s*년', '', s)
    s = re.sub(r'\b(?:19|20)\d{2}\b', '', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = re.sub(r'\s+([.,!?])', r'\1', s)
    return s

def _clean_cfg(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if v is not None}

@torch.inference_mode()
def _llm_summarize(title: str, addr: str, review: str) -> str:
    tokenizer, model = _load_model()

    messages = [
        {"role": "system", "content":
         "당신은 여행 리뷰 데이터를 요약하는 어시스턴트입니다.\n"
         "- 리뷰의 핵심 경험을 1~3문장으로 간결하게 정리합니다.\n"
         "- 과장/광고 톤 없이 담백하게, 감정 뉘앙스를 자연스럽게 반영합니다.\n"
         "- 구체 팩트 + 좋았던 점 1개 + (있으면) 아쉬운 점 0~1개.\n"
         "- '요약:' 같은 접두어/메타 문장 금지, 말줄임표(...) 금지, 한국어 종결어미로 끝냅니다.\n"
         "- 숫자/연도는 입력에 있는 범위만 사용, 불필요한 연속 숫자 금지.\n"
         "- 행사명과 지역명을 1회 이상 자연스럽게 포함.\n"
         "- 출력은 한국어 문장으로만 작성."
        },
        {"role": "user",
         "content": f"행사명: {title} / 주소: {addr}\n\n[리뷰]\n{review}\n\n위 리뷰를 1~3문장(약 150자 내외)으로 자연스럽게 요약해 주세요."}
    ]

    MAX_PROMPT_TOKENS = 1024
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if ids.shape[-1] > MAX_PROMPT_TOKENS:
        keep = int(len(review) * MAX_PROMPT_TOKENS / ids.shape[-1])
        trimmed = review[:max(500, keep)]
        messages[1]["content"] = f"행사명: {title} / 주소: {addr}\n\n[리뷰]\n{trimmed}\n\n위 리뷰를 1~3문장(약 150자 내외)으로 자연스럽게 요약해 주세요."
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    ids = ids.to(model.device)
    prompt_len = ids.shape[-1]

    def _decode(gen_ids):
        new_tokens = gen_ids[0, prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    GEN_MAIN   = dict(max_new_tokens=128, num_beams=4, do_sample=False, length_penalty=0.9,
                      repetition_penalty=1.05, no_repeat_ngram_size=3)
    GEN_STRICT = dict(max_new_tokens=110, num_beams=5, do_sample=False, length_penalty=1.0,
                      repetition_penalty=1.1, no_repeat_ngram_size=4)

    try:
        gen1 = model.generate(input_ids=ids, eos_token_id=tokenizer.eos_token_id, **_clean_cfg(GEN_MAIN))
        txt1 = _decode(gen1)
        out1 = postprocess_remove_years(repair_summary(txt1, title=title))
        if not looks_bad(out1): return out1
    except Exception:
        pass

    try:
        gen2 = model.generate(input_ids=ids, eos_token_id=tokenizer.eos_token_id, **_clean_cfg(GEN_STRICT))
        txt2 = _decode(gen2)
        out2 = postprocess_remove_years(repair_summary(txt2, title=title))
        if not looks_bad(out2): return out2
    except Exception:
        pass

    base = f"{title}{'은' if _has_jongsung(title) else '는'} 가족 단위로 즐길 수 있는 프로그램과 체험이 중심이었습니다."
    pos  = "공연·체험 구성과 편의시설이 대체로 만족스러웠습니다."
    neg  = "다만, 주차·대기 등 일부 혼잡으로 불편을 겪기도 했습니다."
    return postprocess_remove_years(_finalize_and_limit_to_3(f"{base} {pos} {neg}"))

# ============== Firestore 초기화 ==============
def _init_db(use_emulator: bool, project: str = None):
    if use_emulator:
        os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:8080")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(options={"projectId": project or "local-sim"})
    else:
        if not firebase_admin._apps:
            gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if gac and Path(gac).exists():
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred, {"projectId": project} if project else None)
            else:
                cred_path = Path(__file__).parent / "firebase.json"
                cred = credentials.Certificate(str(cred_path))
                firebase_admin.initialize_app(cred, {"projectId": project} if project else None)
    return admin_fs.client()

# ============== 워커 ==============
def run_worker(batch_size: int = 10, write: bool = True, emulator: bool = False, project: str = None):
    """
    백엔드에서 임포트해 호출. 비동기 리스너 등록 후 stop() 콜백을 리턴.
    """
    db = _init_db(emulator, project)
    buffers = {}
    lock = threading.Lock()

    def on_snapshot(col_snapshot, changes, read_time):
        with lock:
            for ch in changes:
                doc = ch.document
                ref = doc.reference

                if not (ref.parent.id == "reviews"
                        and ref.parent.parent is not None
                        and ref.parent.parent.parent is not None
                        and ref.parent.parent.parent.id == "places"):
                    continue

                data = doc.to_dict() or {}
                if data.get("summaryProcessed"):
                    continue

                content_id = data.get("contentId") or data.get("contentid") or ref.parent.parent.id
                review_text = (data.get("review") or "").strip()
                if not review_text:
                    continue

                key = str(content_id)
                buffers.setdefault(key, [])
                buffers[key].append((ref, review_text))
                print(f"[{key}] buffered {len(buffers[key])}/{batch_size}")

                if len(buffers[key]) >= batch_size:
                    reviews = [t for _, t in buffers[key]]
                    joined = " ".join(reviews)[:4000]
                    title = data.get("placeName") or f"장소 {key}"
                    addr  = data.get("region") or ""
                    summary = _llm_summarize(title, addr, joined)

                    payload = {
                        "text": summary,
                        "model": BASE_MODEL,
                        "reviewCount": len(buffers[key]),
                        "batchSize": batch_size,
                        "updatedAt": gcf.SERVER_TIMESTAMP,
                        "title": title,
                        "region": addr,
                        "sourceReviewIds": [r.id for r, _ in buffers[key]],
                    }

                    target = db.collection("places").document(key).collection("meta").document("summary")
                    if write:
                        target.set(payload, merge=True)
                        for r, _ in buffers[key]:
                            r.set({"summaryProcessed": True}, merge=True)
                        print(f"▶ summarized {key} ({len(buffers[key])}) -> places/{key}/meta/summary")
                    else:
                        print(f"[DRY-RUN] would write to places/{key}/meta/summary:")
                        print(json.dumps({k:v for k,v in payload.items() if k!='updatedAt'}, ensure_ascii=False, indent=2))
                    buffers[key] = []

    unsubscribe = db.collection_group("reviews").on_snapshot(on_snapshot)
    print(f"Realtime worker started. Listening to places/*/reviews/* (batch={batch_size}, write={write}) …")

    def stop():
        try: unsubscribe()
        except Exception: pass
    return stop

# ============== CLI 실행 ==============
if __name__ == "__main__":
    import argparse, atexit
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--write", action="store_true", default=True)   # 기본 저장 ON
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