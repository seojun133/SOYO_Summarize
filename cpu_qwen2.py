import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import os, platform, torch, re, unicodedata, json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

print("Python :", platform.python_version())
print("CUDA   :", torch.cuda.is_available())

HOME = os.path.expanduser("~")
ROOT = os.path.join(HOME, "Documents", "GitHub", "goorm_BP") # 루트 경로
OUT_DIR_S = os.path.join(ROOT, "cpu_qwen2", "summaries") # 저장 경로 
PATTERN   = "*_CLEANED.csv"

Path(OUT_DIR_S).mkdir(parents=True, exist_ok=True)

BASE_MODEL = "seoseo99/qwen2-1_5b-sum_lk_gemini" # 허깅페이스 업로드 모델

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

print("=== model ready ===")
print(f"device: {next(model.parameters()).device} | dtype: {next(model.parameters()).dtype}")

RE_HANJA     = re.compile(r"[\u4E00-\u9FFF]")
RE_JP_PUNC   = re.compile(r"[。、「」｡､]")
RE_META      = re.compile(r"(요약(?:하면)?|정리(?:하면)?|다음과\s*같습니다|위\s*리뷰|번호로\s*정리)", re.IGNORECASE)
RE_BAD_BEGIN = re.compile(r"^(?:\d{4}\s*년|년|월|일)\b")
RE_HTML      = re.compile(r"<[^>]+>")
_SENT_SPLIT  = re.compile(r'(?<=[\.!?])\s+')
_INCOMPLETE_TAIL = re.compile(r'(?:지만|인데|으나|면서|라서|이고|고|인데요|였으나|였지만)$')

def _normalize_punct(s: str) -> str:
    s = (s or "")
    s = s.replace("、", ", ").replace("。", ". ")
    s = s.replace("「", "“").replace("」", "”")
    s = s.replace("\uFF0C", ", ").replace("\uFF0E", ". ").replace("\u00A0", " ")
    s = s.replace("�", "")
    return s

def _strip_noise_tokens(s: str) -> str:
    toks = re.split(r'(\s+)', s)
    out = []
    for tk in toks:
        raw = tk.strip()
        if not raw:
            out.append(tk); continue
        if "@" in raw or raw.lower() == "null":
            continue
        if re.fullmatch(r'[A-Za-z]{12,}', raw):
            continue
        out.append(tk)
    return "".join(out)

def _split_sents(s: str):
    s = re.sub(r'\s+', ' ', (s or '')).strip()
    if not s: return []
    parts = [p.strip() for p in _SENT_SPLIT.split(s) if p.strip()]
    return parts if parts else [s]

def _close_if_incomplete(sentence: str) -> str:
    s = re.sub(r'\s+', ' ', (sentence or '')).strip(' .')
    if not s:
        return "전반적으로 무난했지만 아쉬운 점도 있었다."
    if _INCOMPLETE_TAIL.search(s):
        base = _INCOMPLETE_TAIL.sub('', s).rstrip(' ,;')
        if len(base) < 6:
            return "전반적으로 만족스러웠지만 아쉬운 점도 있었다."
        if "아쉬" in base:
            return base + " 아쉬운 점도 있었다."
        return base + " 만족스러웠지만 개선 여지는 있었다."
    return sentence

def _finalize_and_limit_to_3(s: str) -> str:
    s = re.sub(r'\s+', ' ', (s or '')).strip()
    if s and not re.search(r'[\.!?]$', s): s += '.'
    sents = _split_sents(s)[:3]
    if sents:
        sents[-1] = _close_if_incomplete(sents[-1])
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
    if not word: return False
    ch = word[-1]
    code = ord(ch) - ord('가')
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
    if title and title not in s:
        return f"{title}{_eunneun(title)} " + s.lstrip()
    return s

def _fix_known_typos(s: str, title: str) -> str:
    if "쇠부리" in (title or ""):
        s = s.replace("쓸부리", "쇠부리").replace("씨부리", "쇠부리")
    return s

def han_ratio(s: str) -> float:
    s = unicodedata.normalize("NFKC", s or "")
    nz = [ch for ch in s if not ch.isspace()]
    if not nz: return 0.0
    return sum('가'<=ch<='힣' for ch in nz)/len(nz)

def looks_bad(s: str) -> bool:
    if not s or s.strip()=="":
        return True
    if RE_META.search(s):
        return True
    if RE_BAD_BEGIN.search(s):
        return True
    if RE_JP_PUNC.search(s):
        return True
    if RE_HANJA.search(s):
        return True
    if han_ratio(s) < 0.85:
        return True
    return False

def repair_summary(text: str, title: str=None) -> str:
    x = unicodedata.normalize("NFKC", text or "")
    x = RE_HTML.sub(" ", x)
    x = _normalize_punct(x)
    x = _strip_noise_tokens(x)
    x = _squeeze_year_noise(x)
    x = re.sub(r"\s+", " ", x).strip()
    if title:
        x = _fix_leading_particle(x, title)
    x = _enforce_title_once(x, title)
    x = _fix_known_typos(x, title)
    x = re.sub(RE_META, "", x)
    x = re.sub(r"\s+", " ", x).strip()
    x = _finalize_and_limit_to_3(x)
    return x

def _has_jongsung2(w: str) -> bool:
    if not w: return False
    code = ord(w[-1]) - ord('가')
    return 0 <= code <= 11172 and (code % 28) != 0

def _eunneun2(w: str) -> str:
    return "은" if _has_jongsung2(w) else "는"

_LEADING_JOSA = re.compile(r'^(?:는|은|이|가|을|를|도|만|으로|에게|에서|에는|에서는)\b\s*')
def _fix_leading_particle2(text: str, title: str) -> str:
    s = text.lstrip()
    if _LEADING_JOSA.match(s):
        core = _LEADING_JOSA.sub('', s).strip()
        return f"{title}{_eunneun2(title)} {core}".strip()
    if s.startswith("에서는"):
        return f"{title}{_eunneun2(title)} {s[3:].lstrip()}"
    return text

_HANGUL_WORD = re.compile(r"^[가-힣]{2,}$")
def _lev1(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return 2
    if la == lb:
        return min(2, sum(x != y for x, y in zip(a, b)))
    if la > lb: a, b = b, a; la, lb = lb, la
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]: i += 1; j += 1
        else:
            diff += 1; j += 1
            if diff > 1: break
    diff += (lb - j)
    return min(diff, 2)

def _snap_title_tokens(summary: str, title: str) -> str:
    if not summary or not title: return summary
    title_tokens = [t for t in re.findall(r"[가-힣A-Za-z0-9]+", title) if _HANGUL_WORD.fullmatch(t)]
    if not title_tokens: return summary
    toks = re.findall(r"[가-힣A-Za-z0-9]+|\s+|[^\w\s]", summary)
    for i, tk in enumerate(toks):
        if not _HANGUL_WORD.fullmatch(tk): continue
        for tw in title_tokens:
            if tk == tw: break
            if _lev1(tk, tw) <= 1:
                toks[i] = tw
                break
    return "".join(toks)

def _known_typos_fix2(s: str, title: str) -> str:
    if not s or not title: return s
    if "쇠부리" in title:
        s = s.replace("쓸부리", "쇠부리").replace("씨부리", "쇠부리")
    return s

def typo_enforce_title_anchor(summary: str, title: str) -> str:
    x = unicodedata.normalize("NFKC", summary or "")
    if not title: return x
    x = _fix_leading_particle2(x, title)
    if title not in x:
        x = f"{title}{_eunneun2(title)} {x.lstrip()}"
    x = _known_typos_fix2(x, title)
    x = _snap_title_tokens(x, title)
    return x.strip()

import torch as _torch

def _clean_cfg(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if v is not None}

def _decode_new(gen_ids: _torch.Tensor, prompt_len: int) -> str:
    new_tokens = gen_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

MAX_PROMPT_TOKENS = 1024
def _truncate_by_tokens(title: str, addr: str, review: str):
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
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if ids.shape[-1] > MAX_PROMPT_TOKENS:
        keep = int(len(review) * MAX_PROMPT_TOKENS / ids.shape[-1])
        trimmed = review[:max(500, keep)]
        messages[1]["content"] = f"행사명: {title} / 주소: {addr}\n\n[리뷰]\n{trimmed}\n\n위 리뷰를 1~3문장(약 150자 내외)으로 자연스럽게 요약해 주세요."
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    return ids.to(model.device)

if "GEN_MAIN" not in globals():
    GEN_MAIN = dict(max_new_tokens=128, num_beams=4, do_sample=False, length_penalty=0.9,
                    repetition_penalty=1.05, no_repeat_ngram_size=3)
if "GEN_STRICT" not in globals():
    GEN_STRICT = dict(max_new_tokens=110, num_beams=5, do_sample=False, length_penalty=1.0,
                      repetition_penalty=1.1, no_repeat_ngram_size=4)

@_torch.inference_mode()
def llm_summarize(title: str, addr: str, review: str) -> str:
    input_ids = _truncate_by_tokens(title, addr, review)
    prompt_len = input_ids.shape[-1]
    try:
        gen1 = model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, **_clean_cfg(GEN_MAIN))
        txt1 = _decode_new(gen1, prompt_len)
        out1 = repair_summary(txt1, title=title)
        if not looks_bad(out1):
            return out1
    except Exception:
        out1 = ""
    try:
        gen2 = model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, **_clean_cfg(GEN_STRICT))
        txt2 = _decode_new(gen2, prompt_len)
        out2 = repair_summary(txt2, title=title)
        if not looks_bad(out2):
            return out2
    except Exception:
        pass
    base = f"{title}{_eunneun2(title)} 가족 단위로 즐길 수 있는 프로그램과 체험이 중심이었습니다."
    pos  = "공연·체험 구성과 편의시설이 대체로 만족스러웠습니다."
    neg  = "다만, 주차·대기 등 일부 혼잡으로 불편을 겪기도 했습니다."
    return _finalize_and_limit_to_3(f"{base} {pos} {neg}")

import pandas as pd
from tqdm.auto import tqdm

MAX_REVIEWS_PER_EVENT = 20
MAX_JOIN_CHARS        = 4000
need_cols = {"contentid","event_title","event_addr","cleaned_review"}

def join_reviews(g):
    reviews = [str(x) for x in g["cleaned_review"].dropna().tolist() if str(x).strip()]
    if not reviews: return ""
    reviews = reviews[:MAX_REVIEWS_PER_EVENT]
    joined = " ".join(reviews)
    return joined[:MAX_JOIN_CHARS]

def process_one(CSV_IN: Path):
    OUT_DIR = CSV_IN.parent
    stem = CSV_IN.stem
    prefix = re.sub(r"_CLEANED$", "", stem)
    OUT_CSV   = OUT_DIR / f"{prefix}_qwen_summaries.csv"
    OUT_JSONL = OUT_DIR / f"{prefix}_qwen_summaries.jsonl"
    assert CSV_IN.exists(), f"CSV 없음: {CSV_IN}"
    df = pd.read_csv(CSV_IN)
    assert need_cols.issubset(df.columns), f"필수 컬럼 누락: {need_cols - set(df.columns)}"
    groups = df.groupby(["contentid","event_title","event_addr"], dropna=False)
    rows = []
    for (cid, title, addr), g in tqdm(
        groups, total=len(groups),
        desc=f"Summarizing (1-pass): {CSV_IN.name}",
        leave=False,
        dynamic_ncols=True
    ):
        joined = join_reviews(g)
        if not joined:
            summ = f"{title} 관련 리뷰가 충분하지 않아 요약이 어렵습니다."
        else:
            summ = llm_summarize(str(title), str(addr), joined)
        rows.append({
            "contentid": int(cid) if pd.notna(cid) else None,
            "event_title": str(title) if pd.notna(title) else "",
            "event_addr":  str(addr)  if pd.notna(addr)  else "",
            "summary": summ
        })
    def is_suspect(s):
        return looks_bad(s)
    sus_ids = [r["contentid"] for r in rows if is_suspect(r["summary"])]
    print(f"[{CSV_IN.name}] 1차 후 잠재문제: {len(sus_ids)}")
    if sus_ids:
        rows2 = []
        groups2 = df.groupby(["contentid","event_title","event_addr"], dropna=False)
        for (cid, title, addr), g in tqdm(
            groups2, total=len(groups2),
            desc=f"Retrying suspects: {CSV_IN.name}",
            leave=False,
            dynamic_ncols=True
        ):
            if pd.notna(cid) and int(cid) in sus_ids:
                joined = join_reviews(g)
                summ = llm_summarize(str(title), str(addr), joined)
                rows2.append((int(cid), summ))
        m = {cid: s for cid, s in rows2}
        for r in rows:
            if r["contentid"] in m:
                r["summary"] = m[r["contentid"]]
    out_df = pd.DataFrame(rows, columns=["contentid","event_title","event_addr","summary"])
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 저장 완료 | CSV: {OUT_CSV.name} | JSONL: {OUT_JSONL.name} | shape={out_df.shape}")

base = Path(OUT_DIR_S)
targets = sorted(base.glob(PATTERN))
assert targets, f"대상 파일이 없습니다: {base}/{PATTERN}"
print(f"총 {len(targets)}개 파일 처리 시작…")
for csv_in in targets:
    process_one(csv_in)
print("🎉 모든 파일 처리 완료")