# -*- coding: utf-8 -*-
"""
main.py — Realtime Review→Summary Worker (ko 전용, t5_summarize)

- places/**/reviews/** 구독 (isPublic=true, summaryProcessed=true)
- place(=places/{contentId})별 리뷰 5개 버퍼 → 기존 summary + 신규 5개 → t5_summarize.py 로 재요약 → items 문서 업데이트
- "ko"만 업데이트: 같은 contentid가 en/ja/ko에 모두 있어도 **api_data/ko/** 경로만 선택
- 매칭 순서:
  1) FAST(index): items CG에서 contentId / contentid(소문자) / id == {contentId} (str/int 모두 시도) → **ko 경로 우선 선택**
  2) region 포인트 겟: api_data/ko/{region}/{*}/items/{contentId}
  3) ko 전체 스캔:    api_data/ko/**/**/items/{contentId}
  4) 최후 폴백(가능): CG에서 contentId/contentid/id/title(+region)
- 리뷰 6개월 경과 자동 삭제
- 사용 필드/업데이트: summary, summaryUpdatedAt, summary_model, summary_src_hash, usedInSummary

필수 색인:
- items (컬렉션 그룹) 단일: contentid (ASC)
- reviews(컬렉션 그룹) 복합: isPublic (ASC) + summaryProcessed (ASC)
- reviews(컬렉션 그룹) 단일: createdAt (ASC)  # 클린업용
"""
from __future__ import annotations
import os, sys, time, signal, hashlib, threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import DocumentReference

# FieldFilter (경고 없는 where)
try:
    from google.cloud.firestore_v1.base_query import FieldFilter
except Exception:
    FieldFilter = None

# FieldPath (일부 폴백에서 사용 가능)
try:
    from google.cloud.firestore_v1.field_path import FieldPath
except Exception:
    try:
        from google.cloud.firestore import FieldPath
    except Exception:
        FieldPath = None

# ------------------------------
# Config
# ------------------------------
FIREBASE_CRED_PATH = os.environ.get("FIREBASE_CRED_PATH", "/mnt/data/firebase.json")
MIN_BUFFER_SIZE = int(os.environ.get("MIN_BUFFER_SIZE", 5))
CLEANUP_INTERVAL_SEC = int(os.environ.get("CLEANUP_INTERVAL_SEC", 60 * 60))
DELETE_OLDER_THAN = timedelta(days=int(os.environ.get("DELETE_OLDER_THAN_DAYS", 180)))
API_DATA_LANG = "ko"  # ko만 갱신

MATCH_REGION_FIELD = True  # title fallback 시 region 동시 매칭

# ------------------------------
# Import summarizer (NO edits to t5_summarize.py)
# ------------------------------
import importlib.util
SUMMARIZER_PATH = os.environ.get("SUMMARIZER_PATH", "/mnt/data/t5_summarize.py")
if not os.path.exists(SUMMARIZER_PATH):
    print(f"[FATAL] Summarizer file not found: {SUMMARIZER_PATH}")
    sys.exit(1)
spec = importlib.util.spec_from_file_location("t5_summarize", SUMMARIZER_PATH)
t5_sum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5_sum)  # type: ignore

# ------------------------------
# Helpers
# ------------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _to_utc(dt) -> Optional[datetime]:
    try:
        if hasattr(dt, "to_datetime"):
            return dt.to_datetime().replace(tzinfo=timezone.utc)
        if isinstance(dt, datetime):
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return None

# ------------------------------
# Firestore init
# ------------------------------
if not os.path.exists(FIREBASE_CRED_PATH):
    print(f"[FATAL] Firebase credential file not found: {FIREBASE_CRED_PATH}")
    sys.exit(1)
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ------------------------------
# Buffers & Cache
# ------------------------------
_buffers: Dict[str, List[dict]] = {}        # placeDocPath → [{ref,text,createdAt}, ...]
_buffers_lock = threading.Lock()
_item_path_cache: Dict[str, str] = {}       # contentId → item doc path (ko 경로)
CACHE_MAX = 5000

print("[LOAD] Initializing summarizer…")
_summarizer = t5_sum.load_summarizer()
print("[OK] Summarizer ready.")

# ------------------------------
# Item resolvers
# ------------------------------
def _cg_where(colgrp, field, op, val):
    if FieldFilter is not None:
        return colgrp.where(filter=FieldFilter(field, op, val))
    return colgrp.where(field, op, val)  # 경고만 발생

def _resolve_item_ref_fast_with_index(content_id: str) -> Optional[DocumentReference]:
    """
    FAST PATH: items collection_group 단일필드 색인으로 즉시 매칭.
    - contentId / contentid(소문자) / id 에 대해 str/int 모두 시도
    - 동일 id가 다국어에 존재하면 **api_data/ko/** 경로를 우선 선택**
    """
    if not content_id:
        return None

    candidates = [content_id]
    if isinstance(content_id, str) and content_id.isdigit():
        try:
            candidates.append(int(content_id))
        except Exception:
            pass

    prefer_prefix = f"api_data/{API_DATA_LANG}/"

    def _pick_best(hits):
        for h in hits:
            if prefer_prefix in h.reference.path:
                return h
        return hits[0]

    fields = ["contentId", "contentid", "id"]  # contentid 포함
    for fname in fields:
        for val in candidates:
            try:
                q = _cg_where(db.collection_group("items"), fname, "==", val)
                hits = list(q.limit(5).stream())
                if hits:
                    best = _pick_best(hits)
                    print(f"[HIT] CG fast by field {fname} ({type(val).__name__}): {best.reference.path}")
                    return best.reference
            except Exception as e:
                print(f"[INFO] CG fast by '{fname}' failed ({type(val).__name__}): {e}")
    return None

def _resolve_item_ref_region_pointget_by_id(content_id: str, region: Optional[str]) -> Optional[DocumentReference]:
    """api_data/ko/{region}/{*}/items/{contentId} — 카테고리 순회(Point-get, ko만)"""
    if not content_id or not region:
        return None
    base = db.collection("api_data").document(API_DATA_LANG).collection(region)
    try:
        for cat_snap in base.stream():  # foods, stays, tourist attraction, ...
            ref = base.document(cat_snap.id).collection("items").document(content_id)
            snap = ref.get()
            if snap.exists:
                print(f"[HIT] Found by region point-get: {ref.path}")
                return ref
    except Exception as e:
        print(f"[INFO] region point-get failed for region={region}: {e}")
    return None

def _resolve_item_ref_bruteforce_under_ko_by_id(content_id: str) -> Optional[DocumentReference]:
    """api_data/ko/**/**/items/{contentId} (2-depth brute scan, ko만)"""
    if not content_id:
        return None
    ko_doc = db.collection("api_data").document(API_DATA_LANG)
    try:
        for col in ko_doc.collections():     # 지역(대구, 부산, …)
            for d in col.stream():           # 카테고리 문서들(foods 등)
                ref = col.document(d.id).collection("items").document(content_id)
                if ref.get().exists:
                    print(f"[HIT] Found by KO-scan point-get: {ref.path}")
                    return ref
    except Exception as e:
        print(f"[INFO] KO-scan by id failed: {e}")
    return None

def _resolve_item_ref_fallback_collection_group(place_doc: dict, content_id: str) -> Optional[DocumentReference]:
    """최후 폴백(색인 필요 가능): contentId/contentid/id/title(+region) — ko 우선 선택"""
    prefer_prefix = f"api_data/{API_DATA_LANG}/"

    def _pick_best(hits):
        for h in hits:
            if prefer_prefix in h.reference.path:
                return h
        return hits[0]

    # contentId
    try:
        hits = list(_cg_where(db.collection_group("items"), "contentId", "==", content_id).limit(5).stream())
        if hits:
            best = _pick_best(hits)
            print(f"[HIT] CG by field contentId: {best.reference.path}")
            return best.reference
    except Exception as e:
        print(f"[INFO] CG 'contentId' failed: {e}")
    # contentid (소문자)
    try:
        hits = list(_cg_where(db.collection_group("items"), "contentid", "==", content_id).limit(5).stream())
        if hits:
            best = _pick_best(hits)
            print(f"[HIT] CG by field contentid: {best.reference.path}")
            return best.reference
    except Exception as e:
        print(f"[INFO] CG 'contentid' failed: {e}")
    # id
    try:
        hits = list(_cg_where(db.collection_group("items"), "id", "==", content_id).limit(5).stream())
        if hits:
            best = _pick_best(hits)
            print(f"[HIT] CG by field id: {best.reference.path}")
            return best.reference
    except Exception as e:
        print(f"[INFO] CG 'id' failed: {e}")

    # title(+region)
    title = place_doc.get("placeName")
    region = place_doc.get("region")
    if title:
        q = _cg_where(db.collection_group("items"), "title", "==", title)
        if MATCH_REGION_FIELD and region:
            q = q.where(filter=FieldFilter("region", "==", region)) if FieldFilter else q.where("region", "==", region)
        try:
            hits = list(q.limit(5).stream())
            if hits:
                best = _pick_best(hits)
                print(f"[HIT] CG by title/region: {best.reference.path}")
                return best.reference
        except Exception as e:
            print(f"[INFO] CG title/region failed: {e}")
    return None

def _resolve_item_ref(place_ref: DocumentReference, place_doc: dict) -> Optional[DocumentReference]:
    """캐시 → FAST(ko 우선) → region point-get(ko) → ko-scan → 폴백"""
    cid = place_ref.id

    cached = _item_path_cache.get(cid)
    if cached:
        ref = db.document(cached)
        if ref.get().exists:
            print(f"[HIT] cache: {cached}")
            return ref
        _item_path_cache.pop(cid, None)

    ref = _resolve_item_ref_fast_with_index(cid)
    if ref:
        if len(_item_path_cache) >= CACHE_MAX: _item_path_cache.clear()
        _item_path_cache[cid] = ref.path
        return ref

    region = place_doc.get("region")
    ref = _resolve_item_ref_region_pointget_by_id(cid, region)
    if ref:
        if len(_item_path_cache) >= CACHE_MAX: _item_path_cache.clear()
        _item_path_cache[cid] = ref.path
        return ref

    ref = _resolve_item_ref_bruteforce_under_ko_by_id(cid)
    if ref:
        if len(_item_path_cache) >= CACHE_MAX: _item_path_cache.clear()
        _item_path_cache[cid] = ref.path
        return ref

    ref = _resolve_item_ref_fallback_collection_group(place_doc, cid)
    if ref:
        if len(_item_path_cache) >= CACHE_MAX: _item_path_cache.clear()
        _item_path_cache[cid] = ref.path
        return ref

    return None

# ------------------------------
# Core processing
# ------------------------------
def _process_if_ready(place_ref: DocumentReference, place_doc: dict):
    place_path = place_ref.path
    with _buffers_lock:
        entries = list(_buffers.get(place_path, []))
        count = len(entries)
    print(f"[WAIT] {place_path} ({count}/{MIN_BUFFER_SIZE})")
    if count < MIN_BUFFER_SIZE:
        return

    entries.sort(key=lambda e: e.get("createdAt") or _now_utc())
    selected = entries[:MIN_BUFFER_SIZE]
    remaining = entries[MIN_BUFFER_SIZE:]

    item_ref = _resolve_item_ref(place_ref, place_doc)
    if not item_ref:
        print(f"[MISS] item not found for contentId={place_ref.id} "
              f"title={place_doc.get('placeName')} region={place_doc.get('region')}")
        with _buffers_lock:
            _buffers[place_path] = remaining
        return

    item_doc = item_ref.get().to_dict() or {}
    existing_summary = (item_doc.get("summary") or "").strip()

    text_block = existing_summary + ("\n\n" if existing_summary else "")
    text_block += "\n".join([
        (e.get("text") or "").strip()
        for e in selected
        if isinstance(e.get("text"), str) and (e.get("text") or "").strip()
    ])
    if not text_block.strip():
        print(f"[SKIP] Empty text after merge for {item_ref.path}")
        with _buffers_lock:
            _buffers[place_path] = remaining
        return

    t0 = time.time()
    summarized = t5_sum.summarize(text_block, _summarizer)
    elapsed = time.time() - t0

    update_payload = {
        "summary": summarized,
        "summaryUpdatedAt": firestore.SERVER_TIMESTAMP,
        "summary_model": getattr(t5_sum, "MODEL_ID", "unknown"),
        "summary_src_hash": _sha1(text_block),
    }
    item_ref.set(update_payload, merge=True)

    batch = db.batch()
    mark = {
        "usedInSummary": True,
        "usedAt": firestore.SERVER_TIMESTAMP,
        "usedSummaryHash": update_payload["summary_src_hash"],
    }
    for e in selected:
        ref: DocumentReference = e["ref"]
        batch.set(ref, mark, merge=True)
    batch.commit()

    with _buffers_lock:
        _buffers[place_path] = remaining

    print(f"[OK] Updated {item_ref.path} in {elapsed:.2f}s (batch={len(selected)})")

# ------------------------------
# Snapshot callback
# ------------------------------
def _on_reviews_snapshot(col_snapshot, changes, read_time):
    for change in changes:
        if change.type.name != "ADDED":
            continue
        doc = change.document
        data = doc.to_dict() or {}

        if not data.get("isPublic", False):
            continue
        if not data.get("summaryProcessed", False):
            continue
        if data.get("usedInSummary", False):
            continue

        created_dt = _to_utc(data.get("createdAt"))
        if created_dt and (_now_utc() - created_dt) > DELETE_OLDER_THAN:
            print(f"[CLEANUP] Deleting stale review {doc.reference.path}")
            try:
                doc.reference.delete()
            except Exception as e:
                print(f"[WARN] Delete failed: {e}")
            continue

        review_text = (data.get("review") or "").strip()
        if not review_text:
            continue

        place_ref = doc.reference.parent.parent  # places/{contentId}
        if not place_ref:
            continue
        place_doc = place_ref.get().to_dict() or {}

        # place 필드 먼저, 이후 리뷰 필드로 덮어써서 최신 맥락 우선
        merged_place_ctx = dict(place_doc)
        merged_place_ctx.update(data)
        print(f"[CTX] cid={place_ref.id} region={merged_place_ctx.get('region')} title={merged_place_ctx.get('placeName')}")

        entry = {"ref": doc.reference, "text": review_text, "createdAt": created_dt or _now_utc()}

        place_path = place_ref.path
        with _buffers_lock:
            _buffers.setdefault(place_path, []).append(entry)
            count = len(_buffers[place_path])
        print(f"[BUF] {place_path} ({count}/{MIN_BUFFER_SIZE}) +1")

        _process_if_ready(place_ref, merged_place_ctx)

# ------------------------------
# Cleanup thread
# ------------------------------
def _cleanup_task(stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            cutoff = _now_utc() - DELETE_OLDER_THAN
            if FieldFilter:
                q = db.collection_group("reviews").where(filter=FieldFilter("createdAt", "<", cutoff))
            else:
                q = db.collection_group("reviews").where("createdAt", "<", cutoff)
            to_delete = list(q.stream())
            if to_delete:
                print(f"[CLEANUP] Found {len(to_delete)} stale reviews (< {cutoff.isoformat()})")
            for snap in to_delete:
                try:
                    snap.reference.delete()
                except Exception as e:
                    print(f"[WARN] Cleanup delete failed: {e}")
        except Exception as e:
            print(f"[WARN] Cleanup task error: {e}")
        stop_event.wait(CLEANUP_INTERVAL_SEC)

# ------------------------------
# Main
# ------------------------------
def main():
    print("[START] Subscribing to collection group: places/**/reviews/**")
    reviews_cg = db.collection_group("reviews")
    if FieldFilter:
        query = (reviews_cg
                 .where(filter=FieldFilter("isPublic", "==", True))
                 .where(filter=FieldFilter("summaryProcessed", "==", True)))
    else:
        query = reviews_cg.where("isPublic", "==", True).where("summaryProcessed", "==", True)

    stop_event = threading.Event()
    cleanup_thread = threading.Thread(target=_cleanup_task, args=(stop_event,), daemon=False)
    cleanup_thread.start()

    listen = query.on_snapshot(_on_reviews_snapshot)

    def _graceful_stop(*_):
        stop_event.set()
    try:
        signal.signal(signal.SIGINT, _graceful_stop)
        signal.signal(signal.SIGTERM, _graceful_stop)
    except Exception:
        pass

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        print("\n[STOP] Shutting down…")
        try:
            listen.unsubscribe()
        except Exception:
            pass
        stop_event.set()
        try:
            cleanup_thread.join(timeout=10)
        except Exception:
            pass
        time.sleep(0.5)
        try:
            firebase_admin.delete_app(firebase_admin.get_app())
        except Exception:
            pass

if __name__ == "__main__":
    main()