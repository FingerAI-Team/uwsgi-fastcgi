import json
import hashlib
import os
import re
from datetime import datetime
from tqdm import tqdm

MAX_SIZE = 200 * 1024 * 1024  # 200MB
OUTPUT_DIR = "/home/files/result/statute"
TARGET_DIRS = ["/home/files/detail/statute", "/home/files/detail/daily_statute"]
today = datetime.now().strftime("%Y%m%d")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 한글 → 영문 키 매핑
KEY_MAP = {
    "법령명_한글": "law_name_korean",
    "법령명_한자": "law_name_hanja",
    "법령명약칭": "law_name_abbrev",
    "이전법령명": "previous_law_name",
    "법령ID": "law_id",
    "공포번호": "promulgation_number",
    "공포일자": "promulgation_date",
    "시행일자": "effective_date",
    "별표시행일자문자열": "annex_effective_date_string",
    "제개정구분": "revision_type",
    "제명변경여부": "title_changed",
    "공동부령정보": "joint_ministerial_info",
    "공포법령여부": "is_promulgated_law",
    "한글법령여부": "is_korean_law",
    "별표편집여부": "annex_editable",
    "편장절관": "structure_code",
    "언어": "language",
    "전화번호": "phone",
    "소관부처": "supervising_ministry",
    "소관부처코드": "supervising_ministry_code",
    "법종구분": "law_category",
    "법종구분코드": "law_category_code",
    "연락부서": "contact_department",
    "부서연락처": "dept_phone",
    "부서키": "dept_key",
    "부서명": "dept_name",
    "소관부처명": "dept_ministry_name",
    "소관부처코드": "dept_ministry_code",
    "법령일련번호": "mst",
}

def extract_text_recursive(node):
    text = ""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (str, int, float)):
                if key.endswith("내용") or isinstance(value, str):
                    text += str(value).strip() + "\n"
            elif isinstance(value, (list, dict)):
                text += extract_text_recursive(value)
    elif isinstance(node, list):
        for item in node:
            text += extract_text_recursive(item)
    elif isinstance(node, (str, int, float)):
        text += str(node).strip() + "\n"
    return text


def truncate_to_sentence(text, limit=4000):
    if len(text) <= limit:
        return text
    cutoff = text[:limit]
    matches = list(re.finditer(r"[.!?](?=\s)", cutoff))
    if matches:
        last = matches[-1].end()
        return cutoff[:last].strip()
    else:
        return cutoff.strip()


def normalize_date(raw_date):
    digits = ''.join(filter(str.isdigit, str(raw_date)))
    if len(digits) != 8:
        digits = digits.zfill(8)
    try:
        dt = datetime.strptime(digits, "%Y%m%d")
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    except Exception:
        return "0000-00-00"


def generate_doc_id(law_id, doc_type):
    return f"{law_id}_{doc_type}"

def save_chunk(chunk, chunk_index):
    # 원본 경로들 미리 저장
    src_paths = [item.get("source_path") for item in chunk if item.get("source_path")]

    # source_path 제거 후 저장
    for item in chunk:
        item.pop("source_path", None)

    out_path = os.path.join(OUTPUT_DIR, f"{today}_statute_{chunk_index}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    open(out_path + ".fin", "w").close()

    # 처리된 원본 삭제
    for src in src_paths:
        if os.path.exists(src):
            os.remove(src)

def process_law_json(law_json, file_path):
    output = []
    law = law_json.get("법령", {})
    info = law.get("기본정보", {}) or {}

    # 기본정보 한글 → 영문 변환
    mapped_info = {}
    for k, v in info.items():
        if isinstance(v, dict):
            # 중첩 구조 평탄화
            for sub_k, sub_v in v.items():
                eng_key = KEY_MAP.get(sub_k)
                if eng_key:
                    mapped_info[eng_key] = sub_v
        else:
            eng_key = KEY_MAP.get(k)
            if eng_key:
                mapped_info[eng_key] = v

    law_title = info.get("법령명_한글", "Untitled Law")
    law_id = info.get("법령ID", "Untitled_Law")
    create_date = normalize_date(info.get("시행일자", "00000000"))

    def add_doc(doc_type, content, extra_info=None):
        text = extract_text_recursive(content).strip()
        if not text:
            return
        doc = {
            "doc_id": generate_doc_id(law_id, doc_type),
            "title": f"{law_title} - {doc_type}",
            "text": truncate_to_sentence(text) if doc_type == "Article" else text,
            # 기본정보 + create_date/type 병합
            "info": {**mapped_info, "create_date": create_date, "type": doc_type, "title_org": law_title},
            "tags": {"create_date": create_date},
            "source_path": file_path
        }
        if doc_type == "Article" and extra_info:
            doc["info"].update(extra_info)
            doc["info"]["full_text"] = text
        output.append(doc)

    if "개정문" in law:
        add_doc("Amendment", law["개정문"])

    if "부칙" in law:
        add_doc("Supplementary", law["부칙"])

    if "제개정이유" in law:
        add_doc("Reason", law["제개정이유"])

    if "조문" in law and isinstance(law["조문"].get("조문단위"), (dict, list)):
        units = law["조문"]["조문단위"]
        if isinstance(units, dict):
            units = [units]

        full_text = ""
        article_info = []

        for article in units:
            if isinstance(article, dict):
                text = extract_text_recursive(article).strip()
                article_title = article.get("조문제목", "")
                article_number = article.get("조문번호", "")
                if text:
                    full_text += f"{article_title} {article_number}\n{text}\n\n"
                    article_info.append({
                        "article_number": article_number,
                        "article_title": article_title
                    })

        if full_text:
            add_doc("Article", full_text, {
                "article_count": len(article_info),
                "article_info": article_info
            })

    return output


def gather_all_files(dirs):
    file_list = []
    for path in dirs:
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(".json"):
                    full_path = os.path.join(path, file)
                    if os.path.isfile(full_path):
                        file_list.append(full_path)
    return file_list


def process_all():
    file_list = gather_all_files(TARGET_DIRS)
    print(f"[INFO] Total input files: {len(file_list)}")

    seen_hashes = set()
    buffer = []
    buffer_size = 0
    chunk_index = 1

    for file_path in tqdm(file_list, desc="processing...", total=len(file_list)):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode failed: {file_path} → {e}")
            continue

        if "법령" not in data:
            print(f"[WARN] Missing '법령': {file_path}")
            continue

        docs = process_law_json(data, file_path=file_path)
        for item in docs:
            hash_key = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
            if hash_key in seen_hashes:
                continue
            seen_hashes.add(hash_key)

            size_estimate = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            if buffer_size + size_estimate > MAX_SIZE:
                save_chunk(buffer, chunk_index)
                chunk_index += 1
                buffer = []
                buffer_size = 0

            buffer.append(item)
            buffer_size += size_estimate

    if buffer:
        save_chunk(buffer, chunk_index)


if __name__ == "__main__":
    process_all()

