import json
import hashlib
import os
import re
from datetime import datetime
from tqdm import tqdm
from config.config import CONFIG

MAX_SIZE = 200 * 1024 * 1024  # 200MB
OUTPUT_DIR = "/home/files/result/admrul"
TARGET_DIRS = ["/home/files/detail/admrul", "/home/files/detail/daily_admrul"]
today = datetime.now().strftime("%Y%m%d")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 행정규칙 기본정보 한글 → 영문 키 매핑
RULE_KEY_MAP = {
    "현행여부": "is_current",
    "담당부서기관명": "dept_org_name",
    "담당자명": "contact_person",
    "행정규칙종류코드": "rule_type_code",
    "행정규칙명": "rule_name",
    "발령일자": "issue_date",
    "행정규칙종류": "rule_type",
    "제개정구분코드": "revision_type_code",
    "조문형식여부": "has_article_format",
    "소관부처명": "supervising_ministry_name",
    "전화번호": "phone",
    "제개정구분명": "revision_type_name",
    "소관부처코드": "supervising_ministry_code",
    "생성일자": "created_date",
    "행정규칙ID": "rule_id",
    "시행일자": "effective_date",
    "담당부서기관코드": "dept_org_code",
    "발령번호": "issue_number",
    "행정규칙일련번호": "rule_serial_number",
    "상위부처명": "upper_ministry_name"
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


def generate_doc_id(rule_id, doc_type):
    return f"{rule_id}_{doc_type}"


def save_chunk(chunk, chunk_index):
    out_path = os.path.join(OUTPUT_DIR, f"{today}_admrul_{chunk_index}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    open(out_path + ".fin", "w").close()

    # 처리 완료된 파일 삭제
    for item in chunk:
        source = item.get("source_path")
        if source and os.path.exists(source):
            os.remove(source)


def process_rule_json(rule_json, file_path="(unknown_file)"):
    output = []
    rule = rule_json.get("AdmRulService", {})
    info = rule.get("행정규칙기본정보", {}) or {}

    # 기본정보 한글 → 영문 변환
    mapped_info = {}
    for k, v in info.items():
        eng_key = RULE_KEY_MAP.get(k)
        if eng_key:
            mapped_info[eng_key] = v

    rule_title = info.get("행정규칙명", "Untitled Rule")
    rule_id = info.get("행정규칙ID", "Untitled_Rule")
    create_date = normalize_date(info.get("발령일자", ""))

    def add_doc(doc_type, raw_content):
        if not raw_content:
            return
        if isinstance(raw_content, str):
            full_text = raw_content.strip()
        else:
            full_text = extract_text_recursive(raw_content).strip()
        if not full_text:
            return
        short_text = truncate_to_sentence(full_text) if doc_type == "Article" else full_text

        # info에 행정규칙기본정보(영문) 병합
        info_data = {**mapped_info, "title_org": rule_title, "create_date": create_date, "type": doc_type}
        if doc_type == "Article":
            info_data["full_text"] = full_text

        output.append({
            "doc_id": generate_doc_id(rule_id, doc_type),
            "title": f"{rule_title} - {doc_type}",
            "text": short_text,
            "info": info_data,
            "tags": {"create_date": create_date},
            "source_path": file_path
        })

    add_doc("Article", rule.get("조문내용"))
    add_doc("Supplementary", rule.get("부칙"))
    add_doc("Reason", rule.get("제개정이유"))

    return output

def gather_all_files(target_dirs):
    file_paths = []
    for d in target_dirs:
        if os.path.isdir(d):
            for file in os.listdir(d):
                if file.endswith(".json"):
                    full_path = os.path.join(d, file)
                    if os.path.isfile(full_path):
                        file_paths.append(full_path)
    return file_paths


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

        if "AdmRulService" not in data:
            print(f"[WARN] Missing 'AdmRulService': {file_path}")
            continue

        docs = process_rule_json(data, file_path=file_path)
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

