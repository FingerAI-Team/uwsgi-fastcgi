import io
import os
import json
import sys
from tqdm import tqdm
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")

# 한글 → 영문 키 매핑
KEY_MAP = {
    "판시사항": "holding",
    "참조판례": "referenced_cases",
    "사건종류명": "case_type_name",
    "판결요지": "summary_of_decision",
    "참조조문": "referenced_statutes",
    "선고일자": "decision_date",
    "법원명": "court_name",
    "판례내용": "full_text",
    "사건번호": "case_number",
    "사건종류코드": "case_type_code",
    "판례정보일련번호": "precedent_id",
    "선고": "judgement_result",
    "판결유형": "decision_type",
    "법원종류코드": "court_type_code"
}

# 설정
MAX_SIZE = 200 * 1024 * 1024  # 200MB
today = datetime.now().strftime("%Y%m%d")
output_dir = "/home/files/result/prec"
os.makedirs(output_dir, exist_ok=True)

# 처리할 디렉터리들
target_dirs = ["/home/files/detail/prec", "/home/files/detail/daily_prec"]

# 결과 초기화
result = []
current_size = 0
file_index = 1


def save_result_json(data_list, index):
    out_path = os.path.join(output_dir, f"{today}_prec_{index}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    open(out_path + ".fin", "w").close()

    # 처리된 파일 삭제
    for entry in data_list:
        source = entry.get("source_path")
        if source and os.path.exists(source):
            os.remove(source)


def normalize_date(raw_date):
    digits = ''.join(filter(str.isdigit, str(raw_date)))
    if len(digits) != 8:
        digits = digits.zfill(8)
    try:
        dt = datetime.strptime(digits, "%Y%m%d")
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    except Exception:
        return "0000-00-00"


# 모든 타겟 디렉터리의 파일을 순회
all_files = []
for target_dir in target_dirs:
    for file_name in os.listdir(target_dir):
        full_path = os.path.join(target_dir, file_name)
        if os.path.isfile(full_path):
            all_files.append(full_path)

# tqdm 적용
for file_path in tqdm(all_files, desc="processing...", total=len(all_files)):
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    item = json_data.get("PrecService", {})
    full_text = item.get("판례내용", "")
    case_title = item.get("사건명", "")

    info = {}
    doc_id = None

    for kor_key, eng_key in KEY_MAP.items():
        value = item.get(kor_key)
        if value is not None:
            info[eng_key] = value
            if kor_key == "판례정보일련번호":
                doc_id = f"{value}"

    if doc_id is None:
        continue

    # 날짜 포맷 보정
    raw_date = item.get("선고일자", "")
    formatted_date = normalize_date(raw_date)

    info["create_date"] = formatted_date
    info["title_org"] = case_title
    tags = {"create_date": formatted_date}

    tail = full_text[-4000:]
    dot_index = tail.find(".")
    trimmed_text = tail[dot_index + 1:].lstrip() if dot_index != -1 else tail

    json_result = {
        "doc_id": doc_id,
        "title": case_title,
        "text": trimmed_text,
        "info": info,
        "tags": tags,
        "source_path": file_path  # 삭제를 위한 경로 포함
    }

    estimated_size = len(json.dumps(json_result, ensure_ascii=False).encode("utf-8"))

    if current_size + estimated_size > MAX_SIZE:
        save_result_json(result, file_index)
        file_index += 1
        result = []
        current_size = 0

    result.append(json_result)
    current_size += estimated_size

# 마지막 저장 + 삭제
if result:
    save_result_json(result, file_index)

