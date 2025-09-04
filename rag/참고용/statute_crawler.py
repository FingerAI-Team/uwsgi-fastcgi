########## IMPORT MODULES ##########
import os
import sys
import time
import json
import logging
import requests
import pandas as pd
from math import ceil
from tqdm import tqdm
from config.config import CONFIG
from datetime import date, datetime, timedelta

########## CONFIG ##########
oc = CONFIG.OC # OpenAPI 서비스 제공자
target = "law" # OpenAPI 서비스 대상
return_type = "json" # OpenAPI 서비스 응답 형식
target_id = "법령일련번호" # 현행 법령 목록에서 사용할 ID 필드
log_dir = CONFIG.LOG_DIR # 로그 파일 저장 디렉토리
list_url = CONFIG.LIST_URL # 현행 법령 목록 조회 URL
detail_url = CONFIG.DETAIL_URL # 현행 법령 본문 조회 URL
statute_list_dir = CONFIG.STATUTE_LIST_DIR # 현행 법령 목록 저장 디렉토리
statute_detail_dir = CONFIG.STATUTE_DETAIL_DIR # 현행 법령 본문 저장 디렉토리
daily_statute_list_dir = CONFIG.DAILY_STATUTE_LIST_DIR # 현행 법령 일별 목록 저장 디렉토리
daily_statute_detail_dir = CONFIG.DAILY_STATUTE_DETAIL_DIR # 현행 법령 일별 본문 저장 디렉토리

########## LOGGING ##########
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=os.path.join(log_dir, f'{date.today().strftime("%Y%m%d")}.log'),
    filemode='a'
)

########## FUNCTIONS ##########
def list_crawler(url, params, file_nm, efYd=None):
    "현행 법령 목록 크롤러"
    if efYd:
        csv_file_dir = daily_statute_list_dir
        temp = datetime.strptime(efYd, "%Y%m%d")
        efYd_f = temp - timedelta(days=30)
        efYd_f = efYd_f.strftime("%Y%m%d")
        params["efYd"] = f"{efYd_f}~{efYd}"
    else:
        csv_file_dir = statute_list_dir
    
    try:
        # 현행 법령 목록 조회 후 개수 확인
        logging.info(f"Starting to gather statute list from {url}")
        res = requests.get(url=url, params=params)
        res_json = res.json()
        res.raise_for_status()
        total_cnt = int(res_json["LawSearch"]["totalCnt"])
        current_cnt = int(res_json["LawSearch"]["numOfRows"])
    except Exception as e:
        logging.error(f"Error occurred while fetching data from {params}: {e}")
        sys.exit(1)
    
    df = pd.DataFrame()
    # 현행 법령 목록이 여러 페이지로 나뉘어져 있을 경우 페이지별로 조회
    if total_cnt > current_cnt:
        total_pg = ceil(total_cnt / current_cnt)
        logging.info(f"Total records: {total_cnt}, Total pages: {total_pg}")

        for i in tqdm(range(1, total_pg + 1), desc="Gathering data", total=total_pg):
            try:
                params["page"] = i
                res = requests.get(url=url, params=params)
                res.raise_for_status()
                res_json = res.json()
                data = res_json["LawSearch"]["law"]
                if isinstance(data, list):
                    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
                elif isinstance(data, dict):
                    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
            except Exception as e:
                logging.error(f"Error occurred while fetching data from {url} on page {i}: {e}")
                sys.exit(1)
    # 현행 법령 목록이 한 페이지에 모두 있는 경우
    else:
        data = res_json["LawSearch"]["law"]
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])

    csv_path = os.path.join(csv_file_dir, f"{file_nm}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved statute list to {csv_path}")
    return df

def detail_crawler(mst_al, file_prefix, save_dir):
    "현행 법령 본문 크롤러"
    for i, mst in tqdm(enumerate(mst_al, start=1), desc="Gathering details", total=len(mst_al)):
        params = {
            "OC": oc,
            "target": target,
            "type": return_type,
            "MST": mst
        }
        
        success = False
        res = None

        # 3회 시도하여 현행 법령 본문 조회
        for attempt in range(1, 4):
            try:
                res = requests.get(detail_url, params=params)
                res.raise_for_status()
                res_json = res.json()
                res_json["법령"]["기본정보"]["법령일련번호"] = mst
                
                file_path = os.path.join(save_dir, f"{file_prefix}_{i:06d}.json")
                with open (file_path, "w", encoding="utf-8") as f:
                    json.dump(res_json, f, ensure_ascii=False, indent=4)
                
                success = True
                break
            except Exception as e:
                logging.warning(f"[Attempt {attempt}/3] Failed to fetch ID {id}: {e}")
                time.sleep(3)

        # 3회 시도 후에도 실패한 경우 에러 로그 기록
        if not success:
            logging.error(
                f"Failed to fetch statute details for ID {id} after 3 attempts. "
                f"Last response: {res.text if res else 'No response'}"
            )

        # 1000개마다 5초 대기
        if i % 1000 == 0:
            logging.info(f"Processed {i} records. Sleeping for 5 seconds...")
            time.sleep(5)

########## MAIN SCRIPT ##########
if __name__ == "__main__":
    # 현행 법령 목록 조회
    params = {
        "OC": oc,
        "target": target,
        "type": return_type,
        "display": 100
    }

    # efYd 파라미터가 주어지면 해당 시행일 기준으로 조회
    # efYd 파라미터가 없으면 전체 법령 목록 조회
    efYd = sys.argv[1] if len(sys.argv) > 1 else None
    timestamp = time.time()
    today_str = date.today().strftime("%Y%m%d")

    if efYd:
        file_nm = f"daily_statute_{today_str}-{int(timestamp)}"
        json_file_dir = daily_statute_detail_dir
        logging.info(f"Gathering statute from {file_nm} and 30 days prior.")
        df = list_crawler(list_url, params, file_nm, efYd)
    else:
        file_nm = f"statute_{today_str}-{int(timestamp)}"
        json_file_dir = statute_detail_dir
        logging.info("Gathering all statute dataset.")
        df = list_crawler(list_url, params, file_nm)

    # 현행 법령 목록이 비어있으면 종료
    if df.empty:
        logging.error("No statute found.")
        sys.exit(0)
    # 현행 법령 목록이 비어있지 않으면 본문 조회 시작
    else:
        logging.info(f"Statute list gathered. Total records: {len(df)}")
        mst_al = df[target_id]
        
        try:
            detail_crawler(mst_al, file_nm, json_file_dir)
        except Exception as e:
            logging.error(f"Unexpected error while processing details: {e}")
            sys.exit(1)

        logging.info("Statute details gathered.")
        logging.info("Finished statute crawler.")

