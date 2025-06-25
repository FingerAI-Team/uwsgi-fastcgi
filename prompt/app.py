from flask import Flask, request, jsonify, Response, stream_with_context
import os
import json
import requests
import logging
from datetime import datetime
from typing import Dict, Any
import traceback
from services.rag_chat_service import RagChatService

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/prompt/app.log") if os.path.exists("/var/log/prompt") else logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("prompt-backend")

# Flask 앱 초기화
app = Flask(__name__)
app.json.ensure_ascii = False  # 한글 인코딩 처리

config_path = os.environ.get("PROMPT_CONFIG", "/prompt/config.json")

# 환경 변수 설정
RAG_ENDPOINT = os.environ.get("RAG_ENDPOINT", "http://nginx/rag")
RERANKER_ENDPOINT = os.environ.get("RERANKER_ENDPOINT", "http://nginx/reranker")
OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://ollama:11434")
MEMORY_DIR = os.environ.get("MEMORY_DIR", "./memory")

# 전역 RAG 챗봇 서비스 인스턴스
rag_chat_service = None

class AgentService:
    def __init__(self, config_path:str=None):
        """
        Initialize the prompt Agent service
        
        Args:
            config_path: Path to config file, if None, use default settings
        """
        try:
            logger.debug("Loading configuration...")
            self.config = self._load_config(config_path)
            self.default_model = self.config.get("default_model")
            self.search_top = self.config.get("search_top")
            self.rerank_top = self.config.get("rerank_top")
            self.rerank_threshold = self.config.get("rerank_threshold")
        
            logger.info(f"Initializing Agent LLM with model: {self.default_model}")
            logger.debug(f"RAG Search Top {self.search_top}")
            logger.debug(f"Reranking Top {self.rerank_top}")
            logger.debug(f"Reranking Threshold {self.rerank_threshold}")
        except Exception as e:
            logger.error(f"Failed to initialize AgentService: {str(e)}")
            raise
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "search_top": int(os.getenv("RAG_SEARCH_TOP_K", "100")),
            "rerank_top": int(os.getenv("RERANKER_TOP_K", "20")),
            "default_model": os.getenv("DEFAULT_MODEL", "gemma3:12b"),
            "rerank_threshold": float(os.getenv("RERANK_THRESHOLD", "0.1"))
        }
        
        if not config_path:
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return default_config
        
    # 프롬프트 템플릿 로드 함수
    @staticmethod
    def load_prompt_template(template_name):
        template_path = os.path.join(os.path.dirname(__file__), "templates", f"{template_name}.txt")
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"템플릿 파일을 찾을 수 없습니다: {template_path}")
            return None


# 상태 확인 API
@app.route("/prompt/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "prompt-backend"
    })

# 문서 검색 및 요약 API
@app.route("/prompt/summarize", methods=["POST"])
def summarize():
    try:
        data = request.json
        logger.info(f"요청 받음: {json.dumps(data, ensure_ascii=False)}")
        query = data.get("query")
        
        if not query:
            logger.error("쿼리 누락")
            return jsonify({"error": "쿼리가 필요합니다"}), 400
        
        summaryAgent = AgentService(config_path)
        logger.info(f"Agent 초기화 완료: search_top={summaryAgent.search_top}, rerank_top={summaryAgent.rerank_top}")
            
        # 1. RAG 서비스 호출하여 문서 검색
        logger.info(f"RAG 서비스 호출 준비: endpoint={RAG_ENDPOINT}/search")
        search_params = {
            "query_text": query,
            "top_k": summaryAgent.search_top,
            "domains": []  # 기본 빈 도메인 리스트
        }
        
        # 추가 검색 매개변수
        if "domain" in data:  # 단일 도메인 지원
            search_params["domains"] = [data["domain"]]
        elif "domains" in data:  # 복수 도메인 지원
            search_params["domains"] = data["domains"]
            
        for param in ["author", "start_date", "end_date", "title", "info_filter", "tags_filter"]:
            if param in data:
                search_params[param] = data[param]
                logger.info(f"추가 검색 파라미터: {param}={data[param]}")
        
        # curl 형식의 API 호출 로깅
        curl_command = f'''curl -X POST "{RAG_ENDPOINT}/search" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(search_params, ensure_ascii=False)}\''''
        logger.info(f"RAG API curl 형식: {curl_command}")
        
        logger.info(f"RAG 검색 요청: params={json.dumps(search_params, ensure_ascii=False)}")
        search_response = requests.post(f"{RAG_ENDPOINT}/search", json=search_params)
        
        logger.info(f"RAG 응답 코드: {search_response.status_code}")
        if search_response.status_code != 200:
            logger.error(f"RAG 검색 오류 응답: {search_response.text}")
            return jsonify({"error": "문서 검색 중 오류가 발생했습니다"}), 500
            
        search_results = search_response.json()
        logger.info("=== RAG 검색 결과 ===")
        logger.info(f"검색된 문서 수: {len(search_results.get('search_result', []))}")
        logger.info(f"RAG 응답 결과: {json.dumps(search_results, ensure_ascii=False, indent=2)}")
        for idx, doc in enumerate(search_results.get("search_result", []), 1):
            logger.info(f"문서 {idx}:")
            logger.info(f"제목: {doc.get('title', '제목 없음')}")
            logger.info(f"내용: {doc.get('text', '')[:100]}...")
            logger.info(f"점수: {doc.get('score', 'N/A')}")
            logger.info("---")
        
        # 재랭킹 준비
        logger.info("재랭킹 준비 시작")
        
        # 재랭킹에 사용할 변수 초기화
        rerank_passages = []
        original_results_by_id = {}
        
        # 검색 결과 전처리 (필요한 필드만 추출)
        for item in search_results.get("search_result", []):
            # 문서 ID 기반으로 원본 결과 저장 (재랭킹 후 매핑용)
            doc_id = item.get("doc_id")
            if doc_id:
                original_results_by_id[doc_id] = item
                
            # 재랭킹에 필요한 필드만 포함
            rerank_item = {
                "id": item.get("id") or f"{doc_id}_{item.get('passage_id', 0)}",
                "text": item.get("text", ""),
                "doc_id": doc_id,
                "passage_id": item.get("passage_id"),
                "score": item.get("score", 0)
            }
            
            # 기타 중요 필드 복사
            for field in ["title", "author", "domain", "info", "tags"]:
                if field in item:
                    rerank_item[field] = item[field]
            
            rerank_passages.append(rerank_item)
        
        # 2. Reranker 서비스 호출
        logger.info(f"Reranker 서비스 호출 준비: endpoint={RERANKER_ENDPOINT}/rerank")
        rerank_data = {
            "query": query,
            "results": search_results.get("search_result", [])
        }
        
        # curl 형식의 API 호출 로깅
        curl_command = f'''curl -X POST "{RERANKER_ENDPOINT}/rerank?top_k={summaryAgent.rerank_top}" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(rerank_data, ensure_ascii=False)}\''''
        logger.info(f"Reranker API curl 형식: {curl_command}")
        
        logger.info(f"Reranker 요청: top_k={summaryAgent.rerank_top}")
        rerank_response = requests.post(
            f"{RERANKER_ENDPOINT}/rerank",
            params={"top_k": summaryAgent.rerank_top},
            json=rerank_data
        )
        
        logger.info(f"Reranker 응답 코드: {rerank_response.status_code}")
        if rerank_response.status_code != 200:
            logger.error(f"Reranker 오류 응답: {rerank_response.text}")
            return jsonify({"error": "문서 재순위화 중 오류가 발생했습니다"}), 500
            
        reranked_results = rerank_response.json()
        logger.info("=== Reranker 결과 ===")
        logger.info(f"재순위화된 문서 수: {len(reranked_results['results'])}")
        logger.info(f"Reranker 응답 결과: {json.dumps(reranked_results, ensure_ascii=False, indent=2)}")
        for idx, doc in enumerate(reranked_results["results"][:summaryAgent.rerank_top], 1):
            logger.info(f"재순위화된 문서 {idx}:")
            logger.info(f"제목: {doc.get('title', '제목 없음')}")
            logger.info(f"내용: {doc.get('text', '')[:100]}...")
            logger.info(f"점수: {doc.get('score', 'N/A')}")
            logger.info("---")
        
        # 3. 프롬프트 템플릿 준비
        logger.info("프롬프트 템플릿 로드 시작")
        template = summaryAgent.load_prompt_template("summarize")
        
        if not template:
            logger.error("프롬프트 템플릿 로드 실패")
            return jsonify({"error": "프롬프트 템플릿을 로드할 수 없습니다"}), 500
            
        # 컨텍스트 생성
        logger.info(f"컨텍스트 생성 시작 (문서 수: {len(reranked_results['results'][:summaryAgent.rerank_top])})")
        context = ""
        for idx, doc in enumerate(reranked_results['results'][:summaryAgent.rerank_top], 1):
            context += f"[문서 {idx}]\n"
            context += f"제목: {doc.get('title', '제목 없음')}\n"
            context += f"내용: {doc.get('text', '')}\n\n"
        
        # 최종 프롬프트 생성
        final_prompt = template.format(
            query=query,
            context=context
        )
        logger.info(f"최종 프롬프트 길이: {len(final_prompt)} 문자")
        logger.info("=== 최종 프롬프트 내용 ===")
        logger.info(f"{final_prompt}")
        logger.info("========================")
        
        # 4. Ollama API 호출
        logger.info(f"Ollama API 호출 준비: endpoint={OLLAMA_ENDPOINT}, model={summaryAgent.default_model}")
        try:
            logger.info("Ollama 요청 시작")
            ollama_response = requests.post(
                f"{OLLAMA_ENDPOINT}/api/generate",
                json={
                    "model": summaryAgent.default_model,
                    "prompt": final_prompt,
                    "stream": False
                },
                timeout=120
            )
            
            logger.info(f"Ollama 응답 코드: {ollama_response.status_code}")
            if ollama_response.status_code != 200:
                logger.error(f"Ollama API 오류 응답: {ollama_response.text}")
                return jsonify({
                    "error": "LLM 요청 중 오류가 발생했습니다",
                    "details": ollama_response.text
                }), 500
                
            summary = ollama_response.json().get("response", "")
            logger.info("=== 최종 요약 결과 ===")
            logger.info(f"쿼리: {query}")
            logger.info(f"요약 길이: {len(summary)} 문자")
            logger.info(f"요약 내용: {summary}")
            logger.info("=== 처리 완료 ===")
            
            return jsonify({
                "query": query,
                "summary": summary,
                "documents_count": len(reranked_results['results']),
                "prompt_length": len(final_prompt)
            })
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 서비스 연결 오류: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Ollama 서비스에 연결할 수 없습니다",
                "details": str(e)
            }), 503
        
    except Exception as e:
        logger.error(f"처리 중 예외 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# 향상된 검색 API (RAG+Reranker)
@app.route("/prompt/enhanced_search", methods=["POST"])
def enhanced_search():
    try:
        # 로그 설정
        log_dir = "/var/log/prompt" if os.path.exists("/var/log/prompt") else "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 기본 로거 설정
        logger = logging.getLogger("enhanced-search")
        logger.setLevel(logging.INFO)
        
        # 이전 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(f"{log_dir}/enhanced_search.log")
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)
        
        logger.info("=== 새로운 /prompt/enhanced_search 요청 시작 ===")
        
        # 메타데이터 로거는 기본 로거와 동일하게 사용
        metadata_logger = logger
        
        # 요청 시작 시간 기록
        start_time = datetime.now()
        
        data = request.json
        logger.info(f"향상된 검색 요청 받음: {json.dumps(data, ensure_ascii=False)}")
        
        # 필수 파라미터 확인
        query = data.get("query")
        if not query:
            logger.error("쿼리 누락")
            return jsonify({"error": "쿼리가 필요합니다"}), 400
        
        summaryAgent = AgentService(config_path)
        
        # 사용자 지정 파라미터 또는 기본값 사용
        top_m = data.get("top_m", summaryAgent.search_top)  # RAG 검색 결과 수
        top_n = data.get("top_n", summaryAgent.rerank_top)  # Reranker 결과 수
        threshold = data.get("threshold", summaryAgent.rerank_threshold)  # Reranker 점수 임계치
        mrc_weight = data.get("mrc_weight", 0.7)  # MRC 가중치 (기본값 0.7)
        
        # 파라미터 유효성 검사
        if top_m < top_n:
            logger.warning(f"파라미터 오류: top_m({top_m}) < top_n({top_n}), top_m으로 조정합니다")
            top_n = top_m
            
        logger.info(f"검색 파라미터: query='{query}', top_m={top_m}, top_n={top_n}, threshold={threshold}, mrc_weight={mrc_weight}")
            
        # 1. RAG 서비스 호출하여 문서 검색
        logger.info(f"RAG 서비스 호출 준비: endpoint={RAG_ENDPOINT}/search")
        search_params = {
            "query_text": query,
            "top_k": top_m,
            "domains": []  # 기본 빈 도메인 리스트
        }
        
        # 추가 검색 매개변수
        if "domain" in data:  # 단일 도메인 지원
            search_params["domains"] = [data["domain"]]
        elif "domains" in data:  # 복수 도메인 지원
            search_params["domains"] = data["domains"]
            
        for param in ["author", "start_date", "end_date", "title", "info_filter", "tags_filter"]:
            if param in data:
                search_params[param] = data[param]
                logger.info(f"추가 검색 파라미터: {param}={data[param]}")
        
        logger.info(f"RAG 검색 요청: params={json.dumps(search_params, ensure_ascii=False)}")
        
        # RAG 요청 시작 시간
        rag_start_time = datetime.now()
        search_response = requests.post(f"{RAG_ENDPOINT}/search", json=search_params)
        rag_time = (datetime.now() - rag_start_time).total_seconds()
        
        logger.info(f"RAG 응답 코드: {search_response.status_code}, 소요 시간: {rag_time:.3f}초")
        
        if search_response.status_code != 200:
            logger.error(f"RAG 검색 오류 응답: {search_response.text}")
            return jsonify({"error": "문서 검색 중 오류가 발생했습니다"}), 500
            
        search_results = search_response.json()
        logger.info(f"RAG 검색 결과 수: {len(search_results.get('search_result', []))}")
        
        # 검색 결과가 없는 경우
        if not search_results.get("search_result"):
            logger.warning("검색 결과가 없습니다")
            return jsonify({
                "query": query,
                "top_m": top_m,
                "top_n": top_n,
                "search_count": 0,
                "reranked_count": 0,
                "results": []
            })
        
        # 재랭킹 준비
        logger.info("재랭킹 준비 시작")
        
        # 재랭킹에 사용할 변수 초기화
        rerank_passages = []
        original_results_by_id = {}
        
        # 검색 결과 전처리 (필요한 필드만 추출)
        for item in search_results.get("search_result", []):
            # 문서 ID 기반으로 원본 결과 저장 (재랭킹 후 매핑용)
            doc_id = item.get("doc_id")
            if doc_id:
                original_results_by_id[doc_id] = item
                
            # 재랭킹에 필요한 필드만 포함
            rerank_item = {
                "id": item.get("id") or f"{doc_id}_{item.get('passage_id', 0)}",
                "text": item.get("text", ""),
                "doc_id": doc_id,
                "passage_id": item.get("passage_id"),
                "score": item.get("score", 0)
            }
            
            # 기타 중요 필드 복사
            for field in ["title", "author", "domain", "info", "tags"]:
                if field in item:
                    rerank_item[field] = item[field]
            
            rerank_passages.append(rerank_item)
        
        # 재랭킹 수행
        try:
            logger.info(f"하이브리드 재랭킹 요청 시작: 쿼리='{query}', 결과 수={len(rerank_passages)}개")
            rerank_start_time = datetime.now()
            
            # 재랭킹 요청 준비
            rerank_payload = {
                "query": query,
                "results": rerank_passages,
                "total": len(rerank_passages),
                "reranked": False
            }
            
            # 재랭킹 파라미터
            rerank_params = {
                "top_k": int(top_n),  # 상위 N개 결과만 요청 (정수형으로 변환)
                "mrc_weight": mrc_weight  # MRC 가중치 전달
            }
            
            # 재랭킹 요청 수행
            try:
                logger.info(f"Reranker 서비스 호출: {len(rerank_passages)}개 결과, top_k={top_n}, mrc_weight={mrc_weight}")
                rerank_response = requests.post(
                    f"{RERANKER_ENDPOINT}/rerank",
                    json=rerank_payload,
                    params=rerank_params,
                    timeout=60
                )
                
                # 응답 처리
                if rerank_response.status_code == 200:
                    reranked_results = rerank_response.json()
                    logger.info(f"재랭킹 응답 성공: {len(reranked_results.get('results', []))}개 결과")
                    
                    # 디버깅: 첫 번째 결과의 점수 확인
                    if "results" in reranked_results and reranked_results["results"]:
                        first_result = reranked_results["results"][0]
                        logger.info(f"[DEBUG] 첫 번째 결과 점수: {first_result.get('score', 0)}")
                        
                        # FlashRank 초기화 여부 확인 (flashrank_score 필드 존재 여부)
                        if "flashrank_score" in first_result:
                            flashrank_score = first_result.get("flashrank_score", 0)
                            logger.info(f"[FLASHRANK-INIT-CHECK] 첫 번째 결과의 FlashRank 점수: {flashrank_score}")
                        
                        # MRC 초기화 여부 확인 (mrc_score 필드 존재 여부)
                        if "mrc_score" in first_result:
                            mrc_score = first_result.get("mrc_score", 0)
                            logger.info(f"[MRC-INIT-CHECK] 첫 번째 결과의 MRC 점수: {mrc_score}")
                        
                        # 하이브리드 점수 확인
                        if "hybrid_score" in first_result:
                            hybrid_score = first_result.get("hybrid_score", 0)
                            logger.info(f"[HYBRID-SCORE-CHECK] 첫 번째 결과의 하이브리드 점수: {hybrid_score}")
                else:
                    logger.error(f"재랭킹 응답 실패: 상태 코드={rerank_response.status_code}, 응답={rerank_response.text}")
                    reranked_results = {
                        "query": query,
                        "results": search_results.get("search_result", []),
                        "total": len(search_results.get("search_result", [])),
                        "reranked": False,
                        "reranker_type": "none",
                        "error": f"재랭킹 응답 실패: {rerank_response.status_code}"
                    }
            except Exception as e:
                logger.error(f"재랭킹 서비스 요청 실패: {str(e)}", exc_info=True)
                reranked_results = {
                    "query": query,
                    "results": search_results.get("search_result", []),
                    "total": len(search_results.get("search_result", [])),
                    "reranked": False,
                    "reranker_type": "none",
                    "error": f"재랭킹 서비스 요청 실패: {str(e)}"
                }
            
            # 재랭킹 시간 계산
            rerank_time = (datetime.now() - rerank_start_time).total_seconds()
            logger.info(f"재랭킹 완료: {rerank_time:.3f}초")
        except Exception as e:
            logger.error(f"재랭킹 처리 중 예외 발생: {str(e)}", exc_info=True)
            # 예외 발생 시 원본 결과 사용
            reranked_results = {
                "query": query,
                "results": search_results.get("search_result", []),
                "total": len(search_results.get("search_result", [])),
                "reranked": False,
                "reranker_type": "none",
                "error": f"재랭킹 처리 중 예외 발생: {str(e)}"
            }
            rerank_time = 0
        
        # 3. 결과 처리 및 응답 포맷팅
        processed_results = []
        logger.info("결과 처리 및 응답 포맷팅 시작")

        # 재랭킹 직후의 결과 수 저장 (임계치 필터링 전)
        reranked_count = len(reranked_results.get("results", []))
        logger.info(f"재랭킹 직후 결과 수: {reranked_count}")

        # Reranker 결과 처리
        for idx, item in enumerate(reranked_results.get("results", [])):
            # 점수 확인
            rerank_score = item.get("score", 0)
            
            # 임계치 필터링
            if rerank_score < threshold:
                logger.info(f"임계치({threshold}) 미만 결과 필터링: doc_id={item.get('doc_id', 'unknown')}, score={rerank_score}")
                continue
            
            # 결과 아이템 초기화
            result_item = {}
            
            # 1. 원본 검색 결과의 모든 필드 복사 (있는 경우)
            doc_id = item.get("doc_id")
            original_item = None
            if doc_id and doc_id in original_results_by_id:
                original_item = original_results_by_id[doc_id]
                # 원본 검색 결과의 모든 필드 복사 (점수 관련 필드 제외)
                for key, value in original_item.items():
                    if key not in ["score", "flashrank_score", "mrc_score", "hybrid_score"]:
                        result_item[key] = value
            
            # 2. 재랭킹 결과의 필드 복사 (원본 덮어쓰기, 점수 관련 필드 제외)
            for key, value in item.items():
                if key not in ["score", "flashrank_score", "mrc_score", "hybrid_score", "rerank_score"]:
                    result_item[key] = value
            
            # 3. 점수 정보 설정 - 중복 제거하고 명확하게 구분
            # 기본 점수 설정
            result_item["score"] = item.get("score", 0)  # 기본 점수
            result_item["rerank_position"] = idx
            
            # 원본 점수 정보 (있는 경우)
            if original_item and "score" in original_item:
                result_item["original_score"] = original_item["score"]
            
            # 재랭킹 점수 정보 복사 (있는 경우에만)
            for score_field in ["hybrid_score", "flashrank_score", "mrc_score", "mrc_answer", "mrc_char_ids"]:
                if score_field in item:
                    result_item[score_field] = item[score_field]
            
            # MRC 관련 필드가 있는지 확인하고 없으면 기본값 설정
            if "mrc_score" in result_item and "mrc_answer" not in result_item:
                result_item["mrc_answer"] = ""
            if "mrc_score" in result_item and "mrc_char_ids" not in result_item:
                result_item["mrc_char_ids"] = []
            
            # 최종 결과에 추가
            processed_results.append(result_item)
        
        # 총 처리 시간 계산
        total_time = (datetime.now() - start_time).total_seconds()
        
        # 최종 응답 구성
        response = {
            "query": query,
            "top_m": top_m,
            "top_n": top_n,
            "threshold": threshold,
            "search_count": len(search_results.get("search_result", [])),
            "reranked_count": reranked_count,  # 임계치 필터링 전 결과 수
            "filtered_count": len(processed_results),  # 임계치 필터링 후 결과 수
            "mrc_weight": mrc_weight,  # 사용된 MRC 가중치
            "processing_time": {
                "total": total_time,
                "rag_search": rag_time,
                "reranking": rerank_time
            },
            "results": processed_results,
            "reranked": reranked_results.get("reranked", True),  # 재랭커에서 반환된 재랭킹 여부
            "reranker_type": reranked_results.get("reranker_type", "hybrid")  # 사용된 재랭킹 타입
        }
        
        # 도메인별 결과가 있으면 포함
        if "domain_results" in search_results:
            response["domain_results"] = search_results["domain_results"]
        
        logger.info(f"최종 응답: {len(processed_results)}개 결과, 총 처리 시간: {total_time:.3f}초")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"처리 중 오류가 발생했습니다: {str(e)}"}), 500

# 챗봇 API - 단순 질의응답용
@app.route("/prompt/chat", methods=["POST"])
def chat():
    try:
        summaryAgent = AgentService(config_path)
        data = request.json
        query = data.get("query")
        model = data.get("model", summaryAgent.default_model)
        stream = data.get("stream", False)  # 스트리밍 모드 기본값 False
        
        if not query:
            return jsonify({"error": "질문이 필요합니다"}), 400
        
        # 프롬프트 템플릿 없이 사용자 쿼리 직접 사용
        logger.info(f"Ollama API 챗봇 호출 시작: {model}, 스트리밍 모드: {stream}, 쿼리 직접 전달")
        
        try:
            # 스트리밍 모드에 따라 다른 처리
            if stream:
                # 스트리밍 모드로 처리
                def generate():
                    # 응답 누적을 위한 변수
                    accumulated_response = ""
                    
                    # heartbeat 카운터 초기화
                    heartbeat_counter = 0
                    
                    # heartbeat 메시지 전송 (15초마다)
                    def should_send_heartbeat():
                        nonlocal heartbeat_counter
                        heartbeat_counter += 1
                        return heartbeat_counter % 15 == 0
                    
                    with requests.post(
                        f"{OLLAMA_ENDPOINT}/api/generate",
                        json={
                            "model": model,
                            "prompt": query,
                            "stream": True
                        },
                        timeout=60,
                        stream=True
                    ) as ollama_response:
                        if ollama_response.status_code != 200:
                            logger.error(f"Ollama API 오류: {ollama_response.text}")
                            error_response = {
                                "query": query,
                                "model": model,
                                "error": "LLM 요청 중 오류가 발생했습니다",
                                "details": ollama_response.text
                            }
                            yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
                            return
                        
                        # SSE 형식으로 응답 전송
                        for line in ollama_response.iter_lines():
                            # heartbeat 전송
                            if should_send_heartbeat():
                                yield ":\n\n"  # SSE 주석 형식의 heartbeat
                            
                            if line:
                                try:
                                    response_chunk = json.loads(line)
                                    chunk_text = response_chunk.get("response", "")
                                    if chunk_text:
                                        # 응답 누적
                                        accumulated_response += chunk_text
                                        
                                        # 기존 API 응답 형식으로 구성
                                        stream_response = {
                                            "query": query,
                                            "model": model,
                                            "response": accumulated_response,
                                            "streaming": True
                                        }
                                        
                                        # SSE 형식으로 전송
                                        yield f"data: {json.dumps(stream_response, ensure_ascii=False)}\n\n"
                                except json.JSONDecodeError:
                                    logger.error(f"JSON 디코딩 오류: {line}")
                                    continue
                        
                        # 스트림 종료 응답
                        final_response = {
                            "query": query,
                            "model": model,
                            "response": accumulated_response,
                            "streaming": False,
                            "done": True
                        }
                        yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"
                
                # 스트리밍 응답 헤더 설정 및 반환
                response = Response(stream_with_context(generate()), mimetype='text/event-stream')
                response.headers['Cache-Control'] = 'no-cache, no-transform'
                response.headers['X-Accel-Buffering'] = 'no'  # nginx에서 버퍼링 방지
                response.headers['Connection'] = 'keep-alive'  # 연결 유지
                return response
            else:
                # 기존 방식대로 처리 (스트리밍 없음)
                ollama_response = requests.post(
                    f"{OLLAMA_ENDPOINT}/api/generate",
                    json={
                        "model": model,
                        "prompt": query,  # 사용자 쿼리를 직접 전달
                        "stream": False
                    },
                    timeout=60
                )
                
                if ollama_response.status_code != 200:
                    logger.error(f"Ollama API 오류: {ollama_response.text}")
                    return jsonify({
                        "error": "LLM 요청 중 오류가 발생했습니다",
                        "details": ollama_response.text
                    }), 500
                    
                response_text = ollama_response.json().get("response", "")
                
                return jsonify({
                    "query": query,
                    "model": model,
                    "response": response_text
                })
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 서비스 연결 오류: {str(e)}")
            return jsonify({
                "error": "Ollama 서비스에 연결할 수 없습니다",
                "details": str(e)
            }), 503
        
    except Exception as e:
        logger.error(f"챗봇 처리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Ollama 모델 목록 API
@app.route("/prompt/models", methods=["GET"])
def list_models():
    logger.info(f"💬 OLLAMA_ENDPOINT = {OLLAMA_ENDPOINT}")
    logger.info(f"💬 최종 요청 URL = {OLLAMA_ENDPOINT}/api/tags")
    try:
        # Ollama API 호출하여 모델 목록 가져오기
        logger.info("Ollama 모델 목록 요청")
        try:
            models_response = requests.get(
                f"{OLLAMA_ENDPOINT}/api/tags",
                timeout=10
            )
            summaryAgent = AgentService(config_path)
            
            if models_response.status_code != 200:
                logger.error(f"Ollama API 모델 목록 오류: {models_response.text}")
                return jsonify({
                    "error": "모델 목록을 가져오는 중 오류가 발생했습니다",
                    "details": models_response.text
                }), 500
                
            models_data = models_response.json()
            models = [model.get("name") for model in models_data.get("models", [])]
            
            return jsonify({
                "models": models,
                "default_model": summaryAgent.default_model,
                "total": len(models)
            })
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 서비스 연결 오류: {str(e)}")
            logger.exception(e)  # 전체 traceback도 로그에 남기기
            return jsonify({
                "error": "Ollama 서비스에 연결할 수 없습니다",
                "details": str(e)
            }), 503
            
    except Exception as e:
        logger.error(f"모델 목록 처리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# RAG 챗봇 API - 파일 기반 세션 관리 멀티턴 대화
@app.route("/prompt/chatbot", methods=["POST"])
def chatbot():
    global rag_chat_service
    
    try:
        # 요청 데이터 파싱
        data = request.json
        query = data.get("query")
        session_id = data.get("session_id")
        stream = data.get("stream", False)
        model = data.get("model")
        
        # 필수 파라미터 검증
        if not query:
            return jsonify({"error": "질문이 필요합니다"}), 400
        if not session_id:
            return jsonify({"error": "세션 ID가 필요합니다"}), 400
        
        logger.info(f"RAG 챗봇 요청: 세션={session_id}, 쿼리='{query}', 스트리밍={stream}")
        
        # RAG 챗봇 서비스 초기화 (필요한 경우)
        if rag_chat_service is None:
            summaryAgent = AgentService(config_path)
            default_model = model or summaryAgent.default_model
            search_top = summaryAgent.search_top
            rerank_top = summaryAgent.rerank_top
            rerank_threshold = summaryAgent.rerank_threshold
            
            logger.info(f"RAG 챗봇 서비스 초기화: model={default_model}, search_top={search_top}, rerank_top={rerank_top}")
            rag_chat_service = RagChatService(
                memory_dir=MEMORY_DIR,
                ollama_endpoint=OLLAMA_ENDPOINT,
                rag_endpoint=RAG_ENDPOINT,
                reranker_endpoint=RERANKER_ENDPOINT,
                default_model=default_model,
                search_top=search_top,
                rerank_top=rerank_top,
                rerank_threshold=rerank_threshold
            )
        
        # 추가 검색 파라미터 추출
        search_params = {}
        for param in ["domains", "domain", "author", "start_date", "end_date", "title", "info_filter", "tags_filter"]:
            if param in data:
                search_params[param] = data[param]
        
        # 스트리밍 모드에 따라 다른 처리
        if stream:
            def generate():
                for chunk in rag_chat_service.generate_response(
                    session_id, query, model=model, stream=True, **search_params
                ):
                    yield f"data: {chunk}\n\n"
            
            response = Response(stream_with_context(generate()), mimetype='text/event-stream')
            response.headers['Cache-Control'] = 'no-cache, no-transform'
            response.headers['X-Accel-Buffering'] = 'no'  # nginx에서 버퍼링 방지
            response.headers['Connection'] = 'keep-alive'  # 연결 유지
            return response
        else:
            # 비스트리밍 모드
            result = rag_chat_service.generate_response(
                session_id, query, model=model, stream=False, **search_params
            )
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"RAG 챗봇 처리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# 세션 관리 API - 세션 초기화
@app.route("/prompt/session/clear", methods=["POST"])
def clear_session():
    global rag_chat_service
    
    try:
        data = request.json
        session_id = data.get("session_id")
        
        if not session_id:
            return jsonify({"error": "세션 ID가 필요합니다"}), 400
        
        # RAG 챗봇 서비스 초기화 (필요한 경우)
        if rag_chat_service is None:
            _initialize_rag_service()
        
        result = rag_chat_service.clear_session(session_id)
        logger.info(f"세션 초기화: {session_id}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"세션 초기화 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# 세션 관리 API - 만료된 세션 정리
@app.route("/prompt/session/cleanup", methods=["POST"])
def cleanup_sessions():
    global rag_chat_service
    
    try:
        # RAG 챗봇 서비스 초기화 (필요한 경우)
        if rag_chat_service is None:
            _initialize_rag_service()
        
        result = rag_chat_service.cleanup_expired_sessions()
        logger.info("만료된 세션 정리 완료")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"만료된 세션 정리 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# 세션 관리 API - 세션 목록 조회
@app.route("/prompt/session/list", methods=["GET"])
def list_sessions():
    global rag_chat_service
    
    try:
        # RAG 챗봇 서비스 초기화 (필요한 경우)
        if rag_chat_service is None:
            _initialize_rag_service()
        
        session_ids = rag_chat_service.session_manager.get_all_sessions()
        return jsonify({"sessions": session_ids, "count": len(session_ids)})
    
    except Exception as e:
        logger.error(f"세션 목록 조회 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# 세션 관리 API - 세션 내용 조회
@app.route("/prompt/session/get", methods=["GET"])
def get_session():
    global rag_chat_service
    
    try:
        session_id = request.args.get("session_id")
        
        if not session_id:
            return jsonify({"error": "세션 ID가 필요합니다"}), 400
        
        # RAG 챗봇 서비스 초기화 (필요한 경우)
        if rag_chat_service is None:
            _initialize_rag_service()
        
        # 세션 데이터 로드
        session_data = rag_chat_service.session_manager.load_session(session_id)
        
        # 응답 데이터 구성
        response_data = {
            "session_id": session_id,
            "history": session_data.get("history", []),
            "created_at": session_data.get("created_at"),
            "last_updated": session_data.get("last_updated"),
            "message_count": len(session_data.get("history", [])),
            "exists": rag_chat_service.session_manager.session_exists(session_id)
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"세션 내용 조회 중 오류 발생: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# 서비스 초기화 헬퍼 함수
def _initialize_rag_service():
    global rag_chat_service
    
    summaryAgent = AgentService(config_path)
    default_model = summaryAgent.default_model
    search_top = summaryAgent.search_top
    rerank_top = summaryAgent.rerank_top
    rerank_threshold = summaryAgent.rerank_threshold
    
    logger.info(f"RAG 챗봇 서비스 초기화: model={default_model}, search_top={search_top}, rerank_top={rerank_top}")
    rag_chat_service = RagChatService(
        memory_dir=MEMORY_DIR,
        ollama_endpoint=OLLAMA_ENDPOINT,
        rag_endpoint=RAG_ENDPOINT,
        reranker_endpoint=RERANKER_ENDPOINT,
        default_model=default_model,
        search_top=search_top,
        rerank_top=rerank_top,
        rerank_threshold=rerank_threshold
    )

if __name__ == "__main__":
    # 개발 환경에서만 사용
    app.run(host="0.0.0.0", port=5000, debug=True) 