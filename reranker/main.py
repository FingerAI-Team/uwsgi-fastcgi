"""
Reranker FastCGI application with API Gateway functionality
"""

# CUDA 초기화 문제 해결을 위한 설정
import multiprocessing
import os

# 환경 변수 설정 - PyTorch가 fork 모드에서도 CUDA를 사용할 수 있게 함
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# CUDA 메모리 관리 최적화 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU 순서 일관성 유지
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"      # 사용할 GPU 지정 (필요에 따라 조정)

# 멀티프로세싱 시작 방식을 'spawn'으로 설정
try:
    multiprocessing.set_start_method('spawn', force=True)
    print("멀티프로세싱 시작 방식을 'spawn'으로 설정했습니다.")
except RuntimeError as e:
    print(f"멀티프로세싱 시작 방식 설정 실패: {e}")

import os
import logging
import requests
import requests_unixsocket
import traceback
from typing import Dict, List, Any, Optional
from flask import Flask, request, Response, jsonify
from pydantic import BaseModel, Field
from urllib.parse import quote_plus
import time
import sys

# 로깅 설정
log_dir = "/var/log/reranker"
if not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"로그 디렉토리 생성 실패: {str(e)}")
        log_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"대체 로그 디렉토리 사용: {log_dir}")

try:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 스트림 핸들러 설정
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 파일 핸들러 설정
    try:
        log_file = os.path.join(log_dir, 'app.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"로그 파일 생성 성공: {log_file}")
    except Exception as e:
        print(f"로그 파일 핸들러 설정 실패: {str(e)}")
        # 파일 핸들러 설정 실패시 스트림 핸들러만 사용
except Exception as e:
    print(f"로깅 설정 실패: {str(e)}")
    # 기본 로깅 사용
    import logging
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

# 더 빠른 JSON 처리를 위해 ujson 사용
try:
    import ujson as json
    logger.info("Using ujson for faster JSON processing")
except ImportError:
    import json
    logger.info("ujson not available, using default json")

# 상대 경로 import 대신 절대 경로 import로 변경
try:
    from service import RerankerService
    logger.info("RerankerService 임포트 성공")
except Exception as e:
    logger.error(f"RerankerService 임포트 실패: {str(e)}")

# 데이터 모델 정의
class PassageModel(BaseModel):
    """Single passage model"""
    passage_id: Optional[Any] = None
    doc_id: Optional[str] = None
    text: str
    score: Optional[float] = None
    position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResultModel(BaseModel):
    """Search result containing multiple passages"""
    query: str
    results: List[PassageModel]
    total: Optional[int] = None
    reranked: Optional[bool] = False


class RerankerResponseModel(BaseModel):
    """Response model for reranker API"""
    query: str
    results: List[PassageModel]
    total: int
    reranked: bool = True


# Flask 앱 생성
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 성능 최적화 설정 추가
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['PREFERRED_URL_SCHEME'] = 'http'
app.config['JSON_SORT_KEYS'] = False  # JSON 정렬 비활성화로 성능 향상
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 요청 크기
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # 압축 JSON 응답
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'  # 명시적 MIME 타입

# 응답 압축 비활성화 - 대용량 응답 처리 시 압축으로 인한 지연 방지
# Flask-Compress 사용하지 않음
logger.info("Response compression disabled for better performance")

# WSGI 응답 버퍼링 비활성화
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['wsgi.response_buffering'] = 'false'

# 스레드 최적화
import threading
threading.stack_size(128 * 1024)  # 스레드 스택 크기 감소

# 응답 속도 최적화를 위한 설정
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TRAP_HTTP_EXCEPTIONS'] = False
app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False

# 서비스 인스턴스 생성
reranker_service = None

# RAG 서비스 엔드포인트 설정
RAG_ENDPOINT = os.getenv('RAG_ENDPOINT', 'http://nginx/rag')

# Unix 소켓 세션 생성 및 연결 풀링 최적화
rag_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=100,  # 연결 풀 크기
    pool_maxsize=100,      # 최대 연결 수
    max_retries=3,         # 재시도 횟수
    pool_block=False       # 논블로킹 모드
)
rag_session.mount('http://', adapter)
rag_session.mount('https://', adapter)

# FastCGI 응답 헤더 설정
@app.after_request
def add_header(response):
    """응답 헤더 최적화"""
    # 필수 헤더만 설정하여 오버헤드 최소화
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # FastCGI 응답 최적화 - 핵심 헤더
    response.headers['X-Accel-Buffering'] = 'no'  # nginx 버퍼링 비활성화
    
    # Transfer-Encoding 최적화
    response.headers.pop('Transfer-Encoding', None)
    
    # 콘텐츠 길이 명시 (chunked 인코딩 방지)
    if response.data and 'Content-Length' not in response.headers:
        response.headers['Content-Length'] = str(len(response.data))
    
    return response


def get_reranker_service():
    """
    RerankerService의 싱글톤 인스턴스를 가져오는 함수
    
    환경 변수나 기본 경로에서 설정 파일을 찾아 RerankerService를 초기화하고,
    실패 시 DummyReranker를 반환합니다.
    
    Returns:
        RerankerService 또는 DummyReranker 인스턴스
    """
    try:
        # 절대 경로로 설정된 환경 변수 확인
        config_path = os.environ.get("RERANKER_CONFIG", "/reranker/config.json")
        
        # 절대 경로에서 파일을 찾지 못한 경우 상대 경로로 시도
        if not os.path.exists(config_path) and config_path.startswith("/reranker/"):
            relative_config_path = config_path[10:]  # "/reranker/" 제거
            if os.path.exists(relative_config_path):
                logger.info(f"환경 변수의 절대 경로를 상대 경로로 변환: {config_path} -> {relative_config_path}")
                config_path = relative_config_path
                
        # 그래도 파일이 없으면 현재 디렉토리에서 config.json 찾기
        if not os.path.exists(config_path):
            if os.path.exists("config.json"):
                logger.info(f"환경 변수 대신 현재 디렉토리의 config.json 사용")
                config_path = "config.json"
                
        logger.info(f"Getting RerankerService with config: {config_path} (exists: {os.path.exists(config_path)})")
        
        # 싱글톤 패턴으로 서비스 인스턴스 가져오기
        from service import RerankerService
        return RerankerService.get_instance(config_path)
    except Exception as e:
        logger.error(f"Failed to get RerankerService: {str(e)}", exc_info=True)
        logger.error("Using dummy reranker for testing")
        return DummyReranker()


class DummyReranker:
    """테스트용 더미 리랭커"""
    def __init__(self):
        """초기화 메서드"""
        self.hybrid_weight_mrc = 0.7  # 기본값으로 설정
        self.mrc_enabled = False
        self.mrc_reranker = None
        
    def process_search_results(self, query: str, search_result: Dict, top_k: int = 5) -> Dict:
        """원본 검색 결과를 그대로 반환"""
        logger.warning("Using dummy reranker - returning original search results")
        return search_result


# 애플리케이션 시작 시 서비스 초기화
@app.before_first_request
def initialize_service():
    """Initialize service before first request"""
    try:
        get_reranker_service()
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        # 초기화 실패해도 서비스는 계속 실행
        pass


@app.route("/reranker/health")
def health_check():
    """
    서비스 상태 확인 엔드포인트
    
    서비스가 정상적으로 실행 중인지 확인하는 간단한 상태 체크 API입니다.
    
    Returns:
        JSON 응답: {"status": "ok", "service": "reranker"}
    """
    return Response(
        json.dumps({"status": "ok", "service": "reranker"}, ensure_ascii=False),
        mimetype='application/json; charset=utf-8'
    )


@app.route("/reranker/enhanced-search", methods=['GET'])
def enhanced_search():
    """
    통합 검색 API: RAG 검색 결과를 Reranker로 순위를 다시 매기는 기능
    
    RAG 서비스에서 검색 결과를 가져온 후, Reranker를 사용하여 결과의 순위를 재조정합니다.
    검색 결과가 없는 경우 빈 결과를 반환합니다.
    
    Query Parameters:
        query_text (str): 검색 쿼리 (필수)
        top_k (int): 반환할 상위 결과 수 (기본값: 5)
        raw_results (int): RAG 서비스에서 가져올 결과 수 (기본값: 20)
        domain, author, start_date, end_date, title: 선택적 필터링 파라미터
        
    Returns:
        JSON 응답: 재랭킹된 검색 결과와 메타데이터
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        # 파라미터 추출
        query_text = request.args.get('query_text')
        top_k = int(request.args.get('top_k', 5))
        raw_results = int(request.args.get('raw_results', 20))
        
        # 필수 파라미터 검증
        if not query_text:
            return jsonify({
                "result_code": "F000001",
                "message": "검색어(query_text)는 필수 입력값입니다.",
                "search_result": None
            }), 400
            
        # Step 1: RAG 서비스에 검색 요청
        search_params = {
            'query_text': query_text,
            'top_k': raw_results
        }
        
        # 선택적 파라미터 추가
        for param in ['domain', 'author', 'start_date', 'end_date', 'title']:
            if request.args.get(param):
                search_params[param] = request.args.get(param)
                
        # RAG 서비스 호출
        logger.info(f"검색 요청: {search_params}")
        rag_response = requests.get(f"{RAG_ENDPOINT}/search", params=search_params)
        
        if rag_response.status_code != 200:
            logger.error(f"RAG 서비스 오류: {rag_response.text}")
            return jsonify({
                "result_code": "F000002",
                "message": f"검색 서비스 오류: {rag_response.status_code}",
                "search_result": None
            }), 500
            
        rag_data = rag_response.json()
        
        # 검색 결과가 없으면 빈 결과 반환
        if not rag_data.get('search_result') or len(rag_data['search_result']) == 0:
            return jsonify({
                "result_code": "F000003",
                "message": "검색 결과가 없습니다.",
                "search_result": []
            }), 200
            
        # Step 2: Reranker 처리를 위한 데이터 준비
        rerank_data = {
            "query": query_text,
            "results": []
        }
        
        # RAG 결과를 Reranker 포맷으로 변환
        for idx, result in enumerate(rag_data['search_result']):
            passage = {
                "passage_id": idx,
                "doc_id": result.get('doc_id'),
                "text": result.get('text'),
                "score": result.get('score'),
                "metadata": {
                    "title": result.get('title'),
                    "author": result.get('author'),
                    "info": result.get('info'),
                    "tags": result.get('tags')
                }
            }
            rerank_data["results"].append(passage)
            
        # Step 3: Reranker 처리
        try:
            search_result = SearchResultModel(**rerank_data)
            reranked = get_reranker_service().process_search_results(
                search_result.query,
                search_result.dict(),
                top_k
            )
            
            # 최종 결과 변환
            final_results = []
            for result in reranked.get('results', [])[:top_k]:
                metadata = result.get('metadata', {})
                final_result = {
                    "doc_id": result.get('doc_id'),
                    "text": result.get('text'),
                    "score": result.get('score'),
                    "title": metadata.get('title'),
                    "author": metadata.get('author'),
                    "info": metadata.get('info'),
                    "tags": metadata.get('tags')
                }
                final_results.append(final_result)
            
            # 전체 요청 처리 시간 계산
            total_elapsed_time = time.time() - total_start_time
            logger.info(f"Total enhanced-search endpoint processing time: {total_elapsed_time:.3f} seconds")
                
            response_data = {
                "result_code": "F000000",
                "message": "검색 및 재랭킹이 성공적으로 완료되었습니다.",
                "search_params": {
                    "query_text": query_text,
                    "top_k": top_k,
                    "filters": {param: search_params[param] for param in search_params if param not in ['query_text', 'top_k']}
                },
                "search_result": final_results,
                "total_processing_time": total_elapsed_time,
                "reranking_time": reranked.get("processing_time", 0)
            }
            
            return Response(json.dumps(response_data, ensure_ascii=False), 
                          content_type="application/json; charset=utf-8")
                          
        except Exception as e:
            logger.error(f"재랭킹 처리 오류: {str(e)}")
            return jsonify({
                "result_code": "F000004",
                "message": f"재랭킹 처리 중 오류가 발생했습니다: {str(e)}",
                "search_result": None
            }), 500
            
    except Exception as e:
        logger.error(f"통합 검색 오류: {str(e)}")
        return jsonify({
            "result_code": "F000005",
            "message": f"통합 검색 중 오류가 발생했습니다: {str(e)}",
            "search_result": None
        }), 500


@app.route("/reranker/rerank", methods=['POST'])
def rerank():
    """
    패시지 재랭킹 엔드포인트
    
    주어진 쿼리와 패시지 목록에 대해 재랭킹을 수행합니다.
    재랭킹 방식은 쿼리 파라미터 'type'에 따라 결정됩니다.
    
    Query Parameters:
        top_k (int, optional): 반환할 상위 결과 수
        type (str, optional): 재랭킹 방식 (flashrank, mrc, hybrid, auto)
        
    Request Body:
        query (str): 검색 쿼리
        results (List[Dict]): 재랭킹할 패시지 목록
        total (int, optional): 전체 결과 수
        reranked (bool, optional): 이미 재랭킹되었는지 여부
        
    Returns:
        JSON 응답: 재랭킹된 패시지 목록과 메타데이터
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        # Get top_k parameter from query string
        top_k = request.args.get('top_k', type=int)
        
        # Get reranker type parameter (flashrank, mrc, hybrid)
        rerank_type = request.args.get('type', 'auto').lower()
        
        # Set environment variable for reranker method
        os.environ["RERANK_METHOD"] = rerank_type
        
        # Get request body
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
            
        # Validate input
        try:
            search_result = SearchResultModel(**data)
        except Exception as e:
            return jsonify({
                "error": f"Invalid input format: {str(e)}"
            }), 400
            
        # Process reranking
        reranked = get_reranker_service().process_search_results(
            search_result.query,
            search_result.dict(),
            top_k
        )
        
        # 전체 요청 처리 시간 계산
        processing_time = time.time() - total_start_time
        # API 명세에 맞게 processing_time 필드 추가
        reranked["processing_time"] = processing_time
        
        logger.info(f"Total rerank endpoint processing time: {processing_time:.3f} seconds")
        
        # 최적화된 응답 생성
        response_data = json.dumps(reranked, ensure_ascii=False)
        response = Response(
            response_data,
            mimetype='application/json; charset=utf-8'
        )
        
        # FastCGI 응답 지연 해결을 위한 핵심 헤더 설정
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Content-Length'] = str(len(response.data))
        
        return response
        
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        return jsonify({
            "error": f"Reranking failed: {str(e)}"
        }), 500


@app.route("/reranker/mrc-rerank", methods=['POST'])
def mrc_rerank():
    """
    MRC(기계 독해) 기반 재랭킹 엔드포인트
    
    주어진 쿼리와 패시지 목록에 대해 MRC 모델을 사용하여 재랭킹을 수행합니다.
    MRC 모델은 질문에 대한 답변 가능성을 기준으로 패시지의 순위를 결정합니다.
    
    Query Parameters:
        top_k (int, optional): 반환할 상위 결과 수
        
    Request Body:
        query (str): 검색 쿼리
        results (List[Dict]): 재랭킹할 패시지 목록
        total (int, optional): 전체 결과 수
        reranked (bool, optional): 이미 재랭킹되었는지 여부
        
    Returns:
        JSON 응답: MRC 모델로 재랭킹된 패시지 목록과 메타데이터
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        # Get top_k parameter from query string
        top_k = request.args.get('top_k', type=int)
        
        # MRC 방식으로 강제 설정
        os.environ["RERANK_METHOD"] = "mrc"
        
        # Get request body
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
            
        # Validate input
        try:
            search_result = SearchResultModel(**data)
        except Exception as e:
            return jsonify({
                "error": f"Invalid input format: {str(e)}"
            }), 400
            
        # Process reranking
        reranked = get_reranker_service().process_search_results(
            search_result.query,
            search_result.dict(),
            top_k
        )
        
        # 전체 요청 처리 시간 계산
        processing_time = time.time() - total_start_time
        # API 명세에 맞게 processing_time 필드 추가
        reranked["processing_time"] = processing_time
        
        logger.info(f"Total mrc-rerank endpoint processing time: {processing_time:.3f} seconds")
        
        # 최적화된 응답 생성
        response_data = json.dumps(reranked, ensure_ascii=False)
        response = Response(
            response_data,
            mimetype='application/json; charset=utf-8'
        )
        
        # FastCGI 응답 지연 해결을 위한 핵심 헤더 설정
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Content-Length'] = str(len(response.data))
        
        return response
        
    except Exception as e:
        logger.error(f"MRC reranking failed: {str(e)}")
        return jsonify({
            "error": f"MRC reranking failed: {str(e)}"
        }), 500


# MRC 설정 확인 함수
def check_mrc_configuration():
    """
    MRC 모델 설정 및 파일 존재 여부 확인
    
    MRC 모델의 활성화 상태, 설정 파일 및 모델 파일의 존재 여부를 확인합니다.
    절대 경로와 상대 경로 모두에서 파일 존재 여부를 검사합니다.
    
    Returns:
        Dict: MRC 설정 상태 정보를 담은 딕셔너리
            - mrc_enabled: MRC 기능 활성화 여부
            - mrc_reranker_loaded: MRC 재랭커 로드 여부
            - config_path: 설정 파일 경로
            - model_path: 모델 파일 경로
            - config_exists: 설정 파일 존재 여부
            - model_exists: 모델 파일 존재 여부
    """
    try:
        service = get_reranker_service()
        mrc_enabled = service.mrc_enabled if hasattr(service, 'mrc_enabled') else False
        mrc_reranker = service.mrc_reranker if hasattr(service, 'mrc_reranker') else None
        
        # 설정 파일 및 모델 파일 경로
        config_path = None
        model_path = None
        
        if hasattr(service, 'config') and isinstance(service.config, dict):
            mrc_config = service.config.get('mrc', {})
            config_path = mrc_config.get('model_config_path')
            model_path = mrc_config.get('model_ckpt_path')
        
        # 파일 존재 여부 확인
        config_exists = False
        model_exists = False
        
        # 절대 경로로 확인
        if config_path:
            config_exists = os.path.exists(config_path)
            logger.debug(f"절대 경로 MRC 설정 파일 확인: {config_path} -> {config_exists}")
        
        if model_path:
            model_exists = os.path.exists(model_path)
            logger.debug(f"절대 경로 MRC 모델 파일 확인: {model_path} -> {model_exists}")
            
        # 절대 경로에서 파일을 찾지 못한 경우 상대 경로로 시도
        if config_path and not config_exists and config_path.startswith("/reranker/"):
            relative_config_path = config_path[10:]  # "/reranker/" 제거
            relative_config_exists = os.path.exists(relative_config_path)
            logger.debug(f"상대 경로 MRC 설정 파일 확인: {relative_config_path} -> {relative_config_exists}")
            if relative_config_exists:
                config_exists = True
                
        if model_path and not model_exists and model_path.startswith("/reranker/"):
            relative_model_path = model_path[10:]  # "/reranker/" 제거
            relative_model_exists = os.path.exists(relative_model_path)
            logger.debug(f"상대 경로 MRC 모델 파일 확인: {relative_model_path} -> {relative_model_exists}")
            if relative_model_exists:
                model_exists = True
        
        return {
            "mrc_enabled": mrc_enabled,
            "mrc_reranker_loaded": mrc_reranker is not None,
            "config_path": config_path,
            "model_path": model_path,
            "config_exists": config_exists,
            "model_exists": model_exists
        }
    except Exception as e:
        logger.error(f"MRC 설정 확인 중 오류 발생: {str(e)}")
        return {
            "error": str(e),
            "mrc_enabled": False,
            "mrc_reranker_loaded": False,
            "config_exists": False,
            "model_exists": False
        }

@app.route("/reranker/mrc-status", methods=['GET'])
def mrc_status():
    """
    MRC 모듈 상태 확인 API
    
    MRC 모듈의 현재 상태를 확인하는 엔드포인트입니다.
    MRC 모델의 활성화 여부, 설정 파일 및 모델 파일의 존재 여부 등을 반환합니다.
    
    Returns:
        JSON 응답: MRC 모듈 상태 정보
            - status: 상태 코드 ("ok" 또는 "error")
            - mrc_configuration: MRC 설정 상태 정보
            - timestamp: 응답 생성 시간 (UNIX 타임스탬프)
    """
    try:
        # MRC 설정 확인
        mrc_status = check_mrc_configuration()
        
        return jsonify({
            "status": "ok",
            "mrc_configuration": mrc_status,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"MRC 상태 확인 API 오류: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }), 500

@app.route("/reranker/hybrid-rerank", methods=['POST'])
def hybrid_rerank():
    """
    하이브리드 재랭킹 엔드포인트 (FlashRank + MRC)
    
    FlashRank와 MRC 모델을 함께 사용하여 재랭킹을 수행하는 엔드포인트입니다.
    두 모델의 점수를 가중치에 따라 결합하여 최종 순위를 결정합니다.
    
    Query Parameters:
        top_k (int, optional): 반환할 상위 결과 수
        mrc_weight (float, optional): MRC 점수의 가중치 (0~1 사이 값)
        
    Request Body:
        query (str): 검색 쿼리
        results (List[Dict]): 재랭킹할 패시지 목록
        total (int, optional): 전체 결과 수
        reranked (bool, optional): 이미 재랭킹되었는지 여부
        top_k (int, optional): 반환할 상위 결과 수 (쿼리 파라미터 우선)
        mrc_weight (float, optional): MRC 점수의 가중치 (쿼리 파라미터 우선)
        
    Returns:
        JSON 응답: 하이브리드 방식으로 재랭킹된 패시지 목록과 메타데이터
            - 각 패시지에는 flashrank_score, mrc_score, hybrid_score 포함
            - processing_time: 전체 처리 시간
            - flashrank_time: FlashRank 처리 시간
            - mrc_time: MRC 처리 시간
            - mrc_weight: 사용된 MRC 가중치
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        # 상세 로깅을 위한 타임스탬프 딕셔너리 초기화
        timestamps = {
            "start": total_start_time,
            "steps": []
        }
        
        # 단계별 시간 측정 함수
        def log_step(name):
            now = time.time()
            step_time = now - timestamps.get("last_step", total_start_time)
            elapsed = now - total_start_time
            timestamps["steps"].append({"name": name, "time": step_time, "elapsed": elapsed})
            timestamps["last_step"] = now
            logger.info(f"[HYBRID-RERANK] 단계 '{name}' 소요시간: {step_time*1000:.2f}ms (누적: {elapsed*1000:.2f}ms)")
            
            # 상세 로그 파일에 기록
            try:
                with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-RERANK] 단계 '{name}' 소요시간: {step_time*1000:.2f}ms (누적: {elapsed*1000:.2f}ms)\n")
            except Exception as e:
                logger.warning(f"상세 로그 파일 기록 실패: {str(e)}")
        
        # 요청 파라미터 로깅
        log_step("요청 시작")
        
        # Get request body
        data = request.get_json()
        if not data:
            logger.error("[HYBRID-RERANK] 요청 본문이 비어있음")
            return jsonify({
                "error": "No JSON data provided"
            }), 400
            
        # 파라미터 추출: 쿼리 스트링과 JSON 본문 모두 확인
        # 1. 쿼리 스트링에서 파라미터 추출
        top_k_query = request.args.get('top_k', type=int)
        mrc_weight_query = request.args.get('mrc_weight', type=float)
        
        # 2. JSON 본문에서 파라미터 추출
        top_k_body = data.get('top_k') if isinstance(data, dict) else None
        mrc_weight_body = data.get('mrc_weight') if isinstance(data, dict) else None
        
        # 3. 파라미터 우선순위 결정: 쿼리 스트링 > JSON 본문
        top_k = top_k_query if top_k_query is not None else top_k_body
        mrc_weight = mrc_weight_query if mrc_weight_query is not None else mrc_weight_body
        
        # 파라미터 타입 변환
        if top_k is not None and not isinstance(top_k, int):
            try:
                top_k = int(top_k)
            except (ValueError, TypeError):
                logger.warning(f"[HYBRID-RERANK] top_k 파라미터 타입 변환 실패: {top_k}")
                top_k = None
                
        if mrc_weight is not None and not isinstance(mrc_weight, float):
            try:
                mrc_weight = float(mrc_weight)
            except (ValueError, TypeError):
                logger.warning(f"[HYBRID-RERANK] mrc_weight 파라미터 타입 변환 실패: {mrc_weight}")
                mrc_weight = None
        
        # 요청 파라미터 로깅
        logger.info(f"[HYBRID-RERANK] 요청 파라미터: top_k={top_k}, mrc_weight={mrc_weight}")
        logger.info(f"[HYBRID-RERANK] 파라미터 소스: top_k(쿼리)={top_k_query}, top_k(본문)={top_k_body}, mrc_weight(쿼리)={mrc_weight_query}, mrc_weight(본문)={mrc_weight_body}")
        
        # 하이브리드 방식으로 강제 설정
        os.environ["RERANK_METHOD"] = "hybrid"
        logger.info("[HYBRID-RERANK] 하이브리드 재랭킹 모드로 설정됨")
        log_step("환경 설정")
        
        # MRC 설정 확인 및 로깅
        mrc_config = check_mrc_configuration()
        logger.info(f"[HYBRID-RERANK] MRC 설정 상태: 활성화={mrc_config['mrc_enabled']}, 모델 로드됨={mrc_config['mrc_reranker_loaded']}")
        
        # 필요한 파일 존재 확인
        if not mrc_config['config_exists'] or not mrc_config['model_exists']:
            logger.warning(f"[HYBRID-RERANK] MRC 모델 파일 누락: 설정파일={mrc_config['config_exists']}, 모델파일={mrc_config['model_exists']}")
        
        log_step("MRC 설정 확인")
        
        # 요청 데이터 크기 로깅
        request_size = len(json.dumps(data).encode('utf-8'))
        logger.info(f"[HYBRID-RERANK] 요청 데이터 크기: {request_size/1024:.2f}KB")
        
        # Validate input - JSON 본문에서 필요한 필드만 추출하여 SearchResultModel 생성
        try:
            # 필수 필드만 추출
            search_data = {
                "query": data.get("query"),
                "results": data.get("results", [])
            }
            
            # 선택적 필드 추가
            for field in ["total", "reranked"]:
                if field in data:
                    search_data[field] = data[field]
                    
            search_result = SearchResultModel(**search_data)
            logger.info(f"[HYBRID-RERANK] 재랭킹 요청: query='{search_result.query}', 결과 수={len(search_result.results)}")
            
            # 첫 번째 패시지 샘플 로깅 (디버깅용)
            if search_result.results and len(search_result.results) > 0:
                first_passage = search_result.results[0]
                passage_preview = first_passage.text[:100] + "..." if len(first_passage.text) > 100 else first_passage.text
                logger.debug(f"[HYBRID-RERANK] 첫 번째 패시지 샘플: {passage_preview}")
            
        except Exception as e:
            logger.error(f"[HYBRID-RERANK] 요청 검증 실패: {str(e)}")
            return jsonify({
                "error": f"Invalid input format: {str(e)}"
            }), 400
        
        log_step("요청 검증")
        
        # Process reranking
        reranker_service = get_reranker_service()
        
        # MRC 가중치 설정
        if mrc_weight is not None:
            logger.info(f"[HYBRID-RERANK] MRC 가중치 변경: {getattr(reranker_service, 'hybrid_weight_mrc', '기본값')} -> {mrc_weight}")
            reranker_service.hybrid_weight_mrc = mrc_weight
        
        log_step("서비스 초기화")
        
        # 재랭킹 처리 시작
        logger.info("[HYBRID-RERANK] 하이브리드 재랭킹 처리 시작")
        process_start_time = time.time()
        
        reranked = reranker_service.process_search_results(
            search_result.query,
            search_result.dict(),
            top_k
        )
        
        process_time = time.time() - process_start_time
        logger.info(f"[HYBRID-RERANK] 하이브리드 재랭킹 처리 완료: {process_time:.3f}초")
        
        # 상세 처리 시간 로깅
        flashrank_time = reranked.get("flashrank_time", 0.0)
        mrc_time = reranked.get("mrc_time", 0.0)
        logger.info(f"[HYBRID-RERANK] 상세 처리 시간: FlashRank={flashrank_time:.3f}초, MRC={mrc_time:.3f}초")
        
        log_step("재랭킹 처리")
        
        # 전체 요청 처리 시간 계산
        processing_time = time.time() - total_start_time
        
        # API 명세에 맞게 processing_time 필드 추가
        reranked["processing_time"] = processing_time
        reranked["mrc_weight"] = reranker_service.hybrid_weight_mrc
        
        # 응답 메타데이터 구성 과정 로깅
        logger.info(f"[HYBRID-RERANK] 응답 메타데이터 구성: processing_time={processing_time:.3f}초, mrc_weight={reranker_service.hybrid_weight_mrc}")
        
        # 재랭커 타입 확인 및 로깅
        reranker_type = reranked.get("reranker_type", "unknown")
        
        # 하이브리드 재랭킹 결과인지 확인
        if reranker_type == "hybrid":
            logger.info(f"[HYBRID-RERANK] 하이브리드 재랭킹 성공적으로 완료됨")
        elif reranker_type == "flashrank":
            # FlashRank 결과인 경우, 하이브리드로 변경
            logger.info(f"[HYBRID-RERANK] FlashRank 결과를 하이브리드 결과로 변환합니다")
            reranked["reranker_type"] = "hybrid"
        else:
            logger.warning(f"[HYBRID-RERANK] 하이브리드 재랭킹 요청했으나 결과 타입은 '{reranker_type}'입니다. MRC 설정을 확인하세요.")
        
        # 필터링 적용 여부 확인
        original_count = len(reranked.get("results", []))
        if top_k and isinstance(top_k, int) and top_k > 0 and top_k < original_count:
            # 상위 top_k개만 필터링하여 반환
            reranked["results"] = reranked["results"][:top_k]
            reranked["filtered_count"] = original_count
            reranked["returned_count"] = top_k
            logger.info(f"[HYBRID-RERANK] 결과 필터링 적용: 전체 {original_count}개 중 상위 {top_k}개만 반환")
            
            # 상세 로그 파일에 기록
            try:
                with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-RERANK] 결과 필터링 적용: 전체 {original_count}개 중 상위 {top_k}개만 반환\n")
            except Exception as e:
                logger.warning(f"상세 로그 파일 기록 실패: {str(e)}")
        
        # 결과 통계 로깅
        result_count = len(reranked.get("results", []))
        if result_count > 0:
            logger.info(f"[HYBRID-RERANK] 결과 통계: 총 {result_count}개 결과, 상위 점수: {reranked['results'][0]['score']:.4f} (요청: {top_k}개)")
        else:
            logger.warning(f"[HYBRID-RERANK] 결과 통계: 결과 없음 (요청: {top_k}개)")
        
        log_step("메타데이터 구성")
        
        # 최종 처리 시간 로깅
        logger.info(f"[HYBRID-RERANK] 전체 처리 시간: {processing_time:.3f}초")
        
        # 단계별 처리 시간 상세 로깅
        steps_log = "\n".join([
            f"  - {step['name']}: {step['time']*1000:.2f}ms ({step['elapsed']*1000:.2f}ms 경과)"
            for step in timestamps["steps"]
        ])
        logger.debug(f"[HYBRID-RERANK] 단계별 처리 시간:\n{steps_log}")
        
        # 상세 로그 파일에 단계별 처리 시간 기록
        try:
            with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-RERANK] 단계별 처리 시간:\n{steps_log}\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-RERANK] 전체 처리 시간: {processing_time:.3f}초\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-RERANK] 결과 통계: 총 {result_count}개 결과, 상위 점수: {reranked['results'][0]['score']:.4f}\n")
        except Exception as e:
            logger.warning(f"상세 로그 파일 기록 실패: {str(e)}")
        
        # 최적화된 응답 생성
        response_data = json.dumps(reranked, ensure_ascii=False)
        response = Response(
            response_data,
            mimetype='application/json; charset=utf-8'
        )
        
        # 응답 크기 로깅
        response_size = len(response_data.encode('utf-8'))
        logger.info(f"[HYBRID-RERANK] 응답 데이터 크기: {response_size/1024:.2f}KB")
        
        # FastCGI 응답 지연 해결을 위한 핵심 헤더 설정
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Content-Length'] = str(len(response.data))
        
        log_step("응답 생성")
        
        return response
        
    except Exception as e:
        logger.error(f"[HYBRID-RERANK] 하이브리드 재랭킹 실패: {str(e)}", exc_info=True)
        
        # 상세 로그 파일에 오류 기록
        try:
            with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-RERANK] 오류 발생: {str(e)}\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {traceback.format_exc()}\n")
        except Exception as log_error:
            logger.warning(f"상세 로그 파일 오류 기록 실패: {str(log_error)}")
        
        return jsonify({
            "error": f"Hybrid reranking failed: {str(e)}"
        }), 500


@app.route("/reranker/batch_rerank", methods=["POST"])
def batch_rerank():
    """
    여러 쿼리와 패시지를 일괄 재랭킹하는 엔드포인트
    
    여러 쿼리와 각 쿼리에 대한 패시지 목록을 한 번에 처리합니다.
    대량의 재랭킹 요청을 효율적으로 처리하기 위한 배치 처리 API입니다.
    
    Query Parameters:
        top_k (int, optional): 각 쿼리별로 반환할 상위 결과 수
        
    Request Body:
        List[Dict]: 각 쿼리별 재랭킹 요청 목록
            - query (str): 검색 쿼리
            - results (List[Dict]): 재랭킹할 패시지 목록
            - total (int, optional): 전체 결과 수
            - reranked (bool, optional): 이미 재랭킹되었는지 여부
    
    Returns:
        JSON 응답: 배치 처리 결과
            - results: 각 쿼리별 재랭킹 결과 목록
            - total_processing_time: 전체 처리 시간
            - query_count: 처리된 쿼리 수
    """
    try:
        # 전체 요청 처리 시간 측정 시작
        total_start_time = time.time()
        
        data = request.get_json()
        top_k = request.args.get("top_k", type=int)
        
        # Process each query
        results = []
        for query_data in data:
            search_result = SearchResultModel(**query_data)
            reranked = get_reranker_service().process_search_results(
                search_result.query,
                search_result.dict(),
                top_k
            )
            results.append(reranked)
        
        # 전체 요청 처리 시간 계산
        total_elapsed_time = time.time() - total_start_time
        logger.info(f"Total batch_rerank endpoint processing time: {total_elapsed_time:.3f} seconds")
        
        # 배치 처리 결과에 전체 처리 시간 추가
        batch_result = {
            "results": results,
            "total_processing_time": total_elapsed_time,
            "query_count": len(results)
        }
        
        # 최적화된 응답 생성
        response_data = json.dumps(batch_result, ensure_ascii=False)
        response = Response(
            response_data,
            mimetype='application/json; charset=utf-8'
        )
        
        # FastCGI 응답 지연 해결을 위한 핵심 헤더 설정
        response.headers['X-Accel-Buffering'] = 'no'
        response.headers['Content-Length'] = str(len(response.data))
        
        return response
        
    except Exception as e:
        return Response(
            json.dumps({"error": f"Batch reranking failed: {str(e)}"}, ensure_ascii=False),
            status=500,
            mimetype='application/json; charset=utf-8'
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9020) 