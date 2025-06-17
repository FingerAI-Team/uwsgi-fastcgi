"""
Service layer for reranking functionality
"""

import os
import sys
import logging
import torch
import time
import threading
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel
from flashrank import Ranker, RerankRequest

# PyTorch 프로파일러 임포트
try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    
# 더 빠른 JSON 처리를 위해 ujson 사용
try:
    import ujson as json
    print("Using ujson for faster JSON processing")
except ImportError:
    import json
    print("ujson not available, using default json")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 파일 로그 추가 (볼륨에 저장)
try:
    file_handler = logging.FileHandler('/var/log/reranker/reranker_detail.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info("상세 로그 파일 설정 완료: /var/log/reranker/reranker_detail.log")
except Exception as e:
    logger.warning(f"로그 파일 설정 실패: {str(e)}")

# MRC 모듈 임포트
try:
    # 여러 경로 시도
    try:
        from src.mrc import MRCReranker
        MRC_AVAILABLE = True
        logger.info("MRC 모듈 가져오기 성공 (from src.mrc)")
    except ImportError as e:
        logger.error(f"MRC 모듈 가져오기 실패 (from src.mrc): {str(e)}")
        try:
            import sys
            sys.path.append('/reranker')
            logger.debug(f"Python 경로에 '/reranker' 추가: {sys.path}")
            from src.mrc import MRCReranker
            MRC_AVAILABLE = True
            logger.info("MRC 모듈 가져오기 성공 (from /reranker/src.mrc)")
        except ImportError as e:
            logger.error(f"MRC 모듈 가져오기 실패 (from /reranker/src.mrc): {str(e)}")
            try:
                from reranker.src.mrc import MRCReranker
                MRC_AVAILABLE = True
                logger.info("MRC 모듈 가져오기 성공 (from reranker.src.mrc)")
            except ImportError as e:
                logger.error(f"MRC 모듈 가져오기 실패 (from reranker.src.mrc): {str(e)}")
                try:
                    from .src.mrc import MRCReranker
                    MRC_AVAILABLE = True
                    logger.info("MRC 모듈 가져오기 성공 (from .src.mrc)")
                except ImportError as e:
                    logger.error(f"MRC 모듈 가져오기 실패 (from .src.mrc): {str(e)}")
                    MRC_AVAILABLE = False
                    logger.warning("MRC 모듈을 가져올 수 없습니다")
except Exception as e:
    MRC_AVAILABLE = False
    logger.error(f"MRC 모듈 임포트 중 오류 발생: {str(e)}", exc_info=True)

# 메모리 캐시 - 자주 사용되는 재랭킹 요청 결과 캐싱
_RERANK_CACHE = {}
_CACHE_SIZE_LIMIT = 1000  # 최대 캐시 항목 수

# 프로파일링 활성화 여부
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "0") == "1"
PROFILE_OUTPUT_DIR = os.getenv("PROFILE_OUTPUT_DIR", "/reranker/profiles")

def get_cache_key(query: str, passages_hash: str) -> str:
    """캐시 키 생성"""
    return f"{query}:{passages_hash}"

def hash_passages(passages: List[Dict]) -> str:
    """패시지 리스트의 해시 생성"""
    try:
        passage_texts = [p.get('text', '')[:100] for p in passages]  # 각 패시지의 앞부분만 사용
        return str(hash(tuple(passage_texts)))
    except Exception as e:
        logger.warning(f"Failed to hash passages: {e}")
        return str(hash(str(passages)))  # 폴백 해싱

# GPU 메모리 상태 로깅 함수 추가
def log_gpu_memory(tag: str = ""):
    """GPU 메모리 사용량 로깅"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        logger.info(f"GPU Memory [{tag}] - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB, Max: {max_allocated:.2f}MB")
    else:
        logger.info(f"GPU Memory [{tag}] - CUDA not available")

class PassageModel(BaseModel):
    """Single passage model"""
    passage_id: Optional[Any] = None
    doc_id: Optional[str] = None
    text: str
    score: Optional[float] = None
    position: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_encoders = {
            str: lambda v: v.encode('utf-8').decode('utf-8')
        }
        
    # 메모리 효율성을 위한 __slots__ 추가
    __slots__ = ('passage_id', 'doc_id', 'text', 'score', 'position', 'metadata')


class SearchResultModel(BaseModel):
    """Search result containing multiple passages"""
    query: str
    results: List[PassageModel]
    total: Optional[int] = None
    reranked: Optional[bool] = False

    class Config:
        json_encoders = {
            str: lambda v: v.encode('utf-8').decode('utf-8')
        }
        
    # 메모리 효율성을 위한 __slots__ 추가
    __slots__ = ('query', 'results', 'total', 'reranked')


class RerankerResponseModel(BaseModel):
    """Response model for reranker API"""
    query: str
    results: List[PassageModel]
    total: int
    reranked: bool = True

    class Config:
        json_encoders = {
            str: lambda v: v.encode('utf-8').decode('utf-8')
        }
        
    # 메모리 효율성을 위한 __slots__ 추가
    __slots__ = ('query', 'results', 'total', 'reranked')

    def json(self, **kwargs):
        # ujson 사용 가능하면 ujson으로 직렬화
        if 'ujson' in globals():
            return json.dumps(self.dict(), ensure_ascii=False, **kwargs)
        return json.dumps(self.dict(), ensure_ascii=False, **kwargs)


class RerankerService:
    """Service for reranking passages"""
    
    _instance = None  # 싱글톤 인스턴스
    
    @classmethod
    def get_instance(cls, config_path=None):
        """싱글톤 패턴으로 인스턴스 반환"""
        if cls._instance is None:
            logger.info(f"Creating new RerankerService instance with config: {config_path}")
            cls._instance = cls(config_path)
        else:
            logger.info("Returning existing RerankerService instance")
        return cls._instance
    
    def __init__(self, config_path: str = None):
        """
        Initialize the reranker service
        
        Args:
            config_path: Path to config file, if None, use default settings
        """
        try:
            init_start_time = time.time()  # 초기화 시작 시간
            logger.debug("Loading configuration...")
            self.config = self._load_config(config_path)
            logger.info(f"Configuration loaded in {(time.time() - init_start_time)*1000:.2f}ms")
            
            # 로그 레벨 설정
            log_level = self.config.get("log_level", "INFO")
            log_level_int = getattr(logging, log_level.upper(), logging.INFO)
            logger.setLevel(log_level_int)
            logger.info(f"Log level set to {log_level}")
            
            # FlashRank 초기화
            self.ranker = None
            self.model_name = self.config.get("model_name", "BAAI/bge-reranker-large")
            self.batch_size = self._get_batch_size()
            
            # GPU 사용 가능 여부 확인
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # 모델 로딩 시작
            model_start_time = time.time()
            try:
                logger.info(f"Loading FlashRank model: {self.model_name}")
                self.ranker = Ranker(model_name_or_path=self.model_name)
                
                # 모델이 GPU를 사용하는지 확인
                if self.device == "cuda":
                    # 모델 디바이스 확인
                    if hasattr(self.ranker, 'model'):
                        model_device = next(self.ranker.model.parameters()).device
                        logger.info(f"FlashRank model device: {model_device}")
                        
                        # CPU에 있으면 GPU로 이동
                        if str(model_device) == "cpu":
                            logger.warning("FlashRank model is on CPU! Moving to GPU...")
                            try:
                                self.ranker.model.to('cuda')
                                new_device = next(self.ranker.model.parameters()).device
                                logger.info(f"FlashRank model moved to: {new_device}")
                                
                                # 로그 파일에 기록
                                with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [INIT] FlashRank model moved from CPU to {new_device}\n")
                            except Exception as e:
                                logger.error(f"Failed to move FlashRank model to GPU: {str(e)}")
                
                logger.info(f"FlashRank model loaded in {(time.time() - model_start_time):.2f}s")
            except Exception as e:
                logger.error(f"Failed to load FlashRank model: {str(e)}")
                self.ranker = None
            
            # 로그 레벨 설정
            log_level = self.config.get("log_level", "INFO")
            log_level_int = getattr(logging, log_level.upper(), logging.INFO)
            logger.setLevel(log_level_int)
            logger.info(f"Log level set to {log_level}")
            
            self.cache_dir = os.getenv("FLASHRANK_CACHE_DIR", self.config.get("cache_dir", "/reranker/models"))
            self.max_length = int(os.getenv("FLASHRANK_MAX_LENGTH", self.config.get("max_length", 512)))
            
            # 배치 크기 설정
            self.batch_size = self._get_batch_size()
            
            # GPU 동시 접근 제한 (환경변수로 설정 가능)
            self.max_gpu_workers = int(os.getenv('MAX_GPU_WORKERS', '7'))
            self._gpu_semaphore = threading.Semaphore(self.max_gpu_workers)
            logger.info(f"GPU 동시 작업 제한: {self.max_gpu_workers}개")
            
            # 프로파일링 설정
            self.enable_profiling = ENABLE_PROFILING and PROFILER_AVAILABLE
            self.profile_dir = PROFILE_OUTPUT_DIR
            if self.enable_profiling:
                logger.info(f"PyTorch profiling enabled. Profiles will be saved to {self.profile_dir}")
                os.makedirs(self.profile_dir, exist_ok=True)
            
            logger.info(f"Initializing FlashRank reranker with model: {self.model_name}")
            logger.debug(f"Cache directory: {self.cache_dir}")
            logger.debug(f"Max length: {self.max_length}")
            logger.debug(f"Batch size: {self.batch_size}")
            
            # 시스템 정보 로깅
            self._log_system_info()
            
            # MRC 재랭커 초기화 (설정에서 활성화된 경우)
            self.mrc_enabled = self.config.get("mrc", {}).get("enabled", False)
            self.mrc_reranker = None
            self.hybrid_weight_mrc = self.config.get("mrc", {}).get("hybrid_weight_mrc", 0.7)
            
            logger.debug(f"MRC 초기화 시작: enabled={self.mrc_enabled}, MRC_AVAILABLE={MRC_AVAILABLE}")
            logger.debug(f"MRC 설정: {self.config.get('mrc', {})}")
            
            if self.mrc_enabled and MRC_AVAILABLE:
                try:
                    logger.info("MRC 재랭커 초기화 중...")
                    mrc_config_path = self.config.get("mrc", {}).get("model_config_path")
                    mrc_model_path = self.config.get("mrc", {}).get("model_ckpt_path")
                    
                    # 절대 경로 변환 시도
                    if not os.path.isabs(mrc_config_path) and not os.path.exists(mrc_config_path):
                        abs_config_path = os.path.abspath(mrc_config_path)
                        logger.debug(f"절대 경로로 변환: {mrc_config_path} -> {abs_config_path}")
                        mrc_config_path = abs_config_path
                        
                    if not os.path.isabs(mrc_model_path) and not os.path.exists(mrc_model_path):
                        abs_model_path = os.path.abspath(mrc_model_path)
                        logger.debug(f"절대 경로로 변환: {mrc_model_path} -> {abs_model_path}")
                        mrc_model_path = abs_model_path
                    
                    logger.debug(f"MRC 설정 파일 경로: {mrc_config_path}, 존재 여부: {os.path.exists(mrc_config_path)}")
                    logger.debug(f"MRC 모델 파일 경로: {mrc_model_path}, 존재 여부: {os.path.exists(mrc_model_path)}")
                    
                    # 파일 내용 로깅 (디버깅용)
                    try:
                        if os.path.exists(mrc_config_path):
                            with open(mrc_config_path, 'r') as f:
                                config_content = f.read()
                            logger.debug(f"MRC 설정 파일 내용: {config_content[:500]}...")
                    except Exception as e:
                        logger.warning(f"MRC 설정 파일 읽기 실패: {str(e)}")
                    
                    # MRC 모델 디렉토리 확인 및 생성
                    if mrc_config_path and mrc_model_path:
                        os.makedirs(os.path.dirname(mrc_config_path), exist_ok=True)
                        os.makedirs(os.path.dirname(mrc_model_path), exist_ok=True)
                        
                        # MRC 모델 다운로드 설정이 있는 경우
                        config_gdrive_id = self.config.get("mrc", {}).get("model_config_gdrive_id")
                        model_gdrive_id = self.config.get("mrc", {}).get("model_ckpt_gdrive_id")
                        
                        # 설정 파일 체크 및 다운로드
                        if config_gdrive_id and not os.path.exists(mrc_config_path):
                            try:
                                # 여러 경로에서 다운로드 함수 임포트 시도
                                try:
                                    from src.mrc import download_checkpoints
                                    logger.info("다운로드 함수 임포트 성공 (from src.mrc)")
                                except ImportError:
                                    try:
                                        from reranker.src.mrc import download_checkpoints
                                        logger.info("다운로드 함수 임포트 성공 (from reranker.src.mrc)")
                                    except ImportError:
                                        from .src.mrc import download_checkpoints
                                        logger.info("다운로드 함수 임포트 성공 (from .src.mrc)")
                                
                                logger.info(f"MRC 설정 파일 다운로드 중: {mrc_config_path}")
                                download_checkpoints(mrc_config_path, config_gdrive_id)
                            except Exception as e:
                                logger.warning(f"MRC 설정 파일 다운로드 실패: {e}")
                                logger.info(f"설정 파일을 '{mrc_config_path}' 경로에 수동으로 추가해주세요.")
                        
                        # 모델 체크포인트 체크 및 다운로드
                        if model_gdrive_id and not os.path.exists(mrc_model_path):
                            try:
                                # 여러 경로에서 다운로드 함수 임포트 시도
                                try:
                                    from src.mrc import download_checkpoints
                                    logger.info("다운로드 함수 임포트 성공 (from src.mrc)")
                                except ImportError:
                                    try:
                                        from reranker.src.mrc import download_checkpoints
                                        logger.info("다운로드 함수 임포트 성공 (from reranker.src.mrc)")
                                    except ImportError:
                                        from .src.mrc import download_checkpoints
                                        logger.info("다운로드 함수 임포트 성공 (from .src.mrc)")
                                
                                logger.info(f"MRC 모델 파일 다운로드 중: {mrc_model_path}")
                                download_checkpoints(mrc_model_path, model_gdrive_id)
                            except Exception as e:
                                logger.warning(f"MRC 모델 파일 다운로드 실패: {e}")
                                logger.info(f"모델 파일을 '{mrc_model_path}' 경로에 수동으로 추가해주세요.")
                    
                    # 경로 처리 (절대 경로에서 상대 경로로 변환)
                    if not os.path.exists(mrc_config_path) and mrc_config_path.startswith("/reranker/"):
                        relative_config_path = mrc_config_path[10:]  # "/reranker/" 제거
                        if os.path.exists(relative_config_path):
                            logger.info(f"상대 경로로 변환: {mrc_config_path} -> {relative_config_path}")
                            mrc_config_path = relative_config_path
                            
                    if not os.path.exists(mrc_model_path) and mrc_model_path.startswith("/reranker/"):
                        relative_model_path = mrc_model_path[10:]  # "/reranker/" 제거
                        if os.path.exists(relative_model_path):
                            logger.info(f"상대 경로로 변환: {mrc_model_path} -> {relative_model_path}")
                            mrc_model_path = relative_model_path
                
                    # MRC 재랭커 인스턴스 생성
                    logger.debug("MRCReranker.get_instance 호출 시작")
                    try:
                        logger.debug(f"최종 MRC 설정 파일 경로: {mrc_config_path}")
                        logger.debug(f"최종 MRC 모델 파일 경로: {mrc_model_path}")
                        
                        # NumPy 버전 체크 (디버깅용)
                        try:
                            import numpy
                            logger.info(f"NumPy 버전: {numpy.__version__}")
                        except Exception as e:
                            logger.warning(f"NumPy 버전 확인 실패: {str(e)}")
                            
                        # PyTorch 버전 체크 (디버깅용)
                        try:
                            import torch
                            logger.info(f"PyTorch 버전: {torch.__version__}")
                        except Exception as e:
                            logger.warning(f"PyTorch 버전 확인 실패: {str(e)}")
                            
                        # TorchText 버전 체크 (디버깅용)
                        try:
                            import torchtext
                            logger.info(f"TorchText 버전: {torchtext.__version__}")
                        except Exception as e:
                            logger.warning(f"TorchText 버전 확인 실패: {str(e)}")
                            
                        # PyTorch Lightning 버전 체크 (디버깅용)
                        try:
                            import pytorch_lightning
                            logger.info(f"PyTorch Lightning 버전: {pytorch_lightning.__version__}")
                        except Exception as e:
                            logger.warning(f"PyTorch Lightning 버전 확인 실패: {str(e)}")
                            
                        # Munch 버전 체크 (디버깅용)
                        try:
                            import munch
                            logger.info(f"Munch 버전: {munch.__version__ if hasattr(munch, '__version__') else '알 수 없음'}")
                        except Exception as e:
                            logger.warning(f"Munch 버전 확인 실패: {str(e)}")
                        
                        self.mrc_reranker = MRCReranker.get_instance(mrc_config_path, mrc_model_path)
                        logger.info("MRC 재랭커 초기화 완료")
                        logger.debug(f"MRC 재랭커 객체: {self.mrc_reranker}")
                    except Exception as inner_e:
                        logger.error(f"MRCReranker.get_instance 호출 실패: {str(inner_e)}", exc_info=True)
                        raise inner_e
                except Exception as e:
                    logger.error(f"MRC 재랭커 초기화 실패: {str(e)}", exc_info=True)
                    logger.error(f"상세 오류 정보: {type(e).__name__}", exc_info=True)
                    self.mrc_enabled = False
            elif self.mrc_enabled and not MRC_AVAILABLE:
                logger.warning("MRC 모듈을 가져올 수 없어 MRC 재랭킹이 비활성화됩니다")
                logger.warning(f"Python 경로: {sys.path}")
                self.mrc_enabled = False
                
        except Exception as e:
            logger.error(f"Failed to initialize RerankerService: {str(e)}")
            raise
    
    def _log_system_info(self):
        """시스템 정보 로깅"""
        logger.info("=== System Information ===")
        
        # Python 버전
        import sys
        logger.info(f"Python version: {sys.version}")
        
        # PyTorch 버전
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # CUDA 정보
        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("CUDA available: False")
            
        logger.info("========================")
    
    def _get_batch_size(self) -> int:
        """환경에 맞는 배치 크기 결정"""
        # 기본 배치 크기
        default_batch_size = {
            "cpu": 32,
            "gpu": 256
        }
        
        # GPU 여부에 따라 배치 크기 선택
        mode = "gpu" if torch.cuda.is_available() else "cpu"
        
        # 설정된 배치 크기 가져오기
        if isinstance(self.config.get("batch_size"), dict):
            return self.config["batch_size"].get(mode, default_batch_size[mode])
        elif isinstance(self.config.get("batch_size"), (int, str)):
            return int(self.config["batch_size"])
        else:
            return default_batch_size[mode]
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_batch_size = {
            "cpu": 32,
            "gpu": 256
        }
        
        default_config = {
            "model_name": os.getenv("FLASHRANK_MODEL", "ms-marco-TinyBERT-L-2-v2"),
            "cache_dir": os.getenv("FLASHRANK_CACHE_DIR", "/reranker/models"),
            "max_length": int(os.getenv("FLASHRANK_MAX_LENGTH", "512")),
            "batch_size": default_batch_size,
            "mrc": {
                "enabled": False
            }
        }
        
        if not config_path:
            logger.warning("No config path provided, using default configuration")
            return default_config
            
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(config_path):
                logger.warning(f"Config file not found at {config_path}")
                
                # 상대 경로 시도
                if config_path.startswith("/reranker/"):
                    relative_path = config_path[10:]  # "/reranker/" 제거
                    if os.path.exists(relative_path):
                        logger.info(f"Using relative path instead: {relative_path}")
                        config_path = relative_path
                    else:
                        logger.warning(f"Config file not found at relative path {relative_path} either")
                        return default_config
                else:
                    return default_config
            
            logger.info(f"Loading config from {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.debug(f"Loaded config content: {json.dumps(config)}")
                
                # MRC 설정 검증 및 디버그 로깅
                if "mrc" in config:
                    logger.info(f"MRC 설정 확인: {json.dumps(config['mrc'])}")
                    
                    # 파일 경로 변환 (절대 경로에서 상대 경로로)
                    mrc_config = config.get("mrc", {})
                    config_path = mrc_config.get("model_config_path")
                    model_path = mrc_config.get("model_ckpt_path")
                    
                    if config_path and config_path.startswith("/reranker/"):
                        relative_path = config_path[10:]
                        if os.path.exists(relative_path):
                            logger.info(f"MRC 설정 파일 상대 경로로 변환: {config_path} -> {relative_path}")
                            config["mrc"]["model_config_path"] = relative_path
                            
                    if model_path and model_path.startswith("/reranker/"):
                        relative_path = model_path[10:]
                        if os.path.exists(relative_path):
                            logger.info(f"MRC 모델 파일 상대 경로로 변환: {model_path} -> {relative_path}")
                            config["mrc"]["model_ckpt_path"] = relative_path
                
                # GPU 여부에 따라 배치 사이즈 선택
                if isinstance(config.get("batch_size"), dict):
                    mode = "gpu" if torch.cuda.is_available() else "cpu"
                    config["batch_size"] = config["batch_size"].get(mode, default_batch_size[mode])
                elif isinstance(config.get("batch_size"), (int, str)):
                    # 이전 형식의 설정을 위한 하위 호환성 유지
                    config["batch_size"] = int(config["batch_size"])
                    
                # 설정 병합 및 반환
                merged_config = {**default_config, **config}
                logger.debug(f"Final merged config: {json.dumps(merged_config)}")
                return merged_config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}", exc_info=True)
            logger.warning("Using default configuration due to error")
            return default_config
    
    def process_search_results(self, query: str, search_result: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Process search results with reranking
        
        Args:
            query: Search query
            search_result: Search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked search results
        """
        try:
            # 성능 측정 시작
            start_time = time.time()
            
            # 상세 시간 측정을 위한 타임스탬프 딕셔너리
            timestamps = {
                "start": start_time,
                "steps": []
            }
            
            def log_step(name):
                now = time.time()
                step_time = now - timestamps.get("last_time", start_time)
                elapsed = now - start_time
                timestamps["steps"].append({"name": name, "time": step_time, "elapsed": elapsed})
                timestamps["last_time"] = now
                logger.debug(f"Step '{name}' took {step_time*1000:.2f}ms (elapsed: {elapsed*1000:.2f}ms)")
            
            # 모델이 초기화되지 않은 경우 MRC만 사용하거나 원본 결과를 반환
            if self.ranker is None:
                logger.warning("FlashRank reranker not initialized")
                
                # MRC가 활성화되어 있고 초기화되었다면 MRC만 사용
                if self.mrc_enabled and self.mrc_reranker:
                    logger.info("Using MRC reranker only since FlashRank is not available")
                    return self.mrc_reranker.process_search_results(query, search_result, top_k)
                
                # 그렇지 않으면 원본 결과 반환
                logger.warning("No rerankers available, returning original results")
                return search_result
                
            # 결과가 없으면 빈 결과 반환
            if not search_result.get("results"):
                logger.warning("No results to rerank")
                return search_result
            
            # GPU 메모리 초기 상태 로깅
            log_gpu_memory("재랭킹 시작")
                
            # Convert passages to FlashRank format
            passages = []
            for result in search_result["results"]:
                passage = {
                    "id": result.get("passage_id"),
                    "text": result["text"],
                    "meta": {
                        "doc_id": result.get("doc_id"),
                        "original_score": result.get("score")
                    }
                }
                passages.append(passage)
            
            log_step("데이터 포맷 변환")
            
            # 캐시 사용하지 않음 (디버깅 및 테스트용으로 비활성화)
            log_step("캐시 사용 안함")
            
            # 재랭킹 메소드 결정 (환경변수로 제어 가능)
            rerank_method = os.getenv("RERANK_METHOD", "auto").lower()
            
            # 'auto' 모드: MRC가 활성화되어 있으면 하이브리드, 아니면 FlashRank만 사용
            if rerank_method == "auto":
                rerank_method = "hybrid" if self.mrc_enabled and self.mrc_reranker else "flashrank"
            
            # 재랭킹 수행
            if rerank_method == "mrc" and self.mrc_enabled and self.mrc_reranker:
                # MRC 방식만 사용
                logger.info("MRC 방식으로 재랭킹 수행")
                return self.mrc_reranker.process_search_results(query, search_result, top_k)
                
            elif rerank_method == "hybrid" and self.mrc_enabled and self.mrc_reranker:
                # 하이브리드 방식 (FlashRank + MRC)
                logger.info("[HYBRID-DETAIL] 하이브리드 방식으로 재랭킹 수행 시작")
                logger.info(f"[HYBRID-DETAIL] MRC 모듈 활성화 상태: {self.mrc_enabled}, MRC 리랭커 객체 존재: {self.mrc_reranker is not None}")
                
                if not self.mrc_enabled or self.mrc_reranker is None:
                    logger.error("[HYBRID-DETAIL] MRC 모듈이 비활성화되었거나 초기화되지 않았습니다. 하이브리드 재랭킹을 수행할 수 없습니다.")
                    logger.error("[HYBRID-DETAIL] MRC 모델 파일이 올바르게 설치되었는지 확인하세요: /reranker/models/mrc/config.json, /reranker/models/mrc/model.ckpt")
                    # FlashRank 방식으로 폴백
                    logger.info("[HYBRID-DETAIL] FlashRank 방식으로 대체 수행합니다.")
                    return self.perform_flashrank_reranking(query, passages, top_k, search_result)
                
                try:
                    # FlashRank 재랭킹 수행 (FlashRank가 초기화되었을 경우만)
                    flashrank_result = None
                    flashrank_scores = []
                    
                    # FlashRank 시작 시간 기록
                    flashrank_start_time = time.time()
                    
                    if self.ranker is not None:
                        logger.info("[HYBRID-DETAIL] FlashRank 재랭킹 시작")
                        flashrank_result, flashrank_scores, flashrank_time = self._flashrank_rerank(query, passages, None, search_result)
                        
                        # FlashRank 처리 시간 계산
                        flashrank_time = time.time() - flashrank_start_time
                        logger.info(f"[HYBRID-DETAIL] FlashRank 재랭킹 완료: {flashrank_time:.3f}초")
                        
                        # flashrank_result가 딕셔너리인지 리스트인지 확인
                        if isinstance(flashrank_result, dict) and "results" in flashrank_result:
                            # 딕셔너리 형태로 반환된 경우
                            flashrank_scores = [p.get("score", 0.0) for p in flashrank_result["results"]]
                            logger.info(f"[HYBRID-DETAIL] FlashRank 재랭킹 완료 (딕셔너리 형태), 결과 수: {len(flashrank_scores)}")
                            
                            # 점수 분포 로깅
                            if flashrank_scores:
                                min_score = min(flashrank_scores)
                                max_score = max(flashrank_scores)
                                avg_score = sum(flashrank_scores) / len(flashrank_scores)
                                logger.info(f"[HYBRID-DETAIL] FlashRank 점수 분포: 최소={min_score:.4f}, 최대={max_score:.4f}, 평균={avg_score:.4f}")
                                
                                # 상위 3개 점수 로깅
                                top_scores = sorted(flashrank_scores, reverse=True)[:3]
                                logger.info(f"[HYBRID-DETAIL] FlashRank 상위 3개 점수: {[f'{score:.4f}' for score in top_scores]}")
                                
                                # 상세 로그 파일에 기록
                                try:
                                    with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] FlashRank 재랭킹 완료: {flashrank_time:.3f}초\n")
                                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] FlashRank 점수 분포: 최소={min_score:.4f}, 최대={max_score:.4f}, 평균={avg_score:.4f}\n")
                                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] FlashRank 상위 3개 점수: {[f'{score:.4f}' for score in top_scores]}\n")
                                except Exception as e:
                                    logger.warning(f"[HYBRID-DETAIL] 상세 로그 파일 기록 실패: {str(e)}")
                        
                        elif isinstance(flashrank_result, list):
                            # 리스트 형태로 반환된 경우
                            flashrank_scores = [p.get("score", 0.0) for p in flashrank_result]
                            # 딕셔너리 형태로 변환
                            flashrank_result = {
                                "query": query,
                                "results": flashrank_result,
                                "total": len(flashrank_result),
                                "reranked": True,
                                "reranker_type": "flashrank"
                            }
                            logger.info(f"[HYBRID-DETAIL] FlashRank 재랭킹 완료 (리스트 형태), 결과 수: {len(flashrank_scores)}")
                        else:
                            # 예상치 못한 형태인 경우
                            logger.error(f"[HYBRID-DETAIL] 예상치 못한 FlashRank 결과 형식: {type(flashrank_result)}")
                            # 안전하게 빈 리스트로 초기화
                            flashrank_result = {
                                "query": query,
                                "results": passages,
                                "total": len(passages),
                                "reranked": False
                            }
                            flashrank_scores = [p.get("meta", {}).get("original_score", 0.5) for p in passages]
                    else:
                        logger.warning("[HYBRID-DETAIL] FlashRank 재랭커가 초기화되지 않아 MRC만 사용합니다")
                        # FlashRank 결과가 없으면 원본 결과를 사용
                        flashrank_result = {
                            "query": query,
                            "results": passages,
                            "total": len(passages),
                            "reranked": False
                        }
                        # 원본 점수 또는 기본값 사용
                        flashrank_scores = [p.get("meta", {}).get("original_score", 0.5) for p in passages]
                    
                    # 하이브리드 재랭킹 수행
                    logger.info("[HYBRID-DETAIL] MRC 하이브리드 재랭킹 시작")
                    hybrid_start_time = time.time()
                    
                    # MRC 리랭커 존재 여부 확인 (불필요한 예외 방지)
                    if not hasattr(self.mrc_reranker, 'hybrid_rerank'):
                        logger.error("[HYBRID-DETAIL] MRC 리랭커에 hybrid_rerank 메소드가 없습니다.")
                        raise AttributeError("hybrid_rerank method missing in MRC reranker")
                    
                    # MRC 가중치 로깅
                    logger.info(f"[HYBRID-DETAIL] MRC 가중치: {self.hybrid_weight_mrc}, FlashRank 가중치: {1.0 - self.hybrid_weight_mrc}")
                    
                    # 하이브리드 재랭킹 수행 - 전체 검색 결과 사용 (제한 없이 모든 결과 처리)
                    logger.info(f"[HYBRID-DETAIL] MRC 재랭킹 시작: 총 {len(flashrank_result['results'])}개 항목 처리")
                    
                    # 처리 대상 항목 수 로깅
                    try:
                        with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] MRC 재랭킹 시작: 총 {len(flashrank_result['results'])}개 항목 처리\n")
                    except Exception as e:
                        logger.error(f"로그 파일 기록 실패: {str(e)}")
                    
                    # 모든 결과 처리를 위해 top_k=None으로 설정
                    reranked_passages, mrc_scores = self.mrc_reranker.hybrid_rerank(
                        query, 
                        flashrank_result["results"], 
                        flashrank_scores, 
                        weight_mrc=self.hybrid_weight_mrc,
                        top_k=None,  # None으로 설정하여 모든 결과 처리
                        return_mrc_scores=True  # MRC 점수도 함께 반환
                    )
                    mrc_processing_time = time.time() - hybrid_start_time
                    logger.info(f"[HYBRID-DETAIL] MRC 하이브리드 재랭킹 완료, 소요 시간: {mrc_processing_time:.3f}초, 결과 수: {len(reranked_passages)}")
                    
                    # MRC 점수 확인
                    if mrc_scores:
                        # 점수 분포 로깅
                        min_score = min(mrc_scores)
                        max_score = max(mrc_scores)
                        avg_score = sum(mrc_scores) / len(mrc_scores)
                        logger.info(f"[HYBRID-DETAIL] MRC 점수 분포: 최소={min_score:.4f}, 최대={max_score:.4f}, 평균={avg_score:.4f}")
                        
                        # 상위 3개 점수 로깅
                        top_indices = sorted(range(len(mrc_scores)), key=lambda i: mrc_scores[i], reverse=True)[:3]
                        top_mrc_scores = [mrc_scores[i] for i in top_indices]
                        logger.info(f"[HYBRID-DETAIL] MRC 상위 3개 점수: {[f'{score:.4f}' for score in top_mrc_scores]}")
                        
                        # 상세 로그 파일에 기록
                        try:
                            with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] MRC 하이브리드 재랭킹 완료: {mrc_processing_time:.3f}초\n")
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] MRC 점수 분포: 최소={min_score:.4f}, 최대={max_score:.4f}, 평균={avg_score:.4f}\n")
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] MRC 상위 3개 점수: {[f'{score:.4f}' for score in top_mrc_scores]}\n")
                        except Exception as e:
                            logger.warning(f"[HYBRID-DETAIL] 상세 로그 파일 기록 실패: {str(e)}")
                    
                except Exception as e:
                    logger.error(f"[HYBRID-DETAIL] 하이브리드 재랭킹 중 오류 발생: {str(e)}", exc_info=True)
                    
                    # 오류 상세 정보 로깅
                    try:
                        with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] 하이브리드 재랭킹 오류: {str(e)}\n")
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {traceback.format_exc()}\n")
                    except Exception as log_error:
                        logger.warning(f"[HYBRID-DETAIL] 상세 로그 파일 오류 기록 실패: {str(log_error)}")
                    
                    # FlashRank가 초기화되었으면 FlashRank로 대체
                    if self.ranker is not None:
                        logger.error("[HYBRID-DETAIL] FlashRank 방식으로 대체 수행합니다.")
                        return self.perform_flashrank_reranking(query, passages, top_k, search_result)
                    else:
                        # 둘 다 사용할 수 없으면 원본 결과 반환
                        logger.error("[HYBRID-DETAIL] 모든 재랭커를 사용할 수 없어 원본 결과를 반환합니다.")
                        return search_result
                
                # 결과에 세부 점수 추가
                logger.info("[HYBRID-DETAIL] 결과에 세부 점수 추가 시작")
                combined_scores = []
                
                for i, passage in enumerate(reranked_passages):
                    if "metadata" not in passage:
                        passage["metadata"] = {}
                    
                    # 원본 메타데이터 유지하면서 세부 점수 추가
                    metadata = passage.get("metadata", {})
                    
                    # FlashRank 점수 추가
                    flashrank_score = float(flashrank_scores[i]) if i < len(flashrank_scores) else 0.0
                    metadata["flashrank_score"] = flashrank_score
                    
                    # MRC 점수 추가
                    mrc_score = float(mrc_scores[i]) if i < len(mrc_scores) else 0.0
                    metadata["mrc_score"] = mrc_score
                    
                    # 최종 점수 계산 (이미 계산되어 있지만 로깅용으로 다시 계산)
                    combined_score = (1.0 - self.hybrid_weight_mrc) * flashrank_score + self.hybrid_weight_mrc * mrc_score
                    combined_scores.append(combined_score)
                    
                    # 메타데이터 업데이트
                    passage["metadata"] = metadata
                
                # 상위 3개 결과의 점수 비교 로깅
                if len(reranked_passages) >= 3:
                    for i in range(3):
                        passage = reranked_passages[i]
                        fr_score = passage["metadata"]["flashrank_score"]
                        mrc_score = passage["metadata"]["mrc_score"]
                        final_score = passage["score"]
                        logger.info(f"[HYBRID-DETAIL] 상위 결과 #{i+1}: FlashRank={fr_score:.4f}, MRC={mrc_score:.4f}, 최종={final_score:.4f}")
                        
                        # 상세 로그 파일에 기록
                        try:
                            with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                                passage_preview = passage["text"][:100] + "..." if len(passage["text"]) > 100 else passage["text"]
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] 상위 결과 #{i+1}: FlashRank={fr_score:.4f}, MRC={mrc_score:.4f}, 최종={final_score:.4f}\n")
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] 패시지 #{i+1}: {passage_preview}\n")
                        except Exception as e:
                            logger.warning(f"[HYBRID-DETAIL] 상세 로그 파일 기록 실패: {str(e)}")
                
                # 결과 포맷팅
                result = {
                    "query": query,
                    "results": reranked_passages,
                    "total": len(reranked_passages),
                    "reranked": True,
                    "reranker_type": "hybrid",  # hybrid로 명확하게 표시
                    "processing_time": time.time() - start_time,
                    "flashrank_time": flashrank_time,
                    "mrc_time": mrc_processing_time,
                    "mrc_weight": self.hybrid_weight_mrc
                }
                
                # 로그에 하이브리드 재랭킹 결과 기록
                logger.info(f"[HYBRID-DETAIL] 하이브리드 재랭킹 완료: 결과 수={len(reranked_passages)}, MRC 가중치={self.hybrid_weight_mrc}")
                logger.info(f"[HYBRID-DETAIL] 총 처리 시간: {result['processing_time']:.3f}초 (FlashRank: {flashrank_time:.3f}초, MRC: {mrc_processing_time:.3f}초)")
                
                # 메타데이터 중복 제거 - 각 결과 항목에서 metadata 내부의 필드를 상위 레벨로 이동하고 metadata 필드 제거
                for passage in result["results"]:
                    # 중요 점수 정보를 상위 레벨로 명확하게 복사
                    if "score" in passage:
                        passage["hybrid_score"] = passage["score"]  # 기본 score를 hybrid_score로 명확하게 복사
                    
                    # MRC 관련 필드 상위 레벨로 복사 (있는 경우)
                    if "mrc_score" in passage:
                        passage["mrc_score"] = passage["mrc_score"]
                    if "mrc_answer" in passage:
                        passage["mrc_answer"] = passage["mrc_answer"]
                    if "mrc_char_ids" in passage:
                        passage["mrc_char_ids"] = passage["mrc_char_ids"]
                    
                    # FlashRank 점수 상위 레벨로 복사 (있는 경우)
                    if "flashrank_score" in passage:
                        passage["flashrank_score"] = passage["flashrank_score"]
                    
                    # metadata 필드 처리
                    if "metadata" in passage:
                        # metadata 내부의 모든 필드를 상위 레벨로 복사
                        for key, value in passage["metadata"].items():
                            if key not in passage:  # 이미 존재하는 필드는 덮어쓰지 않음
                                passage[key] = value
                                
                            # 중요 점수 정보는 상위 레벨로 명확하게 복사
                            if key == "flashrank_score" and "flashrank_score" not in passage:
                                passage["flashrank_score"] = value
                            elif key == "mrc_score" and "mrc_score" not in passage:
                                passage["mrc_score"] = value
                            elif key == "original_score" and "original_score" not in passage:
                                passage["original_score"] = value
                        
                        # metadata 필드 제거
                        del passage["metadata"]
                    
                    # MRC 관련 필드가 있는지 확인하고 없으면 기본값 설정
                    if "mrc_score" in passage and "mrc_answer" not in passage:
                        passage["mrc_answer"] = ""
                    if "mrc_score" in passage and "mrc_char_ids" not in passage:
                        passage["mrc_char_ids"] = []
                
                # 최종 응답에서 top_k개만 반환
                if top_k and isinstance(top_k, int) and top_k > 0 and top_k < len(reranked_passages):
                    original_count = len(reranked_passages)
                    reranked_passages = reranked_passages[:top_k]
                    result["results"] = reranked_passages
                    result["filtered_count"] = original_count
                    result["returned_count"] = top_k
                    logger.info(f"[HYBRID-DETAIL] 최종 응답 필터링: 전체 {original_count}개 중 상위 {top_k}개만 반환")
                    
                    # 상세 로그 파일에 기록
                    try:
                        with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] 최종 응답 필터링: 전체 {original_count}개 중 상위 {top_k}개만 반환\n")
                    except Exception as e:
                        logger.warning(f"[HYBRID-DETAIL] 상세 로그 파일 기록 실패: {str(e)}")
                
                # 최종 로그 기록
                try:
                    with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] 하이브리드 재랭킹 완료: 총 처리 시간={result['processing_time']:.3f}초\n")
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] 처리 세부 시간: FlashRank={flashrank_time:.3f}초, MRC={mrc_processing_time:.3f}초\n")
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] 결과 수: {len(reranked_passages)}개, MRC 가중치: {self.hybrid_weight_mrc}\n")
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [HYBRID-DETAIL] ===== 하이브리드 재랭킹 처리 완료 =====\n\n")
                except Exception as e:
                    logger.warning(f"[HYBRID-DETAIL] 상세 로그 파일 기록 실패: {str(e)}")
                
                return result
                
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # 오류 발생 시 원본 결과 반환
            return search_result
    
    def _flashrank_rerank(self, query: str, passages: List[dict], top_k: int = None, search_result: Dict = None) -> Tuple[List[dict], Dict, float]:
        """
        Rerank passages using FlashRank
        
        Args:
            query: Query string
            passages: List of passages to rerank
            top_k: Number of top passages to return, if None, return all
            search_result: 원본 검색 결과 (있는 경우)
            
        Returns:
            Tuple of (reranked passages, scores dict, processing time)
        """
        if not self.ranker:
            logger.warning("FlashRank reranker not initialized, returning original passages")
            return passages, {}, 0.0
        
        if not passages:
            logger.warning("No passages to rerank")
            return [], {}, 0.0
        
        start_time = time.time()
        
        try:
            # 모델 디바이스 재확인 (GPU 사용 중인지)
            if self.device == "cuda" and hasattr(self.ranker, 'model'):
                model_device = next(self.ranker.model.parameters()).device
                if str(model_device) == "cpu":
                    logger.warning("FlashRank model is still on CPU! Moving to GPU...")
                    try:
                        self.ranker.model.to('cuda')
                        new_device = next(self.ranker.model.parameters()).device
                        logger.info(f"FlashRank model moved to: {new_device}")
                    except Exception as e:
                        logger.error(f"Failed to move FlashRank model to GPU: {str(e)}")
            
            # 대량 패시지 처리 최적화
            total_passages = len(passages)
            logger.info(f"Reranking {total_passages} passages for query: '{query}'")
            
            # 배치 처리를 위한 최적 크기 계산 - 성능 최적화
            batch_size = min(64, total_passages)  # 배치 크기 제한
            logger.info(f"[FLASHRANK-DETAIL] 배치 크기 설정: {batch_size}, 총 패시지 수: {total_passages}")
            
            # GPU 또는 CPU 사용 여부 확인 및 로깅
            device_info = "GPU" if torch.cuda.is_available() else "CPU"
            logger.info(f"[FLASHRANK-DETAIL] 현재 사용 중인 디바이스: {device_info}")
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"[FLASHRANK-DETAIL] GPU 정보: {device_name}")
            
            # 상세 로그 파일에 기록
            try:
                with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] 배치 크기: {batch_size}, 총 패시지 수: {total_passages}, 디바이스: {device_info}\n")
                    if torch.cuda.is_available():
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] GPU 정보: {device_name}\n")
                        # GPU 메모리 사용량 기록
                        mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                        mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB 단위
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] GPU 메모리 사용량: {mem_allocated:.2f}GB (할당) / {mem_reserved:.2f}GB (예약)\n")
            except Exception as e:
                logger.error(f"로그 파일 기록 실패: {str(e)}")
            
            # 배치 처리 수행
            num_batches = (total_passages + batch_size - 1) // batch_size
            logger.info(f"Processing in {num_batches} batches")
            
            reranked_results = []
            scores_dict = {}  # 점수 저장을 위한 딕셔너리
            
            for i in range(0, total_passages, batch_size):
                batch_start = time.time()
                batch_end = min(i + batch_size, total_passages)
                batch_passages = passages[i:batch_end]
                
                logger.debug(f"Processing batch {i//batch_size + 1}/{num_batches} with {len(batch_passages)} passages")
                
                # 배치 시작 로깅
                logger.info(f"[FLASHRANK-DETAIL] 배치 {i//batch_size + 1}/{num_batches} 처리 시작: {len(batch_passages)}개 항목")
                try:
                    with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] 배치 {i//batch_size + 1}/{num_batches} 처리 시작: {len(batch_passages)}개 항목\n")
                except Exception as e:
                    logger.error(f"로그 파일 기록 실패: {str(e)}")
                
                # FlashRank 요청 구성
                try:
                    # 요청 객체 생성
                    request = RerankRequest(
                        query=query,
                        passages=[{"id": str(idx), "text": p["text"], "meta": p.get("meta", {})} for idx, p in enumerate(batch_passages)]
                    )
                    
                    # GPU 사용 중인지 확인
                    if self.device == "cuda" and hasattr(self.ranker, 'model'):
                        # 추론 전에 CUDA 캐시 정리
                        torch.cuda.empty_cache()
                        
                        # GPU 메모리 상태 로깅
                        mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                        logger.info(f"[FLASHRANK-DETAIL] 배치 처리 전 GPU 메모리: {mem_allocated:.2f}GB")
                    
                    # 재랭킹 수행
                    batch_results = self.ranker.rerank(request)
                    
                    # CUDA 동기화 (비동기 작업 완료 대기)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    
                    # 배치 처리 시간 계산
                    batch_time = (time.time() - batch_start) * 1000  # ms 단위
                    passages_per_sec = len(batch_passages) / (batch_time / 1000)
                    logger.debug(f"Batch {i//batch_size + 1} completed in {batch_time:.2f}ms ({passages_per_sec:.1f} passages/sec)")
                    
                    # 배치 처리 결과 로깅
                    try:
                        with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] 배치 {i//batch_size + 1} 완료: {batch_time:.2f}ms ({passages_per_sec:.1f} passages/sec)\n")
                            if self.device == "cuda":
                                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] 배치 처리 후 GPU 메모리: {mem_allocated:.2f}GB\n")
                    except Exception as e:
                        logger.error(f"로그 파일 기록 실패: {str(e)}")
                    
                    # 결과 처리
                    for rank, scored_passage in enumerate(batch_results):
                        passage_id = int(scored_passage["id"])
                        original_passage = batch_passages[passage_id]
                        score = scored_passage["score"]
                        
                        # 원본 패시지에 점수 및 순위 정보 추가
                        original_passage["score"] = score
                        original_passage["rank"] = rank + i  # 전체 순위 계산
                        
                        # 결과 및 점수 저장
                        reranked_results.append(original_passage)
                        scores_dict[i + passage_id] = score
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")
                    # 오류 발생 시 원본 패시지 사용
                    for idx, passage in enumerate(batch_passages):
                        passage["score"] = 0.0
                        passage["rank"] = i + idx
                        reranked_results.append(passage)
            
            # 전체 처리 시간 계산
            processing_time = time.time() - start_time
            logger.info(f"FlashRank 재랭킹 완료: {processing_time:.3f}초")
            
            # 결과 정렬 및 상위 결과 선택
            reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # top_k가 지정된 경우 결과 제한
            if top_k is not None and top_k > 0:
                reranked_results = reranked_results[:top_k]
            
            # 결과 포맷팅
            if search_result:
                search_result["results"] = reranked_results
                search_result["total"] = len(reranked_results)
                search_result["reranked"] = True
                search_result["reranker_type"] = "flashrank"
                search_result["processing_time"] = processing_time
                
                # 메타데이터 중복 제거 - 각 결과 항목에서 metadata 내부의 필드를 상위 레벨로 이동하고 metadata 필드 제거
                for passage in search_result["results"]:
                    if "metadata" in passage:
                        # metadata 내부의 모든 필드를 상위 레벨로 복사
                        for key, value in passage["metadata"].items():
                            if key not in passage:  # 이미 존재하는 필드는 덮어쓰지 않음
                                passage[key] = value
                        # metadata 필드 제거
                        del passage["metadata"]
                    
                    # MRC 관련 필드가 있는지 확인하고 없으면 기본값 설정
                    if "mrc_answer" not in passage and passage.get("mrc_score") is not None:
                        passage["mrc_answer"] = ""
                    if "mrc_char_ids" not in passage and passage.get("mrc_score") is not None:
                        passage["mrc_char_ids"] = []
                
                return search_result
            else:
                result = {
                    "query": query,
                    "results": reranked_results,
                    "total": len(reranked_results),
                    "reranked": True,
                    "reranker_type": "flashrank",
                    "processing_time": processing_time
                }
                
                # 메타데이터 중복 제거 - 각 결과 항목에서 metadata 내부의 필드를 상위 레벨로 이동하고 metadata 필드 제거
                for passage in result["results"]:
                    if "metadata" in passage:
                        # metadata 내부의 모든 필드를 상위 레벨로 복사
                        for key, value in passage["metadata"].items():
                            if key not in passage:  # 이미 존재하는 필드는 덮어쓰지 않음
                                passage[key] = value
                        # metadata 필드 제거
                        del passage["metadata"]
                    
                    # MRC 관련 필드가 있는지 확인하고 없으면 기본값 설정
                    if "mrc_answer" not in passage and passage.get("mrc_score") is not None:
                        passage["mrc_answer"] = ""
                    if "mrc_char_ids" not in passage and passage.get("mrc_score") is not None:
                        passage["mrc_char_ids"] = []
                
                return result
        except Exception as e:
            logger.error(f"Error in _flashrank_rerank: {str(e)}", exc_info=True)
            return passages, {}, 0.0
    
    def perform_flashrank_reranking(self, query: str, passages: List[Dict], top_k: int = None, search_result: Dict = None):
        """FlashRank를 사용한 재랭킹 수행"""
        try:
            # 시작 시간 기록
            start_time = time.time()
            
            # 대량 패시지 처리 최적화
            total_passages = len(passages)
            logger.info(f"Reranking {total_passages} passages for query: '{query}'")
            
            # 배치 처리를 위한 최적 크기 계산 - 성능 최적화
            batch_size = min(64, total_passages)  # 배치 크기 제한
            logger.info(f"[FLASHRANK-DETAIL] 배치 크기 설정: {batch_size}, 총 패시지 수: {total_passages}")
            
            # GPU 또는 CPU 사용 여부 확인 및 로깅
            device_info = "GPU" if torch.cuda.is_available() else "CPU"
            logger.info(f"[FLASHRANK-DETAIL] 현재 사용 중인 디바이스: {device_info}")
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"[FLASHRANK-DETAIL] GPU 정보: {device_name}")
            
            # 상세 로그 파일에 기록
            try:
                with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] 배치 크기: {batch_size}, 총 패시지 수: {total_passages}, 디바이스: {device_info}\n")
                    if torch.cuda.is_available():
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] GPU 정보: {device_name}\n")
                        # GPU 메모리 사용량 기록
                        mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                        mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB 단위
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] GPU 메모리 사용량: {mem_allocated:.2f}GB (할당) / {mem_reserved:.2f}GB (예약)\n")
            except Exception as e:
                logger.error(f"로그 파일 기록 실패: {str(e)}")
            
            # 배치 처리 수행
            num_batches = (total_passages + batch_size - 1) // batch_size
            logger.info(f"Processing in {num_batches} batches")
            
            reranked_results = []
            scores_dict = {}  # 점수 저장을 위한 딕셔너리
            
            for i in range(0, total_passages, batch_size):
                batch_start = time.time()
                batch_end = min(i + batch_size, total_passages)
                batch_passages = passages[i:batch_end]
                
                logger.debug(f"Processing batch {i//batch_size + 1}/{num_batches} with {len(batch_passages)} passages")
                
                # 배치 시작 로깅
                logger.info(f"[FLASHRANK-DETAIL] 배치 {i//batch_size + 1}/{num_batches} 처리 시작: {len(batch_passages)}개 항목")
                try:
                    with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] 배치 {i//batch_size + 1}/{num_batches} 처리 시작: {len(batch_passages)}개 항목\n")
                except Exception as e:
                    logger.error(f"로그 파일 기록 실패: {str(e)}")
                
                # FlashRank 요청 구성
                try:
                    # 요청 객체 생성
                    request = RerankRequest(
                        query=query,
                        passages=[{"id": str(idx), "text": p["text"], "meta": p.get("meta", {})} for idx, p in enumerate(batch_passages)]
                    )
                    
                    # GPU 사용 중인지 확인
                    if self.device == "cuda" and hasattr(self.ranker, 'model'):
                        # 추론 전에 CUDA 캐시 정리
                        torch.cuda.empty_cache()
                        
                        # GPU 메모리 상태 로깅
                        mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                        logger.info(f"[FLASHRANK-DETAIL] 배치 처리 전 GPU 메모리: {mem_allocated:.2f}GB")
                    
                    # 재랭킹 수행
                    batch_results = self.ranker.rerank(request)
                    
                    # CUDA 동기화 (비동기 작업 완료 대기)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    
                    # 배치 처리 시간 계산
                    batch_time = (time.time() - batch_start) * 1000  # ms 단위
                    passages_per_sec = len(batch_passages) / (batch_time / 1000)
                    logger.debug(f"Batch {i//batch_size + 1} completed in {batch_time:.2f}ms ({passages_per_sec:.1f} passages/sec)")
                    
                    # 배치 처리 결과 로깅
                    try:
                        with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] 배치 {i//batch_size + 1} 완료: {batch_time:.2f}ms ({passages_per_sec:.1f} passages/sec)\n")
                            if self.device == "cuda":
                                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FLASHRANK-DETAIL] 배치 처리 후 GPU 메모리: {mem_allocated:.2f}GB\n")
                    except Exception as e:
                        logger.error(f"로그 파일 기록 실패: {str(e)}")
                    
                    # 결과 처리
                    for rank, scored_passage in enumerate(batch_results):
                        passage_id = int(scored_passage["id"])
                        original_passage = batch_passages[passage_id]
                        score = scored_passage["score"]
                        
                        # 원본 패시지에 점수 및 순위 정보 추가
                        original_passage["score"] = score
                        original_passage["rank"] = rank + i  # 전체 순위 계산
                        
                        # 결과 및 점수 저장
                        reranked_results.append(original_passage)
                        scores_dict[i + passage_id] = score
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")
                    # 오류 발생 시 원본 패시지 사용
                    for idx, passage in enumerate(batch_passages):
                        passage["score"] = 0.0
                        passage["rank"] = i + idx
                        reranked_results.append(passage)
            
            # 전체 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 정렬 및 상위 결과 선택
            if isinstance(reranked_results, list):
                reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                # top_k 제한 제거 - 모든 결과 반환
                # if top_k and isinstance(top_k, int) and top_k > 0:
                #     reranked_results = reranked_results[:top_k]
            
            # 결과 포맷팅
            if search_result:
                search_result["results"] = reranked_results
                search_result["total"] = len(reranked_results)
                search_result["reranked"] = True
                search_result["reranker_type"] = "flashrank"
                search_result["processing_time"] = processing_time
                
                # 메타데이터 중복 제거 - 각 결과 항목에서 metadata 내부의 필드를 상위 레벨로 이동하고 metadata 필드 제거
                for passage in search_result["results"]:
                    if "metadata" in passage:
                        # metadata 내부의 모든 필드를 상위 레벨로 복사
                        for key, value in passage["metadata"].items():
                            if key not in passage:  # 이미 존재하는 필드는 덮어쓰지 않음
                                passage[key] = value
                        # metadata 필드 제거
                        del passage["metadata"]
                    
                    # MRC 관련 필드가 있는지 확인하고 없으면 기본값 설정
                    if "mrc_answer" not in passage and passage.get("mrc_score") is not None:
                        passage["mrc_answer"] = ""
                    if "mrc_char_ids" not in passage and passage.get("mrc_score") is not None:
                        passage["mrc_char_ids"] = []
                
                return search_result
            else:
                result = {
                    "query": query,
                    "results": reranked_results,
                    "total": len(reranked_results),
                    "reranked": True,
                    "reranker_type": "flashrank",
                    "processing_time": processing_time
                }
                
                # 메타데이터 중복 제거 - 각 결과 항목에서 metadata 내부의 필드를 상위 레벨로 이동하고 metadata 필드 제거
                for passage in result["results"]:
                    if "metadata" in passage:
                        # metadata 내부의 모든 필드를 상위 레벨로 복사
                        for key, value in passage["metadata"].items():
                            if key not in passage:  # 이미 존재하는 필드는 덮어쓰지 않음
                                passage[key] = value
                        # metadata 필드 제거
                        del passage["metadata"]
                    
                    # MRC 관련 필드가 있는지 확인하고 없으면 기본값 설정
                    if "mrc_answer" not in passage and passage.get("mrc_score") is not None:
                        passage["mrc_answer"] = ""
                    if "mrc_char_ids" not in passage and passage.get("mrc_score") is not None:
                        passage["mrc_char_ids"] = []
                
                return result
        except Exception as e:
            logger.error(f"Error in perform_flashrank_reranking: {str(e)}", exc_info=True)
            if search_result:
                return search_result
            else:
                return passages 