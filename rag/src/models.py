from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import TextStreamer, GenerationConfig
from openai import OpenAI
from abc import ABC, abstractmethod
import numpy as np
import torch
import warnings
import os
import logging
import time
import threading

# 멀티프로세싱 관련 설정 - 서버에서는 이미 프로세스가 생성된 후에만 이 코드가 실행되므로 
# spawn 메소드를 설정하면 오류가 발생할 수 있음
# 대신 메인 프로세스에서만 GPU 관련 작업을 처리하도록 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU 순서를 일관되게 유지
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 명시적으로 첫 번째 GPU만 사용

# CUDA 초기화 관련 설정
torch.backends.cudnn.benchmark = True           # 성능 향상
torch.backends.cudnn.deterministic = False      # 성능 우선
torch.backends.cuda.matmul.allow_tf32 = True    # TensorFloat-32 활성화 (Ampere 이상)
torch.backends.cudnn.allow_tf32 = True          # TF32 연산 허용

class Model:
    def __init__(self, config):
        self.config = config 
        self.gpu_initialized = False  # GPU 초기화 상태 추적 - 명시적으로 먼저 False로 설정
        self.set_gpu()
       
    def set_gpu(self):
        try:
            if torch.cuda.is_available():
                # CUDA 초기화 전 메모리 정리
                torch.cuda.empty_cache()
                
                # CUDA 초기화 - 첫 사용 시 발생하는 지연 미리 처리
                torch.cuda.init()
                _ = torch.zeros(1).cuda()  # 첫 CUDA 할당 미리 수행
                
                self.device = torch.device("cuda:0")
                self.gpu_initialized = True  # 초기화 성공 시 플래그 설정
                
                # CUDA 상태 출력
                device_name = torch.cuda.get_device_name(0)
                device_props = torch.cuda.get_device_properties(0)
                logging.info(f"GPU initialized: {device_name}")
                logging.info(f"GPU total memory: {device_props.total_memory/1024**2:.2f}MB")
                logging.info(f"GPU compute capability: {device_props.major}.{device_props.minor}")
                logging.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                logging.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
            else:
                logging.warning("No CUDA device available, using CPU")
                self.device = torch.device("cpu")
        except Exception as e:
            logging.error(f"Error initializing GPU: {str(e)}")
            logging.warning("Falling back to CPU due to GPU initialization error")
            self.device = torch.device("cpu")

    def set_random_state(self, seed=42):
        self.random_state = seed


class EmbModel(Model):
    # 클래스 변수로 인스턴스 추적
    _instances_count = 0
    _instances_lock = threading.Lock()
    
    def __init__(self, config):
        super().__init__(config)
        # GPU/CPU에 따른 기본 배치 사이즈 설정
        self.default_batch_sizes = {
            "cpu": 12,
            "gpu": 64  # 더 작은 배치 사이즈로 설정, 안정성 향상
        }
        self._model_loaded = False  # 모델 로드 상태 추적
        # GPU 동시 접근 제한 (환경변수로 설정 가능)
        # MAX_GPU_WORKERS 환경변수로 GPU 동시 접근 수 제한
        self.max_gpu_workers = int(os.getenv('MAX_GPU_WORKERS', '1'))  # 배치 임베딩을 위해 1로 설정
        # 세마포어는 초기화하지 않고 get_gpu_semaphore 메서드에서 획득
        # 활성 작업 수 추적을 위한 변수
        self._active_gpu_tasks = 0
        self._task_lock = threading.Lock()
        logging.info(f"GPU 동시 작업 제한: {self.max_gpu_workers}개")
        # 임베딩 결과 캐시 추가
        self._embedding_cache = {}
        self._cache_size = int(os.getenv('EMBEDDING_CACHE_SIZE', '1000'))
        self._cache_lock = threading.Lock()
        logging.info(f"임베딩 캐시 초기화: 최대 크기 {self._cache_size}개 항목")
        
        # RAW API 전용 캐시 최적화 설정
        self._raw_cache_hits = 0
        self._raw_cache_misses = 0
        self._enable_cache_stats = True  # 캐시 통계 수집 활성화
        
        # 인스턴스 카운터 증가
        with self.__class__._instances_lock:
            self.__class__._instances_count += 1
            self._instance_id = self.__class__._instances_count
            logging.warning(f"EmbModel 인스턴스 생성: ID={self._instance_id}, 총 인스턴스 수={self.__class__._instances_count}")
    
    def set_embbeding_config(self, batch_size=None, max_length=1024):
        # GPU 여부에 따라 기본 배치 사이즈 선택
        if batch_size is None:
            # 디버깅을 위한 상세 로깅 추가
            cuda_available = torch.cuda.is_available()
            gpu_init = self.gpu_initialized
            
            # 각 조건 상태 로깅
            logging.info(f"디버깅 - CUDA 사용 가능: {cuda_available}, GPU 초기화 상태: {gpu_init}")
            
            # 조건 검사
            mode = "gpu" if cuda_available and gpu_init else "cpu"
            
            # 어떤 조건이 실패했는지 로깅
            if not cuda_available:
                logging.warning("CUDA 사용 불가로 CPU 모드 사용")
            elif not gpu_init:
                logging.warning("GPU 초기화 실패로 CPU 모드 사용")
                
            batch_size = self.default_batch_sizes[mode]
            logging.info(f"Using {mode.upper()} mode with batch_size: {batch_size}")
        
        self.emb_config = {
            "batch_size": batch_size,
            "max_length": max_length
        }

    def set_emb_model(self, model_type='bge'):
        if model_type == 'bge':
            from FlagEmbedding import BGEM3FlagModel
            
            # GPU 사용 가능 여부 재확인
            use_gpu = torch.cuda.is_available() and self.gpu_initialized
            device = self.device
            mode = "gpu" if use_gpu else "cpu"
            batch_size = self.default_batch_sizes[mode]
            
            # GPU 상태 자세히 로깅
            logging.info(f"GPU 사용 여부 체크: cuda_available={torch.cuda.is_available()}, gpu_initialized={self.gpu_initialized}, use_gpu={use_gpu}")
            logging.info(f"선택된 디바이스: {device}, 모드: {mode}, 배치 크기: {batch_size}")
            
            # 인스턴스 정보 로깅
            logging.warning(f"인스턴스 {self._instance_id}에서 모델 로드 시작")
            
            model_path = os.getenv('MODEL_PATH', '/rag/models/bge-m3')
            if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
                logging.error(f"모델 파일이 {model_path}에 존재하지 않습니다.")
                raise FileNotFoundError(f"Model files not found in {model_path}")
            
            try:
                logging.info(f"Loading BGE model on {device} with batch_size {batch_size}")
                
                # 메모리 정리
                if use_gpu:
                    torch.cuda.empty_cache()
                
                # 모델 로드 전 메모리 상태 확인
                if use_gpu:
                    before_load = torch.cuda.memory_allocated()/1024**2
                    before_reserved = torch.cuda.memory_reserved()/1024**2
                    logging.warning(f"GPU Memory before model load: allocated={before_load:.2f}MB, reserved={before_reserved:.2f}MB")
                
                # 모델 로드 시 타임아웃 설정 (안전장치)
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Model loading timed out")
                
                # 타임아웃 설정 (120초)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)
                
                try:
                    # 모델 생성
                    self.bge_emb = BGEM3FlagModel(
                        model_path,
                        use_fp16=use_gpu,
                        device=device,
                        compute_dtype=torch.float16 if use_gpu else torch.float32,
                        batch_size=batch_size
                    )
                    # 타임아웃 비활성화
                    signal.alarm(0)
                    self._model_loaded = True
                except TimeoutError:
                    logging.error("Model loading timed out, trying with smaller batch")
                    signal.alarm(0)
                    # 더 작은 배치 사이즈로 재시도
                    batch_size = max(1, batch_size // 4)
                    self.bge_emb = BGEM3FlagModel(
                        model_path,
                        use_fp16=use_gpu,
                        device=device,
                        compute_dtype=torch.float16 if use_gpu else torch.float32,
                        batch_size=batch_size
                    )
                    self._model_loaded = True
                
                # 모델 로드 후 메모리 상태 확인
                if use_gpu:
                    after_load = torch.cuda.memory_allocated()/1024**2
                    after_reserved = torch.cuda.memory_reserved()/1024**2
                    logging.warning(f"GPU Memory after model load: allocated={after_load:.2f}MB, reserved={after_reserved:.2f}MB")
                    logging.warning(f"Model size in memory: {after_load - before_load:.2f}MB")
                
                logging.info(f"Successfully loaded BGE model on {device}")
                
                # 모델 워밍업 (첫 추론 시간 단축)
                logging.info("Performing model warmup...")
                with torch.no_grad():
                    _ = self.bge_emb.encode("워밍업 텍스트", max_length=128)['dense_vecs']
                logging.info("Model warmup completed")
                
                # 모델 로드 완료 로깅
                logging.warning(f"인스턴스 {self._instance_id}에서 모델 로드 완료")
                
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                if use_gpu:
                    logging.warning("GPU error encountered. Trying with CPU...")
                    self.device = torch.device("cpu")
                    batch_size = self.default_batch_sizes["cpu"]
                    # CPU로 재시도
                    self.bge_emb = BGEM3FlagModel(
                        model_path,
                        use_fp16=False,
                        device=torch.device("cpu"),
                        compute_dtype=torch.float32,
                        batch_size=batch_size
                    )
                    self._model_loaded = True
                    logging.info(f"Successfully loaded BGE model on CPU as fallback")
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # 세마포어 관련 메서드 추가
    def get_gpu_semaphore(self):
        """GPU 접근을 제한하기 위한 세마포어를 반환합니다."""
        # InteractManager 클래스의 세마포어 사용 (모든 임베딩 처리에서 공유)
        from .pipe import InteractManager
        sem = InteractManager.get_gpu_semaphore()
        
        # 세마포어 현재 값 로깅
        sem_value = InteractManager.get_gpu_semaphore_value()
        active_workers = InteractManager.get_active_workers()
        logging.warning(f"세마포어 정보: 사용 가능={sem_value}, 활성 작업={active_workers}, 최대={InteractManager.get_max_workers()}")
        
        return sem
            
    def embed_text(self, text, model_type=None):
        """텍스트에 임베딩을 적용합니다. 임베딩 벡터를 반환합니다."""
        # 모델 유형 설정 (기본값은 인스턴스 변수에서 가져옴)
        if model_type is None:
            model_type = getattr(self, 'model_type', 'bge')
        
        # 인스턴스 정보 로깅
        logging.info(f"임베딩 요청: 인스턴스 ID={self._instance_id}, 텍스트 길이={len(text)}")
        
        # 캐시 확인 (짧은 텍스트만 캐싱)
        if len(text) < 1000:
            cache_key = f"{model_type}:{text}"
            with self._cache_lock:
                if cache_key in self._embedding_cache:
                    # 캐시 히트
                    logging.info(f"캐시 히트: 인스턴스 ID={self._instance_id}")
                    return self._embedding_cache[cache_key]
        
        # 세마포어 획득
        sem = self.get_gpu_semaphore()
        sem_acquired = False
        
        try:
            # 임베딩 시작 시간 기록
            start_time = time.time()
            
            # 세마포어 획득 시도
            sem_timeout = 60  # 세마포어 획득 최대 대기 시간 (초)
            logging.info(f"세마포어 획득 시도: 인스턴스 ID={self._instance_id}")
            sem_acquired = sem.acquire(timeout=sem_timeout)
            
            if not sem_acquired:
                logging.warning(f"GPU 세마포어 획득 실패: 인스턴스 ID={self._instance_id}, 제한 시간 초과")
                # 세마포어 획득 실패해도 계속 진행 (성능 저하 가능성)
            else:
                logging.info(f"세마포어 획득 성공: 인스턴스 ID={self._instance_id}")
            
            # GPU 메모리 상태 로깅
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()/1024**2
                reserved = torch.cuda.memory_reserved()/1024**2
                logging.info(f"GPU 메모리 상태: 할당={allocated:.2f}MB, 예약={reserved:.2f}MB")
            
            # 활성 작업 수 증가
            with self._task_lock:
                self._active_gpu_tasks += 1
                logging.info(f"활성 작업 증가: 인스턴스 ID={self._instance_id}, 작업 수={self._active_gpu_tasks}")
            
            # 임베딩 생성 - 모델 타입에 따라 분기
            compute_start = time.time()
            
            if model_type == 'bge':
                # BGE 모델 확인
                if not hasattr(self, 'bge_emb') or self.bge_emb is None:
                    logging.warning(f"모델 로드 필요: 인스턴스 ID={self._instance_id}")
                    self.set_emb_model('bge')
                
                # BGE 모델로 임베딩 계산
                with torch.no_grad():
                    logging.info(f"임베딩 계산 시작: 인스턴스 ID={self._instance_id}")
                    result = self.bge_emb.encode(text, max_length=self.emb_config.get('max_length', 1024))
                    logging.info(f"임베딩 계산 완료: 인스턴스 ID={self._instance_id}")
                
                # 벡터 추출 및 numpy 배열로 변환
                if isinstance(result, dict) and 'dense_vecs' in result:
                    embedding_vector = result['dense_vecs']
                else:
                    embedding_vector = result
                
                # numpy 배열을 파이썬 리스트로 변환 (JSON 직렬화 가능하도록)
                if isinstance(embedding_vector, np.ndarray):
                    embedding_vector = embedding_vector.tolist()
                
                # 벡터 차원 검증 및 조정
                expected_dim = 1024
                if len(embedding_vector) != expected_dim:
                    logging.info(f"벡터 길이 부족하여 패딩: {len(embedding_vector)} → {expected_dim}")
                    # 부족한 차원은 0으로 채움
                    if len(embedding_vector) < expected_dim:
                        embedding_vector.extend([0.0] * (expected_dim - len(embedding_vector)))
                    # 초과 차원은 잘라냄
                    else:
                        embedding_vector = embedding_vector[:expected_dim]
            
            else:
                # 지원되지 않는 모델 타입
                logging.warning(f"지원되지 않는 임베딩 모델 타입: {model_type}")
                # 기본 임베딩 (모두 0으로 채워진 1024차원 벡터)
                embedding_vector = [0.0] * 1024
            
            compute_end = time.time()
            compute_time = compute_end - compute_start
            
            # 임베딩 완료 시간 계산
            end_time = time.time()
            total_time = end_time - start_time
            
            # 장시간 소요된 경우 경고 로그
            if total_time > 5.0:
                logging.warning(f"⚠️ 비정상적으로 긴 임베딩 처리 시간: {total_time:.2f}초, 인스턴스 ID={self._instance_id}")
            
            # 일반 로그 (디버깅용)
            logging.info(f"임베딩 완료: 인스턴스 ID={self._instance_id}, 총 {total_time:.4f}초, 계산={compute_time:.4f}초")
            
            # GPU 메모리 상태 다시 로깅
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()/1024**2
                reserved = torch.cuda.memory_reserved()/1024**2
                logging.info(f"임베딩 후 GPU 메모리: 할당={allocated:.2f}MB, 예약={reserved:.2f}MB")
            
            # 캐시 저장 (짧은 텍스트만)
            if len(text) < 1000:
                with self._cache_lock:
                    # 캐시 크기 제한 확인
                    if len(self._embedding_cache) >= self._cache_size:
                        # 오래된 항목 하나 제거 (FIFO)
                        try:
                            oldest_key = next(iter(self._embedding_cache))
                            self._embedding_cache.pop(oldest_key)
                        except Exception:
                            pass
                    
                    # 캐시에 저장
                    self._embedding_cache[cache_key] = embedding_vector
            
            return embedding_vector
            
        except Exception as e:
            logging.error(f"임베딩 생성 오류: 인스턴스 ID={self._instance_id}, 오류={str(e)}")
            # 오류 발생 시 0 벡터 반환
            return [0.0] * 1024
            
        finally:
            # 활성 작업 수 감소
            with self._task_lock:
                self._active_gpu_tasks = max(0, self._active_gpu_tasks - 1)
                logging.info(f"활성 작업 감소: 인스턴스 ID={self._instance_id}, 작업 수={self._active_gpu_tasks}")
            
            # 세마포어 반환 (획득한 경우에만)
            if sem_acquired:
                try:
                    sem.release()
                    logging.info(f"세마포어 반환: 인스턴스 ID={self._instance_id}")
                except Exception as release_error:
                    logging.error(f"세마포어 반환 오류: 인스턴스 ID={self._instance_id}, 오류={str(release_error)}")
            
            # GPU 메모리 정리 (다른 작업이 없을 때만)
            with self._task_lock:
                if self._active_gpu_tasks == 0 and torch.cuda.is_available():
                    logging.info(f"GPU 메모리 정리 시작: 인스턴스 ID={self._instance_id}")
                    torch.cuda.empty_cache()
                    logging.info(f"GPU 메모리 정리 완료: 인스턴스 ID={self._instance_id}")

    def calc_emb_similarity(self, emb1, emb2, metric='L2'):
        if metric == 'L2':   # Euclidean distance
            l2_distance = np.linalg.norm(emb1 - emb2)
            return l2_distance

    @abstractmethod
    def get_hf_encoder(self):
        pass

    @abstractmethod 
    def get_cohere_encoder(self, cohere_api):
        pass

    def bge_embed_data(self, text):
        """
        임베딩 벡터 생성 메서드입니다. embed_text를 호출하여 동일한 기능을 수행합니다.
        텍스트를 받아 임베딩 벡터를 반환합니다.
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            list: 임베딩 벡터 (1024 차원)
        """
        # 로깅 추가 - 메서드 호출 정보
        logging.info(f"bge_embed_data 호출됨: 인스턴스 ID={self._instance_id}, 텍스트 길이 {len(text)}")
        
        # 시작 시간 기록
        start_time = time.time()
        
        # embed_text 메서드 호출
        result = self.embed_text(text, model_type='bge')
        
        # 완료 시간 및 소요 시간 계산
        end_time = time.time()
        duration = end_time - start_time
        
        # 결과 및 성능 로깅
        logging.info(f"bge_embed_data 완료: 인스턴스 ID={self._instance_id}, 소요 시간 {duration:.4f}초")
        
        return result
    
    def bge_embed_data_raw(self, text):
        """
        RAW API 전용 임베딩 벡터 생성 메서드입니다. 
        세마포어를 사용하지 않고 GPU에 직접 접근하여 빠르게 임베딩을 생성합니다.
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            list: 임베딩 벡터 (1024 차원)
        """
        # 로깅 추가 - 메서드 호출 정보
        logging.info(f"bge_embed_data_raw 호출됨: 인스턴스 ID={self._instance_id}, 텍스트 길이 {len(text)}")
        
        # 시작 시간 기록
        start_time = time.time()
        
        # 캐시 확인 - 텍스트가 이미 임베딩된 경우 캐시된 값 반환
        text_hash = hash(text)
        with self._cache_lock:
            if text_hash in self._embedding_cache:
                cached_result = self._embedding_cache[text_hash]
                logging.info(f"캐시 적중: 인스턴스 ID={self._instance_id}, 텍스트 길이 {len(text)}")
                # 캐시 통계 업데이트
                if self._enable_cache_stats:
                    self._raw_cache_hits += 1
                    if self._raw_cache_hits % 10 == 0:  # 10번마다 로깅
                        hit_rate = self._raw_cache_hits / (self._raw_cache_hits + self._raw_cache_misses) * 100 if (self._raw_cache_hits + self._raw_cache_misses) > 0 else 0
                        logging.info(f"RAW 캐시 통계: 적중={self._raw_cache_hits}, 실패={self._raw_cache_misses}, 적중률={hit_rate:.1f}%")
                return cached_result
        
        # 캐시 실패 통계 업데이트
        if self._enable_cache_stats:
            self._raw_cache_misses += 1
        
        try:
            # GPU 사용 가능 여부 확인
            if not torch.cuda.is_available() or not self.gpu_initialized:
                logging.warning(f"GPU 사용 불가: 인스턴스 ID={self._instance_id}, cuda_available={torch.cuda.is_available()}, gpu_initialized={self.gpu_initialized}")
                # CPU 모드로 임베딩 생성
                return self.embed_text(text, model_type='bge')
            
            # BGE 모델 확인
            if not hasattr(self, 'bge_emb') or self.bge_emb is None:
                logging.warning(f"모델 로드 필요: 인스턴스 ID={self._instance_id}")
                self.set_emb_model('bge')
            
            # GPU 메모리 상태 로깅
            allocated_before = torch.cuda.memory_allocated()/1024**2
            reserved_before = torch.cuda.memory_reserved()/1024**2
            logging.info(f"임베딩 전 GPU 메모리: 할당={allocated_before:.2f}MB, 예약={reserved_before:.2f}MB")
            
            # 임베딩 계산 시작
            compute_start = time.time()
            
            # BGE 모델로 직접 임베딩 계산 - 세마포어 없이
            with torch.no_grad():
                logging.info(f"RAW 임베딩 계산 시작: 인스턴스 ID={self._instance_id}")
                
                # 직접 텍스트를 리스트로 감싸서 encode 호출 (단일 항목 배치 처리)
                result = self.bge_emb.encode([text], max_length=self.emb_config.get('max_length', 1024))
                
                logging.info(f"RAW 임베딩 계산 완료: 인스턴스 ID={self._instance_id}")
            
            # 결과 추출 및 변환
            if isinstance(result, dict) and 'dense_vecs' in result:
                dense_vecs = result['dense_vecs']
                if isinstance(dense_vecs, np.ndarray) and len(dense_vecs.shape) == 2:
                    # 첫 번째 벡터만 추출 (단일 텍스트를 인코딩했으므로)
                    embed_vector = dense_vecs[0].tolist()
                else:
                    logging.warning(f"예상치 못한 결과 형식: {type(dense_vecs)}, 형상: {dense_vecs.shape if hasattr(dense_vecs, 'shape') else 'unknown'}")
                    # 안전하게 처리
                    embed_vector = dense_vecs.tolist() if hasattr(dense_vecs, 'tolist') else [0.0] * 1024
            else:
                logging.warning(f"예상치 못한 결과 형식: {type(result)}")
                # 안전 처리: 임베딩을 0 벡터로 설정
                embed_vector = [0.0] * 1024
            
            # 벡터 길이 확인 및 보정
            expected_dim = 1024
            if len(embed_vector) != expected_dim:
                logging.info(f"벡터 길이 부족하여 패딩: {len(embed_vector)} → {expected_dim}")
                # 부족한 차원은 0으로 채움
                if len(embed_vector) < expected_dim:
                    embed_vector.extend([0.0] * (expected_dim - len(embed_vector)))
                # 초과 차원은 잘라냄
                else:
                    embed_vector = embed_vector[:expected_dim]
            
            compute_end = time.time()
            compute_time = compute_end - compute_start
            
            # 임베딩 완료 시간 계산
            end_time = time.time()
            total_time = end_time - start_time
            
            # 임베딩 후 GPU 메모리 상태 로깅
            allocated_after = torch.cuda.memory_allocated()/1024**2
            reserved_after = torch.cuda.memory_reserved()/1024**2
            logging.info(f"임베딩 후 GPU 메모리: 할당={allocated_after:.2f}MB, 예약={reserved_after:.2f}MB")
            
            # 성능 로깅
            logging.info(f"RAW 임베딩 완료: 인스턴스 ID={self._instance_id}, 총 {total_time:.4f}초, 계산={compute_time:.4f}초")
            
            # 결과 캐싱
            with self._cache_lock:
                # 캐시 크기 제한 확인
                if len(self._embedding_cache) >= self._cache_size:
                    # 임의의 항목 제거 (간단한 구현)
                    try:
                        key_to_remove = next(iter(self._embedding_cache))
                        del self._embedding_cache[key_to_remove]
                    except:
                        # 캐시 비우기 실패 시 모두 제거
                        self._embedding_cache.clear()
                # 결과 캐싱
                self._embedding_cache[text_hash] = embed_vector
            
            return embed_vector
            
        except Exception as e:
            logging.error(f"RAW 임베딩 생성 오류: 인스턴스 ID={self._instance_id}, 오류={str(e)}")
            import traceback
            logging.error(f"스택 트레이스: {traceback.format_exc()}")
            # 오류 발생 시 기존 방식으로 폴백
            return self.embed_text(text, model_type='bge')

    def bge_batch_embed_data(self, texts):
        """
        여러 텍스트를 배치로 처리하여 임베딩 벡터를 생성합니다.
        
        Args:
            texts (list): 임베딩할 텍스트 리스트
            
        Returns:
            list: 임베딩 벡터 리스트 (각각 1024 차원)
        """
        if not texts:
            return []
        
        # 로깅 추가 - 배치 처리 정보
        logging.info(f"bge_batch_embed_data 호출됨: 인스턴스 ID={self._instance_id}, 배치 크기 {len(texts)}")
        
        # 시작 시간 기록
        start_time = time.time()
        
        try:
            # GPU 세마포어 획득 (배치 처리에도 동일하게 적용)
            sem = self.get_gpu_semaphore()
            sem_acquired = False
            
            try:
                # 세마포어 획득 시도 (타임아웃 10초)
                sem_acquired = sem.acquire(timeout=10)
                if not sem_acquired:
                    logging.warning(f"GPU 세마포어 획득 실패: 인스턴스 ID={self._instance_id}, 배치 크기={len(texts)}, 제한 시간 초과")
                else:
                    logging.info(f"세마포어 획득 성공: 인스턴스 ID={self._instance_id}, 배치 크기={len(texts)}")
            
                # GPU 메모리 상태 로깅
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()/1024**2
                    reserved = torch.cuda.memory_reserved()/1024**2
                    logging.info(f"GPU 메모리 상태: 할당={allocated:.2f}MB, 예약={reserved:.2f}MB")
            
                # 활성 작업 수 증가
                with self._task_lock:
                    self._active_gpu_tasks += 1
                    logging.info(f"활성 작업 증가: 인스턴스 ID={self._instance_id}, 작업 수={self._active_gpu_tasks}")
            
                # 배치 임베딩 계산 시작
                compute_start = time.time()
                
                # BGE 모델 확인
                if not hasattr(self, 'bge_emb') or self.bge_emb is None:
                    logging.warning(f"모델 로드 필요: 인스턴스 ID={self._instance_id}")
                    self.set_emb_model('bge')
                
                # BGE 모델로 배치 임베딩 계산
                with torch.no_grad():
                    logging.info(f"배치 임베딩 계산 시작: 인스턴스 ID={self._instance_id}, 배치 크기={len(texts)}")
                    
                    # 배치 단위로 직접 encode 호출 (FlagEmbedding의 배치 처리 활용)
                    result = self.bge_emb.encode(texts, max_length=self.emb_config.get('max_length', 1024))
                    
                    logging.info(f"배치 임베딩 계산 완료: 인스턴스 ID={self._instance_id}")
                
                # 결과 추출 및 변환
                embedding_vectors = []
                
                # 벡터 추출 및 처리
                if isinstance(result, dict) and 'dense_vecs' in result:
                    # dense_vecs가 2D 배열인 경우 (배치 처리)
                    dense_vecs = result['dense_vecs']
                    if isinstance(dense_vecs, np.ndarray) and len(dense_vecs.shape) == 2:
                        # 각 벡터를 리스트로 변환
                        for vec in dense_vecs:
                            vec_list = vec.tolist()
                            embedding_vectors.append(vec_list)
                    else:
                        # 단일 벡터만 반환된 경우 (예상치 못한 상황)
                        logging.warning(f"예상치 못한 결과 형식: {type(dense_vecs)}, 형상: {dense_vecs.shape if hasattr(dense_vecs, 'shape') else 'unknown'}")
                        # 안전하게 처리하기 위해 리스트 반환
                        embedding_vectors = [dense_vecs.tolist()] * len(texts)
                else:
                    # 다른 형식의 결과 처리
                    logging.warning(f"예상치 못한 결과 형식: {type(result)}")
                    # 안전 처리: 모든 임베딩을 0 벡터로 설정
                    embedding_vectors = [[0.0] * 1024 for _ in range(len(texts))]
                
                # 벡터 길이 확인 및 보정
                expected_dim = 1024
                for i, vec in enumerate(embedding_vectors):
                    if len(vec) != expected_dim:
                        logging.info(f"벡터 {i} 길이 부족하여 패딩: {len(vec)} → {expected_dim}")
                        # 부족한 차원은 0으로 채움
                        if len(vec) < expected_dim:
                            vec.extend([0.0] * (expected_dim - len(vec)))
                        # 초과 차원은 잘라냄
                        else:
                            embedding_vectors[i] = vec[:expected_dim]
                
                compute_end = time.time()
                compute_time = compute_end - compute_start
                
                # 배치 임베딩 완료 시간 계산
                end_time = time.time()
                total_time = end_time - start_time
                
                # 장시간 소요된 경우 경고 로그
                if total_time > 5.0:
                    logging.warning(f"⚠️ 비정상적으로 긴 배치 임베딩 처리 시간: {total_time:.2f}초, 인스턴스 ID={self._instance_id}, 배치 크기={len(texts)}")
                
                # 일반 로그 (디버깅용)
                logging.info(f"배치 임베딩 완료: 인스턴스 ID={self._instance_id}, 총 {total_time:.4f}초, 계산={compute_time:.4f}초, 항목 수={len(embedding_vectors)}")
                
                # GPU 메모리 상태 다시 로깅
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()/1024**2
                    reserved = torch.cuda.memory_reserved()/1024**2
                    logging.info(f"임베딩 후 GPU 메모리: 할당={allocated:.2f}MB, 예약={reserved:.2f}MB")
                
                return embedding_vectors
                
            except Exception as e:
                logging.error(f"배치 임베딩 생성 오류: 인스턴스 ID={self._instance_id}, 오류={str(e)}")
                import traceback
                logging.error(f"스택 트레이스: {traceback.format_exc()}")
                # 오류 발생 시 0 벡터 반환
                return [[0.0] * 1024 for _ in range(len(texts))]
                
            finally:
                # 활성 작업 수 감소
                with self._task_lock:
                    self._active_gpu_tasks = max(0, self._active_gpu_tasks - 1)
                    logging.info(f"활성 작업 감소: 인스턴스 ID={self._instance_id}, 작업 수={self._active_gpu_tasks}")
                
                # 세마포어 반환 (획득한 경우에만)
                if sem_acquired:
                    try:
                        sem.release()
                        logging.info(f"세마포어 반환: 인스턴스 ID={self._instance_id}")
                    except Exception as release_error:
                        logging.error(f"세마포어 반환 오류: 인스턴스 ID={self._instance_id}, 오류={str(release_error)}")
                
                # GPU 메모리 정리 (다른 작업이 없을 때만)
                with self._task_lock:
                    if self._active_gpu_tasks == 0 and torch.cuda.is_available():
                        logging.info(f"GPU 메모리 정리 시작: 인스턴스 ID={self._instance_id}")
                        torch.cuda.empty_cache()
                        logging.info(f"GPU 메모리 정리 완료: 인스턴스 ID={self._instance_id}")
        
        except Exception as outer_error:
            logging.error(f"배치 임베딩 외부 오류: 인스턴스 ID={self._instance_id}, 오류={str(outer_error)}")
            import traceback
            logging.error(f"외부 오류 스택 트레이스: {traceback.format_exc()}")
            # 오류 발생 시 0 벡터 반환
            return [[0.0] * 1024 for _ in range(len(texts))]


class LLMOpenAI(Model):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI()

    def set_generation_config(self, max_tokens=500, temperature=0.9):
        self.gen_config = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    def get_response(self, query, role="너는 금융권에서 일하고 있는 조수로, 회사 규정에 대해 알려주는 역할을 맡고 있어. 사용자 질문에 대해 간단 명료하게 답을 해줘.", model='gpt-4'):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": query},
                ],
                max_tokens=self.gen_config['max_tokens'],
                temperature=self.gen_config['temperature'],
            )    
        except Exception as e:
            return f"Error: {str(e)}"
        return response.choices[0].message.content

    def set_prompt_template(self, query, context):
        self.rag_prompt_template = """
        다음 질문에 대해 주어진 정보를 참고해서 답을 해줘.
        주어진 정보: {context}
        --------------------------------
        질문: {query} 
        """
        return self.rag_prompt_template.format(query=query, context=context)

