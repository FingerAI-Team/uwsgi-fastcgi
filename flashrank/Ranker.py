import json
from pathlib import Path
from tokenizers import AddedToken, Tokenizer
import onnxruntime as ort
import numpy as np
import os
import zipfile
import requests
from tqdm import tqdm
from flashrank.Config import default_model, default_cache_dir, model_url, model_file_map, listwise_rankers, huggingface_rankers, huggingface_model_map
import collections
from typing import Optional, List, Dict, Any
import logging
import time
import traceback

class RerankRequest:
    """ Represents a reranking request with a query and a list of passages. 
    
    Attributes:
        query (Optional[str]): The query for which the passages need to be reranked.
        passages (List[Dict[str, Any]]): The list of passages to be reranked.
    """

    def __init__(self, query: Optional[str] = None, passages: Optional[List[Dict[str, Any]]] = None):
        self.query: Optional[str] = query
        self.passages: List[Dict[str, Any]] = passages if passages is not None else []

class Ranker:
    """ A ranker class for reranking passages based on a provided query using a pre-trained model.

    Attributes:
        cache_dir (Path): Path to the cache directory where models are stored.
        model_dir (Path): Path to the directory of the specific model being used.
        session (ort.InferenceSession): The ONNX runtime session for making inferences.
        tokenizer (Tokenizer): The tokenizer for text processing.
    """

    def __init__(self, model_name: str = default_model, cache_dir: str = default_cache_dir, max_length: int = 512, log_level: str = "INFO"):
        """ Initializes the Ranker class with specified model and cache settings.

        Args:
            model_name (str): The name of the model to be used.
            cache_dir (str): The directory where models are cached.
            max_length (int): The maximum length of the tokens.
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        
        # 로그 디렉토리 설정
        log_dir = "/var/log/reranker"
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                print(f"로그 디렉토리 생성 실패: {str(e)}")
            
        # 로깅 설정
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
        self.logger = logging.getLogger(__name__)
        
        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러 설정 - ranker.log
        try:
            file_handler = logging.FileHandler(os.path.join(log_dir, 'ranker.log'))
            file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"ranker.log 파일 핸들러 설정 실패: {str(e)}")
        
        # 상세 로그용 파일 핸들러 - reranker_detail.log
        try:
            detail_handler = logging.FileHandler(os.path.join(log_dir, 'reranker_detail.log'))
            detail_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            detail_handler.setFormatter(formatter)
            self.logger.addHandler(detail_handler)
        except Exception as e:
            print(f"reranker_detail.log 파일 핸들러 설정 실패: {str(e)}")

        # 초기화 시작 로그
        self.log_with_detail(f"[FLASHRANK-INIT] FlashRank 초기화 시작: 모델={model_name}, 캐시 디렉토리={cache_dir}")
        
        self.cache_dir: Path = Path(cache_dir)
        self.model_dir: Path = self.cache_dir / model_name
        self.max_length = max_length
        self.name = model_name  # 모델 이름 저장
        
        # GPU 사용 설정
        try:
            import torch
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0 사용
                self.device = "cuda:0"
                self.log_with_detail(f"[FLASHRANK-INIT] GPU 사용: {self.device}")
                self.log_with_detail(f"[FLASHRANK-INIT] GPU 메모리: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                
                # GPU 이름 가져오기 시도 - 오류 발생 가능성이 있는 부분을 try-except로 감싸기
                try:
                    self.log_with_detail(f"[FLASHRANK-INIT] GPU 이름: {torch.cuda.get_device_name(0)}")
                except Exception as e:
                    self.log_with_detail(f"[FLASHRANK-INIT] GPU 이름 확인 실패: {str(e)}")
                    self.log_with_detail("[FLASHRANK-INIT] GPU 이름 확인 실패했지만 계속 진행합니다.")
            else:
                self.device = "cpu"
                self.log_with_detail("[FLASHRANK-INIT] GPU 사용 불가, CPU 사용")
        except ImportError:
            self.device = "cpu"
            self.log_with_detail("[FLASHRANK-INIT] PyTorch 없음, CPU 사용")
        except Exception as e:
            self.log_with_detail(f"[FLASHRANK-INIT] GPU 초기화 중 오류 발생: {str(e)}")
            self.device = "cpu"
            self.log_with_detail("[FLASHRANK-INIT] GPU 초기화 실패로 CPU 사용")
        
        self.llm_model = None
        self.hf_model = None
        self.hf_tokenizer = None
        
        try:
            # HuggingFace 모델 사용 시
            if model_name in huggingface_rankers:
                self.init_huggingface_model(model_name)
            # 기존 모델 사용 시
            else:
                self.init_standard_model(model_name)
                
            self.log_with_detail(f"[FLASHRANK-INIT] FlashRank 초기화 성공: 모델={model_name}")
        except Exception as e:
            error_msg = f"[FLASHRANK-INIT] FlashRank 초기화 실패: {str(e)}"
            self.log_with_detail(error_msg, level="ERROR", include_traceback=True)
            raise RuntimeError(error_msg) from e

    def log_with_detail(self, message: str, level: str = "INFO", include_traceback: bool = False):
        """로그 메시지를 일반 로그와 상세 로그 파일에 모두 기록합니다."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
        
        # 상세 로그 파일에 직접 기록
        try:
            with open('/var/log/reranker/reranker_detail.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
                if include_traceback:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {traceback.format_exc()}\n")
        except Exception as e:
            self.logger.warning(f"상세 로그 파일 기록 실패: {str(e)}")

    def init_huggingface_model(self, model_name: str):
        """HuggingFace 모델을 초기화합니다."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # 모델 이름 가져오기
            hf_model_name = huggingface_model_map[model_name]
            
            self.log_with_detail(f"[FLASHRANK-INIT] HuggingFace 모델 로딩 시작: {hf_model_name}")
            
            # 캐시 디렉토리 설정
            if not self.model_dir.exists():
                self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # 모델과 토크나이저 로드
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                hf_model_name,
                cache_dir=str(self.model_dir),
                local_files_only=False
            )
            
            # GPU 사용 가능 시 half precision 사용
            dtype = torch.float16 if self.device == "cuda:0" else torch.float32
            self.log_with_detail(f"[FLASHRANK-INIT] 모델 데이터 타입: {dtype}")
            
            self.hf_model = AutoModelForSequenceClassification.from_pretrained(
                hf_model_name,
                cache_dir=str(self.model_dir),
                local_files_only=False,
                trust_remote_code=True,
                torch_dtype=dtype
            )
            
            # GPU로 모델 이동
            if self.device == "cuda:0":
                self.log_with_detail("[FLASHRANK-INIT] 모델을 GPU로 이동...")
                self.hf_model.to(self.device)
                # 모델이 실제로 GPU에 있는지 확인
                model_device = next(self.hf_model.parameters()).device
                self.log_with_detail(f"[FLASHRANK-INIT] 모델 현재 장치: {model_device}")
                
                # GPU 메모리 사용량 확인
                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB 단위
                self.log_with_detail(f"[FLASHRANK-INIT] 모델 로딩 후 GPU 메모리: {mem_allocated:.2f}GB 할당, {mem_reserved:.2f}GB 예약")
            
            self.hf_model.eval()
            self.log_with_detail("[FLASHRANK-INIT] HuggingFace 모델 로딩 성공 (평가 모드 설정)")
        except ImportError:
            error_msg = "HuggingFace 모델 사용을 위해 torch와 transformers를 설치하세요: pip install torch transformers"
            self.log_with_detail(f"[FLASHRANK-INIT] {error_msg}", level="ERROR")
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"HuggingFace 모델 {hf_model_name} 로딩 실패: {str(e)}"
            self.log_with_detail(f"[FLASHRANK-INIT] {error_msg}", level="ERROR", include_traceback=True)
            raise RuntimeError(error_msg) from e

    def init_standard_model(self, model_name: str):
        """표준 ONNX 또는 LLM 모델을 초기화합니다."""
        self._prepare_model_dir(model_name)
        model_file = model_file_map[model_name]
        
        self.log_with_detail(f"[FLASHRANK-INIT] 표준 모델 로딩 시작: {model_name}, 파일: {model_file}")
        
        if model_name in listwise_rankers:
            try:
                from llama_cpp import Llama
                # GPU 지원을 위한 옵션 설정
                gpu_layers = -1 if self.device == "cuda" else 0
                self.llm_model = Llama(
                    model_path=str(self.model_dir / model_file),
                    n_ctx=self.max_length,
                    n_threads=8,
                    n_gpu_layers=gpu_layers  # GPU 사용 시 모든 레이어를 GPU로
                )
                self.log_with_detail(f"[FLASHRANK-INIT] LLM 모델 로딩 성공: GPU 레이어={gpu_layers}")
            except ImportError:
                error_msg = "LLM 모델 사용을 위해 GPU 지원과 함께 llama-cpp-python을 설치하세요: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python"
                self.log_with_detail(f"[FLASHRANK-INIT] {error_msg}", level="ERROR")
                raise ImportError(error_msg)
            except Exception as e:
                error_msg = f"LLM 모델 로딩 실패: {str(e)}"
                self.log_with_detail(f"[FLASHRANK-INIT] {error_msg}", level="ERROR", include_traceback=True)
                raise RuntimeError(error_msg) from e
        else:
            # ONNX Runtime providers 설정
            providers = []
            if self.device == "cuda":
                providers.extend([
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider'
                ])
            else:
                providers.append('CPUExecutionProvider')
                
            self.log_with_detail(f"[FLASHRANK-INIT] ONNX Runtime providers: {providers}")
            
            try:
                self.session = ort.InferenceSession(
                    str(self.model_dir / model_file),
                    providers=providers
                )
                self.tokenizer: Tokenizer = self._get_tokenizer(self.max_length)
                self.log_with_detail(f"[FLASHRANK-INIT] ONNX 모델 및 토크나이저 로딩 성공")
            except Exception as e:
                error_msg = f"ONNX 모델 로딩 실패: {str(e)}"
                self.log_with_detail(f"[FLASHRANK-INIT] {error_msg}", level="ERROR", include_traceback=True)
                raise RuntimeError(error_msg) from e

    def _prepare_model_dir(self, model_name: str):
        """ Ensures the model directory is prepared by downloading and extracting the model if not present.

        Args:
            model_name (str): The name of the model to be prepared.
        """
        # HuggingFace 모델인 경우 다운로드 로직을 건너뜁니다
        if model_name in huggingface_rankers:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            if not self.model_dir.exists():
                self.model_dir.mkdir(parents=True, exist_ok=True)
            return

        # 모델 디렉토리가 없으면 생성
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)

        # 모델 파일 확인
        model_file = model_file_map[model_name]
        model_path = self.model_dir / model_file
        if not model_path.exists():
            self.log_with_detail(f"[FLASHRANK-INIT] 모델 파일이 없습니다. 다운로드를 시작합니다: {model_file}")
            self._download_model_files(model_name)
        else:
            self.log_with_detail(f"[FLASHRANK-INIT] 모델 파일이 이미 존재합니다: {model_path}")

    def _download_model_files(self, model_name: str):
        """Downloads model files from the specified URL.

        Args:
            model_name (str): The name of the model to download.
        """
        try:
            model_zip_path = self.model_dir / f"{model_name}.zip"
            self.log_with_detail(f"[FLASHRANK-INIT] 모델 파일 다운로드 시작: {model_url[model_name]} -> {model_zip_path}")
            
            # 모델 파일 다운로드
            response = requests.get(model_url[model_name], stream=True)
            response.raise_for_status()  # HTTP 오류 확인
            
            # 파일 크기 확인
            total_size = int(response.headers.get('content-length', 0))
            self.log_with_detail(f"[FLASHRANK-INIT] 다운로드 크기: {total_size/1024/1024:.2f} MB")
            
            # 파일 저장
            with open(model_zip_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size//8192, desc=f"Downloading {model_name}"):
                    if chunk:
                        f.write(chunk)
            
            # 압축 해제
            self.log_with_detail(f"[FLASHRANK-INIT] 모델 파일 압축 해제 시작: {model_zip_path}")
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.model_dir)
            
            # 다운로드한 zip 파일 삭제
            model_zip_path.unlink()
            self.log_with_detail(f"[FLASHRANK-INIT] 모델 파일 다운로드 및 압축 해제 완료")
        except Exception as e:
            error_msg = f"모델 파일 다운로드 실패: {str(e)}"
            self.log_with_detail(f"[FLASHRANK-INIT] {error_msg}", level="ERROR", include_traceback=True)
            raise RuntimeError(error_msg) from e

    def _get_tokenizer(self, max_length: int = 512) -> Tokenizer:
        """Gets the tokenizer for the model.

        Args:
            max_length (int): The maximum length of the tokens.

        Returns:
            Tokenizer: The tokenizer for the model.
        """
        try:
            self.log_with_detail(f"[FLASHRANK-INIT] 토크나이저 초기화 시작: 최대 길이={max_length}")
            
            # 토크나이저 파일 경로
            tokenizer_path = self.model_dir / "tokenizer.json"
            if not tokenizer_path.exists():
                error_msg = f"토크나이저 파일을 찾을 수 없습니다: {tokenizer_path}"
                self.log_with_detail(f"[FLASHRANK-INIT] {error_msg}", level="ERROR")
                raise FileNotFoundError(error_msg)
            
            # 토크나이저 로드
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            
            # 특수 토큰 추가
            tokenizer.add_special_tokens([
                AddedToken("<s>", normalized=False),
                AddedToken("</s>", normalized=False),
                AddedToken("<pad>", normalized=False)
            ])
            
            # 패딩 토큰 설정
            tokenizer.enable_padding(pad_id=0, pad_token="<pad>", length=max_length)
            
            # 트렁케이션 설정
            tokenizer.enable_truncation(max_length=max_length)
            
            self.log_with_detail(f"[FLASHRANK-INIT] 토크나이저 초기화 완료")
            return tokenizer
        except Exception as e:
            error_msg = f"토크나이저 초기화 실패: {str(e)}"
            self.log_with_detail(f"[FLASHRANK-INIT] {error_msg}", level="ERROR", include_traceback=True)
            raise RuntimeError(error_msg) from e

    def _get_prefix_prompt(self, query, num):
        return [
            {
                "role": "system",
                "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
            },
            {
                "role": "user",
                "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}.",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]

    def _get_postfix_prompt(self, query, num):
        example_ordering = "[2] > [1]"
        return {
            "role": "user",
            "content": f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., {example_ordering}, Only respond with the ranking results, do not say any word or explain.",
        }

    def rerank(self, request: RerankRequest) -> List[Dict[str, Any]]:
        """ Reranks a list of passages based on a query using a pre-trained model.

        Args:
            request (RerankRequest): The request containing the query and passages to rerank.

        Returns:
            List[Dict[str, Any]]: The reranked list of passages with added scores.
        """
        query = request.query
        passages = request.passages

        # HuggingFace 모델 사용 (한국어 reranker)
        if self.hf_model is not None:
            self.logger.debug("Running HuggingFace reranking...")
            import torch
            
            # 모델이 실제로 GPU에 있는지 다시 확인
            if self.device == "cuda:0":
                model_device = next(self.hf_model.parameters()).device
                if str(model_device) == "cpu":
                    self.logger.warning("Model is on CPU! Moving to GPU...")
                    self.hf_model.to(self.device)
                    model_device = next(self.hf_model.parameters()).device
                    self.logger.info(f"Model is now on device: {model_device}")
                
                # GPU 메모리 상태 로깅
                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB 단위
                self.logger.info(f"GPU Memory before inference: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            
            # 배치 크기 설정 (GPU/CPU에 따라 다르게)
            batch_size = 32 if self.device == "cuda:0" else 16
            self.logger.info(f"Using batch size: {batch_size} on device: {self.device}")
            
            # 전체 패시지 수
            total_passages = len(passages)
            self.logger.info(f"Total passages to process: {total_passages}")
            
            # 배치 단위로 처리
            all_scores = []
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    for i in range(0, total_passages, batch_size):
                        batch_passages = passages[i:i + batch_size]
                        batch_pairs = [[query, passage["text"]] for passage in batch_passages]
                        self.logger.info(f"Processing batch {i//batch_size + 1}/{(total_passages + batch_size - 1)//batch_size} with {len(batch_pairs)} pairs")
                        
                        # 배치 시작 시간 기록
                        batch_start_time = time.time() if 'time' in globals() else None
                        
                        with torch.no_grad():
                            # 토크나이징
                            inputs = self.hf_tokenizer(
                                batch_pairs, 
                                padding=True, 
                                truncation=True, 
                                return_tensors='pt', 
                                max_length=self.max_length
                            )
                            
                            # GPU로 입력 이동
                            if self.device == "cuda:0":
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                                
                                # GPU 메모리 상태 로깅
                                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                                self.logger.info(f"GPU Memory after input loading: {mem_allocated:.2f}GB")
                            
                            # 추론 실행
                            outputs = self.hf_model(**inputs, return_dict=True)
                            batch_scores = outputs.logits.view(-1, ).float()
                            
                            # GPU 메모리 상태 로깅
                            if self.device == "cuda:0":
                                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB 단위
                                self.logger.info(f"GPU Memory after inference: {mem_allocated:.2f}GB")
                                # CPU로 결과 이동
                                batch_scores = batch_scores.cpu()
                            
                            # numpy로 변환
                            batch_scores = batch_scores.numpy()
                            # sigmoid 정규화
                            batch_scores = 1 / (1 + np.exp(-batch_scores))
                            all_scores.extend(batch_scores)
                            
                            # 배치 처리 시간 계산
                            if batch_start_time:
                                batch_time = time.time() - batch_start_time
                                passages_per_sec = len(batch_passages) / batch_time
                                self.logger.info(f"Batch {i//batch_size + 1} completed in {batch_time:.3f}s ({passages_per_sec:.1f} passages/sec)")
                            
                            # 메모리 정리
                            del inputs, outputs, batch_scores
                            
                            # 매 배치마다 GPU 캐시 정리 (성능에 영향 있을 수 있음)
                            if self.device == "cuda:0":
                                torch.cuda.empty_cache()
                    
                    # 모든 배치 처리 후 메모리 정리
                    if self.device == "cuda:0":
                        torch.cuda.empty_cache()
                        
                    break  # 성공적으로 처리 완료
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.device == "cuda:0":
                        retry_count += 1
                        if retry_count < max_retries:
                            self.logger.warning(f"GPU OOM error. Reducing batch size and retrying. Attempt {retry_count}/{max_retries}")
                            torch.cuda.empty_cache()  # 메모리 정리
                            batch_size = batch_size // 2  # 배치 사이즈 절반으로 감소
                            all_scores = []  # 점수 리스트 초기화
                            continue
                    raise  # 다른 에러이거나 최대 재시도 횟수 초과
            
            # 점수 할당
            for score, passage in zip(all_scores, passages):
                passage["score"] = float(score)
            
            # 점수 기준으로 내림차순 정렬
            passages.sort(key=lambda x: x["score"], reverse=True)
            
            self.logger.info(f"Completed reranking {total_passages} passages in {(total_passages + batch_size - 1)//batch_size} batches")
            
            return passages

        # LLM 방식 (Listwise ranking)
        elif self.llm_model is not None:
            self.logger.debug("Running listwise ranking..")
            if self.device == "cuda":
                self.logger.info("LLM model is using GPU layers")
            else:
                self.logger.warning("LLM model is running on CPU")
            num_of_passages = len(passages)
            messages = self._get_prefix_prompt(query, num_of_passages)

            result_map = {}
            for rank, passage in enumerate(passages):
                messages.append(
                    {
                        "role": "user",
                        "content": f"[{rank + 1}] {passage['text']}",
                    }
                )
                messages.append(
                        {
                            "role": "assistant", 
                            "content": f"Received passage [{rank + 1}]."
                        }
                )
                
                result_map[rank + 1] = passage

            messages.append(self._get_postfix_prompt(query, num_of_passages))
            raw_ranks = self.llm_model.create_chat_completion(messages)
            results = []
            for rank in raw_ranks["choices"][0]["message"]["content"].split(" > "):
                results.append(result_map[int(rank.strip("[]"))])
            return results    

        # ONNX 모델 방식 (Pairwise ranking)
        else:
            self.logger.debug("Running pairwise ranking..")
            if hasattr(self.session, '_providers'):
                self.logger.info(f"Active ONNX providers: {self.session._providers}")
            
            # 배치 크기 설정
            batch_size = 256 if 'CUDAExecutionProvider' in (self.session._providers if hasattr(self.session, '_providers') else []) else 32
            self.logger.info(f"Using batch size: {batch_size} for ONNX model")
            
            total_passages = len(passages)
            self.logger.info(f"Total passages to process: {total_passages}")
            
            all_scores = []
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    for i in range(0, total_passages, batch_size):
                        batch_passages = passages[i:i + batch_size]
                        batch_pairs = [[query, passage["text"]] for passage in batch_passages]
                        self.logger.info(f"Processing batch {i//batch_size + 1}/{(total_passages + batch_size - 1)//batch_size} with {len(batch_pairs)} pairs")
                        
                        input_text = self.tokenizer.encode_batch(batch_pairs)
                        input_ids = np.array([e.ids for e in input_text])
                        token_type_ids = np.array([e.type_ids for e in input_text])
                        attention_mask = np.array([e.attention_mask for e in input_text])

                        use_token_type_ids = token_type_ids is not None and not np.all(token_type_ids == 0)

                        onnx_input = {
                            "input_ids": input_ids.astype(np.int64), 
                            "attention_mask": attention_mask.astype(np.int64)
                        }
                        if use_token_type_ids:
                            onnx_input["token_type_ids"] = token_type_ids.astype(np.int64)

                        outputs = self.session.run(None, onnx_input)
                        logits = outputs[0]

                        if logits.shape[1] == 1:
                            batch_scores = 1 / (1 + np.exp(-logits.flatten()))
                        else:
                            exp_logits = np.exp(logits)
                            batch_scores = exp_logits[:, 1] / np.sum(exp_logits, axis=1)
                        
                        all_scores.extend(batch_scores)
                        self.logger.info(f"Completed batch {i//batch_size + 1}")
                        
                        # 메모리 정리
                        del input_text, input_ids, token_type_ids, attention_mask, onnx_input, outputs, logits, batch_scores
                    
                    break  # 성공적으로 처리 완료
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) and 'CUDAExecutionProvider' in (self.session._providers if hasattr(self.session, '_providers') else []):
                        retry_count += 1
                        if retry_count < max_retries:
                            self.logger.warning(f"GPU OOM error. Reducing batch size and retrying. Attempt {retry_count}/{max_retries}")
                            batch_size = batch_size // 2  # 배치 사이즈 절반으로 감소
                            all_scores = []  # 점수 리스트 초기화
                            continue
                    raise  # 다른 에러이거나 최대 재시도 횟수 초과

            # 점수 할당
            for score, passage in zip(all_scores, passages):
                passage["score"] = float(score)

            passages.sort(key=lambda x: x["score"], reverse=True)
            self.logger.info(f"Completed reranking {total_passages} passages in {(total_passages + batch_size - 1)//batch_size} batches")
            
            return passages
