"""
Reranker models for improving search results
"""
"""
혹여나 다른 구현 옵션이 필요할까봐 구현
"""

import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Union, Optional, Tuple, Any
import logging


class BaseReranker:
    """Base class for rerankers"""
    
    def __init__(self, model_name_or_path: str, device: str = None):
        """
        Initialize the reranker
        
        Args:
            model_name_or_path: Model name or path to model
            device: Device to use for inference (cuda, cpu, etc.)
        """
        self.model_name = model_name_or_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Initializing reranker with model: {model_name_or_path} on device: {self.device}")
        self.tokenizer = None
        self.model = None
    
    def load(self):
        """Load model and tokenizer"""
        raise NotImplementedError("Subclasses must implement load()")
    
    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """Rerank passages based on query"""
        raise NotImplementedError("Subclasses must implement rerank()")


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker"""
    
    def __init__(self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name_or_path: Model name or path to model
            device: Device to use for inference
        """
        super().__init__(model_name_or_path, device)
        self.max_length = 512
        self.batch_size = 64  # 배치 크기 증가
        self._model_loaded = False  # 모델 로드 상태 추적
    
    def load(self):
        """Load model and tokenizer"""
        if self._model_loaded:
            return self  # 이미 로드된 경우 재사용
            
        print(f"[INFO] Loading cross-encoder model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 모델 로딩 최적화
        try:
            # 메모리 최적화를 위한 설정
            config_kwargs = {
                "torchscript": True,  # 모델 최적화
                "return_dict": False  # 메모리 사용량 감소
            }
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **config_kwargs
            )
            
            # GPU 메모리 최적화
            if self.device == 'cuda':
                # 반정밀도(FP16) 사용
                self.model = self.model.half()
                
            self.model.to(self.device)
            
            # 추론 모드 설정
            self.model.eval()
            
            # 가능하면 모델을 TorchScript로 변환
            try:
                sample_inputs = self.tokenizer(
                    [("query", "passage")], 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=self.max_length
                )
                sample_inputs = {k: v.to(self.device) for k, v in sample_inputs.items()}
                self.model = torch.jit.trace(self.model, (sample_inputs['input_ids'], sample_inputs['attention_mask']), strict=False)
                print(f"[INFO] Model converted to TorchScript for better performance")
            except Exception as e:
                print(f"[WARNING] Failed to convert model to TorchScript: {e}")
                
            self._model_loaded = True
            print(f"[INFO] Model loaded successfully with optimizations")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model with optimizations: {e}")
            # 기본 방식으로 로드 시도
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            
        return self
    
    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank passages based on query
        
        Args:
            query: Search query
            passages: List of passage dictionaries, each with at least a 'text' field
            top_k: Number of top results to return, default is all
        
        Returns:
            Reranked list of passages with added 'rerank_score' field
        """
        if not self.model or not self.tokenizer:
            self.load()
            
        if not passages:
            print("[WARNING] No passages to rerank")
            return []
            
        print(f"[INFO] Reranking {len(passages)} passages for query: {query}")
        
        # Prepare inputs
        passage_texts = [p.get('text', '') for p in passages]
        pairs = [(query, text) for text in passage_texts]
        
        # 배치 처리로 메모리 효율성 향상
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_pairs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs[0].squeeze(-1).cpu().tolist() if isinstance(outputs, tuple) else outputs.logits.squeeze(-1).cpu().tolist()
                scores.extend(batch_scores)
            
            # GPU 메모리 정리
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
        # Add scores to passages
        for i, (passage, score) in enumerate(zip(passages, scores)):
            passage['rerank_score'] = float(score)
            passage['rerank_position'] = i
        
        # Sort by score
        reranked_passages = sorted(passages, key=lambda x: x['rerank_score'], reverse=True)
        
        # Limit to top_k if specified
        if top_k and isinstance(top_k, int) and top_k > 0:
            reranked_passages = reranked_passages[:top_k]
            
        return reranked_passages


class FlashRankReranker(BaseReranker):
    """FlashRank-based reranker for fast and efficient reranking"""
    
    def __init__(self, model_name_or_path: str = "intfloat/e5-base-v2", model_dir: str = None, device: str = None):
        """
        Initialize FlashRank reranker
        
        Args:
            model_name_or_path: Model name or path
            model_dir: Directory to load model from (if downloaded locally)
            device: Device to use for inference
        """
        self.model_dir = model_dir
        if model_dir:
            model_path = os.path.join(model_dir, os.path.basename(model_name_or_path))
            if os.path.exists(model_path):
                model_name_or_path = model_path
                
        super().__init__(model_name_or_path, device)
        self.max_length = 512
        self.batch_size = 256 if self.device == 'cuda' else 64  # GPU/CPU에 따라 배치 크기 조정
        self._model_loaded = False  # 모델 로드 상태 추적
        
    def load(self):
        """Load FlashRank model"""
        if self._model_loaded and hasattr(self, 'flashrank_model'):
            return self  # 이미 로드된 경우 재사용
            
        try:
            print(f"[INFO] Loading FlashRank model: {self.model_name}")
            
            # 최신 flashrank 패키지 임포트
            import flashrank
            
            # 모델 초기화 - 최신 API에 맞게 업데이트
            # 최신 FlashRank는 클래스 이름과 메서드가 변경되었을 수 있음
            try:
                # 방법 1: 만약 Reranker가 클래스 이름인 경우
                from flashrank import Reranker
                
                # 최적화 옵션 설정
                model_kwargs = {
                    "use_fp16": True if self.device == 'cuda' else False,  # GPU에서는 FP16 사용
                    "cache_dir": self.model_dir if self.model_dir else None,
                    "device": self.device,
                    "max_length": self.max_length,
                    "batch_size": self.batch_size
                }
                
                self.flashrank_model = Reranker(
                    model_name_or_path=self.model_name,
                    **model_kwargs
                )
                print("[INFO] FlashRank Reranker class loaded successfully")
            except (ImportError, AttributeError):
                try:
                    # 방법 2: CrossEncoder가 클래스 이름인 경우
                    from flashrank import CrossEncoder
                    
                    # 최적화 옵션 설정
                    model_kwargs = {
                        "use_fp16": True if self.device == 'cuda' else False,
                        "cache_dir": self.model_dir if self.model_dir else None,
                        "device": self.device,
                        "max_length": self.max_length,
                        "batch_size": self.batch_size
                    }
                    
                    self.flashrank_model = CrossEncoder(
                        model_name_or_path=self.model_name,
                        **model_kwargs
                    )
                    print("[INFO] FlashRank CrossEncoder class loaded successfully")
                except (ImportError, AttributeError):
                    # 방법 3: 직접 모듈 함수 사용
                    self.flashrank_model = flashrank
                    print("[INFO] FlashRank module loaded successfully")
            
            print(f"[INFO] FlashRank version: {flashrank.__version__ if hasattr(flashrank, '__version__') else 'unknown'}")
            self._model_loaded = True
            
        except ImportError as e:
            print(f"[ERROR] FlashRank not installed. Error: {str(e)}")
            print("[ERROR] Try installing from source: pip install git+https://github.com/AnswerDotAI/flashrank.git")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load FlashRank model: {str(e)}")
            raise
            
        return self
    
    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank passages using FlashRank
        
        Args:
            query: Search query
            passages: List of passage dictionaries, each with at least a 'text' field
            top_k: Number of top results to return, default is all
        
        Returns:
            Reranked list of passages with added 'rerank_score' field
        """
        if not hasattr(self, 'flashrank_model') or not self._model_loaded:
            self.load()
            
        if not passages:
            print("[WARNING] No passages to rerank")
            return []
            
        print(f"[INFO] FlashRank reranking {len(passages)} passages for query: {query}")
        
        try:
            # 패시지 텍스트 추출
            texts = [p.get('text', '') for p in passages]
            
            # 성능 최적화: 배치 처리
            scores = []
            
            # 대량의 패시지가 있을 경우 배치 처리
            if len(texts) > self.batch_size:
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i + self.batch_size]
                    batch_scores = self._rerank_batch(query, batch_texts)
                    scores.extend(batch_scores)
                    
                    # GPU 메모리 정리
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            else:
                # 소량 데이터는 한 번에 처리
                scores = self._rerank_batch(query, texts)
            
            # 점수가 리스트가 아니면 변환
            if not isinstance(scores, list):
                print(f"[WARNING] Unexpected scores format: {type(scores)}")
                if hasattr(scores, 'tolist'):  # numpy나 torch 텐서인 경우
                    scores = scores.tolist()
                else:
                    scores = [float(scores)] * len(passages)
            
            # 점수를 패시지에 추가
            for i, (passage, score) in enumerate(zip(passages, scores)):
                passage['rerank_score'] = float(score)
                passage['rerank_position'] = i
            
            # 점수로 정렬
            reranked_passages = sorted(passages, key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            # top_k 적용
            if top_k and isinstance(top_k, int) and top_k > 0:
                reranked_passages = reranked_passages[:top_k]
                
            return reranked_passages
            
        except Exception as e:
            print(f"[ERROR] FlashRank reranking failed: {str(e)}")
            print("[INFO] Falling back to default sorting")
            # 오류 발생 시 원래 순서 유지
            for i, passage in enumerate(passages):
                passage['rerank_score'] = passage.get('score', 0.0)
                passage['rerank_position'] = i
            return passages
            
    def _rerank_batch(self, query: str, texts: List[str]) -> List[float]:
        """배치 단위로 리랭킹 수행"""
        try:
            # 다양한 FlashRank API 호출 방식 시도
            if hasattr(self.flashrank_model, 'compute_score'):
                scores = self.flashrank_model.compute_score(query=query, passages=texts)
                return scores
            # 방법 2: rerank 메서드가 있는 경우
            elif hasattr(self.flashrank_model, 'rerank'):
                results = self.flashrank_model.rerank(query=query, passages=texts)
                # results 형식에 따라 점수 추출
                if isinstance(results, dict) and 'scores' in results:
                    return results['scores']
                elif isinstance(results, list):
                    return [r.get('score', 0) for r in results]
                else:
                    return results
            # 방법 3: score_passages 함수가 있는 경우
            elif hasattr(self.flashrank_model, 'score_passages'):
                return self.flashrank_model.score_passages(query=query, passages=texts)
            # 방법 4: 모듈 함수 직접 호출
            else:
                import flashrank
                # 모듈에서 적절한 함수 찾기
                if hasattr(flashrank, 'rerank'):
                    results = flashrank.rerank(query=query, passages=texts, model=self.model_name)
                    return [r.get('score', 0) for r in results]
                elif hasattr(flashrank, 'score'):
                    return flashrank.score(query=query, passages=texts, model=self.model_name)
                else:
                    raise AttributeError("No compatible scoring function found in flashrank")
        except Exception as e:
            print(f"[ERROR] FlashRank API call failed in batch: {str(e)}")
            raise


class CohereCrossEncoder(BaseReranker):
    """Cohere Reranker API based reranker"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Cohere reranker
        
        Args:
            api_key: Cohere API key, falls back to COHERE_API_KEY env var
        """
        super().__init__("cohere-api", None)
        self.api_key = api_key or os.environ.get('COHERE_API_KEY')
        if not self.api_key:
            raise ValueError("Cohere API key is required. Set COHERE_API_KEY env var or pass api_key.")
        
        # Import cohere in init to fail fast if not installed
        try:
            import cohere
            self.cohere = cohere
        except ImportError:
            raise ImportError("Please install cohere package: pip install cohere")
    
    def load(self):
        """Initialize Cohere client"""
        print(f"[INFO] Initializing Cohere client")
        self.client = self.cohere.Client(self.api_key)
        return self
    
    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank passages using Cohere Rerank API
        
        Args:
            query: Search query
            passages: List of passage dictionaries, each with at least a 'text' field
            top_k: Number of top results to return, default is all
        
        Returns:
            Reranked list of passages with added 'rerank_score' field
        """
        if not hasattr(self, 'client'):
            self.load()
            
        if not passages:
            print("[WARNING] No passages to rerank")
            return []
            
        print(f"[INFO] Reranking {len(passages)} passages with Cohere API for query: {query}")
        
        # Prepare documents
        docs = [p.get('text', '') for p in passages]
        
        # Use Cohere rerank API
        try:
            results = self.client.rerank(
                query=query,
                documents=docs,
                top_n=top_k or len(docs),
                model="rerank-english-v2.0"
            )
            
            # Add scores to passages and reorder
            reranked = []
            for idx, result in enumerate(results.results):
                original_idx = result.index
                passages[original_idx]['rerank_score'] = result.relevance_score
                passages[original_idx]['rerank_position'] = idx
                reranked.append(passages[original_idx])
                
            return reranked
            
        except Exception as e:
            print(f"[ERROR] Cohere rerank API error: {str(e)}")
            # Return original passages on error
            for i, passage in enumerate(passages):
                passage['rerank_score'] = 0.0
                passage['rerank_position'] = i
            return passages 