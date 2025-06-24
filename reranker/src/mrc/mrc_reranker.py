import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from .controller import MRCController

logger = logging.getLogger(__name__)

class MRCReranker:
    """MRC 결과를 활용한 재랭킹 기능을 제공하는 클래스"""
    
    _instance = None  # 싱글톤 인스턴스
    
    @classmethod
    def get_instance(cls, config_path=None, model_path=None):
        """싱글톤 패턴으로 인스턴스 반환"""
        if cls._instance is None:
            logger.info(f"Creating new MRCReranker instance")
            cls._instance = cls(config_path, model_path)
        else:
            logger.info("Returning existing MRCReranker instance")
        return cls._instance
    
    def __init__(self, config_path=None, model_path=None):
        """
        MRC 기반 재랭커 초기화
        
        Args:
            config_path: MRC 모델 설정 경로
            model_path: MRC 모델 체크포인트 경로
        """
        self.mrc_controller = MRCController.get_instance(config_path, model_path)
        logger.info("MRCReranker initialized")
        
    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        MRC 기반 재랭킹 수행
        
        Args:
            query: 검색 쿼리
            passages: 재랭킹할 패시지 목록
            top_k: 반환할 상위 결과 수
            
        Returns:
            재랭킹된 패시지 목록
        """
        logger.info(f"MRC 기반 재랭킹 시작: query='{query}', passages={len(passages)}")
        
        # MRC 입력 생성
        samples = []
        for i, passage in enumerate(passages):
            samples.append({
                'question': query,
                'context': passage.get('text', ''),
                'temperature': 1.0,
                'original_index': i  # 원본 인덱스 추적
            })
        
        # MRC 추론 수행
        mrc_results = self.mrc_controller.infer_multi(samples)
        
        # 결과 연결 및 점수 업데이트
        for i, (passage, mrc_result) in enumerate(zip(passages, mrc_results)):
            # MRC 결과 저장
            mrc_score = mrc_result['answerability']
            passage['mrc_answer'] = mrc_result['answer']
            passage['mrc_char_ids'] = mrc_result['char_ids']
            passage['mrc_score'] = mrc_score
            
            # MRC 방식에서는 MRC 점수를 최종 점수로 사용
            passage['score'] = mrc_score
            
            # 디버그 로깅
            logger.debug(f"Passage {i}: mrc_score={mrc_score:.4f}")
        
        # 최종 점수로 정렬
        reranked_passages = sorted(passages, key=lambda x: x.get('score', 0), reverse=True)
        
        # top_k 적용
        if top_k and isinstance(top_k, int) and top_k > 0:
            reranked_passages = reranked_passages[:top_k]
            
        logger.info(f"MRC 기반 재랭킹 완료: {len(reranked_passages)} 결과 반환")
        
        # 메타데이터 중복 제거 - 각 결과 항목에서 metadata 내부의 필드를 상위 레벨로 이동하고 metadata 필드 제거
        for passage in reranked_passages:
            if "metadata" in passage and passage["metadata"] is not None:
                try:
                    # metadata 내부의 모든 필드를 상위 레벨로 복사
                    for key, value in passage["metadata"].items():
                        if key not in passage:  # 이미 존재하는 필드는 덮어쓰지 않음
                            passage[key] = value
                    # metadata 필드 제거
                    del passage["metadata"]
                except AttributeError:
                    # metadata가 None이거나 items() 메소드가 없는 경우
                    logger.warning(f"metadata 필드가 dictionary가 아닙니다: {type(passage['metadata'])}")
                    passage["metadata"] = {}
            
            # MRC 관련 필드가 있는지 확인하고 없으면 기본값 설정
            if "mrc_score" in passage and "mrc_answer" not in passage:
                passage["mrc_answer"] = ""
            if "mrc_score" in passage and "mrc_char_ids" not in passage:
                passage["mrc_char_ids"] = []
        
        return reranked_passages
    
    def process_search_results(self, query: str, search_result: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        검색 결과에 MRC 기반 재랭킹 적용
        (RerankerService.process_search_results와 동일한 인터페이스)
        
        Args:
            query: 검색 쿼리
            search_result: 검색 결과 딕셔너리
            top_k: 반환할 상위 결과 수
            
        Returns:
            재랭킹된 검색 결과
        """
        # search_result가 None인 경우 처리 (오류 수정)
        if search_result is None:
            logger.warning("MRC 재랭킹: search_result가 None입니다.")
            return {
                "query": query,
                "results": [],
                "total": 0,
                "reranked": False,
                "reranker_type": "mrc",
                "error": "검색 결과가 없습니다."
            }
            
        # 재랭킹 수행
        passages = search_result.get("results", [])
        
        # passages가 비어있는 경우 처리 (오류 수정)
        if not passages:
            logger.warning("MRC 재랭킹: 검색 결과가 비어 있습니다.")
            return {
                "query": query,
                "results": [],
                "total": 0,
                "reranked": False,
                "reranker_type": "mrc",
                "error": "검색 결과가 비어 있습니다."
            }
            
        reranked_passages = self.rerank(query, passages, top_k)
        
        # 메타데이터 중복 제거 - 각 결과 항목에서 metadata 내부의 필드를 상위 레벨로 이동하고 metadata 필드 제거
        for passage in reranked_passages:
            if "metadata" in passage and passage["metadata"] is not None:
                try:
                    # metadata 내부의 모든 필드를 상위 레벨로 복사
                    for key, value in passage["metadata"].items():
                        if key not in passage:  # 이미 존재하는 필드는 덮어쓰지 않음
                            passage[key] = value
                    # metadata 필드 제거
                    del passage["metadata"]
                except AttributeError:
                    # metadata가 None이거나 items() 메소드가 없는 경우
                    logger.warning(f"metadata 필드가 dictionary가 아닙니다: {type(passage['metadata'])}")
                    passage["metadata"] = {}
            
            # MRC 관련 필드가 있는지 확인하고 없으면 기본값 설정
            if "mrc_score" in passage and "mrc_answer" not in passage:
                passage["mrc_answer"] = ""
            if "mrc_score" in passage and "mrc_char_ids" not in passage:
                passage["mrc_char_ids"] = []
        
        # 결과 포맷팅
        result = {
            "query": query,
            "results": reranked_passages,
            "total": len(reranked_passages),
            "reranked": True,
            "reranker_type": "mrc"
        }
        
        return result
        
    def hybrid_rerank(self, query: str, passages: List[Dict[str, Any]], 
                      flashrank_scores: List[float], 
                      weight_mrc: float = 0.7,
                      top_k: Optional[int] = None,
                      return_mrc_scores: bool = False) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[float]]]:
        """
        FlashRank 결과와 MRC 결과를 조합한 하이브리드 재랭킹
        
        Args:
            query: 검색 쿼리
            passages: 재랭킹할 패시지 목록
            flashrank_scores: FlashRank에서 계산한 점수 목록
            weight_mrc: MRC 점수 가중치 (0~1 사이)
            top_k: 반환할 상위 결과 수
            return_mrc_scores: MRC 점수 목록도 함께 반환할지 여부
            
        Returns:
            하이브리드 재랭킹된 패시지 목록, 또는 (패시지 목록, MRC 점수 목록) 튜플
        """
        logger.info(f"하이브리드 재랭킹 시작: query='{query}', passages={len(passages)}, weight_mrc={weight_mrc}")
        
        # 입력 데이터 로깅 - original_id 값 확인
        logger.info(f"[DEBUG] 입력 패시지 original_id 샘플: {[p.get('original_id', 'N/A') for p in passages[:5]]}")
        
        # MRC 입력 생성 및 추론
        samples = []
        for passage in passages:
            samples.append({
                'question': query,
                'context': passage.get('text', ''),
                'temperature': 1.0
            })
        
        # 배치 크기 설정 (64로 증가)
        batch_size = 64
        
        # 배치 처리를 위한 준비
        total_samples = len(samples)
        mrc_results = []
        
        # 배치 단위로 처리
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_samples = samples[i:batch_end]
            
            # 배치 처리
            batch_results = self.mrc_controller.infer_multi(batch_samples)
            mrc_results.extend(batch_results)
            
            logger.debug(f"MRC 배치 처리: {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} 완료")
        
        mrc_scores = []  # MRC 점수 목록 저장
        
        # 점수 결합 및 결과 업데이트
        weight_flashrank = 1.0 - weight_mrc
        for i, (passage, mrc_result, flashrank_score) in enumerate(zip(passages, mrc_results, flashrank_scores)):
            # MRC 점수 저장
            mrc_score = mrc_result['answerability']
            mrc_scores.append(mrc_score)
            
            # MRC 결과 저장 - 상위 레벨에 명확하게 저장
            passage['mrc_answer'] = mrc_result['answer']
            passage['mrc_char_ids'] = mrc_result['char_ids']
            passage['mrc_score'] = mrc_score
            passage['flashrank_score'] = flashrank_score
            
            # 하이브리드 점수 계산
            hybrid_score = (flashrank_score * weight_flashrank) + (mrc_score * weight_mrc)
            passage['hybrid_score'] = hybrid_score
            
            # 하이브리드 모드에서는 hybrid_score를 최종 점수로 사용
            passage['score'] = hybrid_score
            
            # 메타데이터 필드가 없으면 생성
            if 'metadata' not in passage:
                passage['metadata'] = {}
                
            # 메타데이터에도 점수 정보 저장 (중복 저장으로 안전성 확보)
            if passage['metadata'] is not None:  # metadata가 None이 아닌 경우에만 처리
                passage['metadata']['mrc_score'] = mrc_score
                passage['metadata']['flashrank_score'] = flashrank_score
                passage['metadata']['hybrid_score'] = hybrid_score
                passage['metadata']['mrc_weight'] = weight_mrc
            
            # original_id 값 로깅
            if i < 5 or i >= len(passages) - 5:  # 처음 5개와 마지막 5개만 로깅
                logger.info(f"[DEBUG] Passage {i}: original_id={passage.get('original_id', 'N/A')}, flashrank={flashrank_score:.4f}, mrc={mrc_score:.4f}, hybrid={passage['hybrid_score']:.4f}")
        
        # 하이브리드 점수로 정렬
        reranked_passages = sorted(passages, key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        # 정렬 후 original_id 값 로깅
        logger.info(f"[DEBUG] 정렬 후 original_id 샘플: {[p.get('original_id', 'N/A') for p in reranked_passages[:5]]}")
        if len(reranked_passages) > 5:
            logger.info(f"[DEBUG] 정렬 후 마지막 항목들 original_id: {[p.get('original_id', 'N/A') for p in reranked_passages[-5:]]}")
        
        # 메타데이터 중복 제거 - 각 결과 항목에서 metadata 내부의 필드를 상위 레벨로 이동하고 metadata 필드 제거
        for i, passage in enumerate(reranked_passages):
            if "metadata" in passage and passage["metadata"] is not None:
                try:
                    # metadata 내부의 모든 필드를 상위 레벨로 복사
                    for key, value in passage["metadata"].items():
                        if key not in passage:  # 이미 존재하는 필드는 덮어쓰지 않음
                            passage[key] = value
                    
                    # 중요 메타데이터 필드 로깅 (처음 5개와 마지막 5개만)
                    if i < 5 or i >= len(reranked_passages) - 5:
                        meta_keys = list(passage["metadata"].keys())
                        logger.info(f"[DEBUG] Passage {i} 메타데이터 키: {meta_keys[:10]}{'...' if len(meta_keys) > 10 else ''}")
                    
                    # metadata 필드 제거
                    del passage["metadata"]
                except AttributeError:
                    # metadata가 None이거나 items() 메소드가 없는 경우
                    logger.warning(f"metadata 필드가 dictionary가 아닙니다: {type(passage['metadata'])}")
                    passage["metadata"] = {}
            
            # MRC 관련 필드가 있는지 확인하고 없으면 기본값 설정
            if "mrc_score" in passage and "mrc_answer" not in passage:
                passage["mrc_answer"] = ""
            if "mrc_score" in passage and "mrc_char_ids" not in passage:
                passage["mrc_char_ids"] = []
        
        # top_k 적용 (점수 목록도 함께 정렬)
        if top_k is not None and isinstance(top_k, int) and top_k > 0:
            # 원본 인덱스 정보를 유지하면서 상위 결과 선택
            logger.info(f"[DEBUG] top_k 적용 전 결과 수: {len(reranked_passages)}")
            logger.info(f"[DEBUG] top_k={top_k} 적용")
            
            # top_k 적용 전 마지막 항목 original_id 로깅
            if len(reranked_passages) > 0:
                logger.info(f"[DEBUG] top_k 적용 전 마지막 항목: original_id={reranked_passages[-1].get('original_id', 'N/A')}")
            
            reranked_passages = reranked_passages[:top_k]
            
            # top_k 적용 후 마지막 항목 original_id 로깅
            if len(reranked_passages) > 0:
                logger.info(f"[DEBUG] top_k 적용 후 마지막 항목: original_id={reranked_passages[-1].get('original_id', 'N/A')}")
            
            # 같은 순서로 mrc_scores 재정렬 (필요한 경우)
            if return_mrc_scores:
                mrc_scores = mrc_scores[:top_k]
        
        # 최종 결과의 original_id 값 로깅
        logger.info(f"[DEBUG] 최종 결과 original_id: {[p.get('original_id', 'N/A') for p in reranked_passages]}")
        
        # 최종 결과의 주요 필드 로깅
        if len(reranked_passages) > 0:
            last_passage = reranked_passages[-1]
            logger.info(f"[DEBUG] 마지막 항목 필드: {list(last_passage.keys())}")
            logger.info(f"[DEBUG] 마지막 항목 title: {last_passage.get('title', 'N/A')}")
            logger.info(f"[DEBUG] 마지막 항목 author: {last_passage.get('author', 'N/A')}")
            logger.info(f"[DEBUG] 마지막 항목 tags: {last_passage.get('tags', 'N/A')}")
        
        logger.info(f"하이브리드 재랭킹 완료: {len(reranked_passages)} 결과 반환")
        
        if return_mrc_scores:
            return reranked_passages, mrc_scores
        else:
            return reranked_passages 