import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from .controller import MRCController

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # DEBUG 레벨로 변경하여 더 상세한 로그 출력

# 파일 로그 추가 (볼륨에 저장)
try:
    file_handler = logging.FileHandler('/var/log/reranker/reranker_detail.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    # 파일 핸들러의 로그 레벨도 DEBUG로 설정
    file_handler.setLevel(logging.DEBUG)
    logger.info("MRC 리랭커 로그 파일 설정 완료: /var/log/reranker/reranker_detail.log (DEBUG 레벨)")
except Exception as e:
    logger.warning(f"MRC 리랭커 로그 파일 설정 실패: {str(e)}")

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
        
        # id 값 확인 로깅
        logger.info(f"[DEBUG] 입력 패시지 id 샘플: {[p.get('id', 'N/A') for p in passages[:3]]}")
        
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
                      original_scores: List[float] = None,  # 원본 점수 파라미터 추가
                      weight_mrc: float = 0.7,
                      weight_flashrank: float = 0.5,
                      weight_original: float = 0.2,
                      normalization_params: Dict[str, Dict] = None,
                      mrc_score_threshold: float = 0.1,
                      top_k: Optional[int] = None,
                      return_mrc_scores: bool = False) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[float]]]:
        """
        FlashRank와 MRC 점수를 결합한 하이브리드 재랭킹을 수행합니다.
        
        Args:
            query: 사용자 쿼리
            passages: 재랭킹할 패시지 목록
            flashrank_scores: FlashRank 점수 목록
            original_scores: 원본 검색 점수 목록 (없으면 passage에서 추출)
            weight_flashrank: FlashRank 점수 가중치
            weight_mrc: MRC 점수 가중치
            weight_original: 원본 점수 가중치
            normalization_params: 정규화 파라미터
            mrc_score_threshold: MRC 점수 임계값 (이 값 이상일 때만 MRC 점수 반영)
            top_k: 상위 k개 결과만 반환
            return_mrc_scores: MRC 점수도 함께 반환할지 여부
            
        Returns:
            재랭킹된 패시지 목록 또는 (패시지 목록, MRC 점수 목록) 튜플
        """
        logger.info(f"하이브리드 재랭킹 시작: query='{query}', passages={len(passages)}, "
                   f"weight_flashrank={weight_flashrank}, weight_mrc={weight_mrc}, weight_original={weight_original}")
        
        # 정규화 파라미터가 없으면 기본값 사용
        if normalization_params is None:
            normalization_params = {
                "flashrank": {"mean": 0.6184, "std": 0.0498, "min": 0.4877, "max": 0.7177, "z_min": -2.62, "z_max": 1.99},
                "mrc": {"mean": 0.0518, "std": 0.0794, "min": 0.0011, "max": 0.3841, "z_min": -0.64, "z_max": 4.19},
                "original": {"mean": 0.5872, "std": 0.0407, "min": 0.4566, "max": 0.6693, "z_min": -3.21, "z_max": 2.02}
            }
        
        # 입력 데이터 로깅 - id 값 확인
        logger.info(f"[DEBUG] 입력 패시지 id 샘플: {[p.get('id', 'N/A') for p in passages[:5]]}")
        
        # id 값의 분포 확인
        ids = [p.get('id', 'N/A') for p in passages if 'id' in p]
        if ids:
            logger.info(f"[DEBUG-ID] id 값 존재: {len(ids)}/{len(passages)}")
            logger.info(f"[DEBUG-ID] id 샘플: {ids[:5]}")
        else:
            logger.info(f"[DEBUG-ID] id 값이 없음")
        
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
        # original_scores는 이미 파라미터로 받았으므로 새로 생성하지 않음
        
        # 정규화를 위한 Z-점수 계산 및 스케일링을 위한 최소/최대값 초기화
        max_final_score = float('-inf')
        min_final_score = float('inf')
        
        # 디버깅용 로그 카운터
        log_count = 0
        
        # 정규화 파라미터 추출
        flashrank_mean = normalization_params["flashrank"].get("mean", 0.6184)
        flashrank_std = normalization_params["flashrank"].get("std", 0.0498)
        mrc_mean = normalization_params["mrc"].get("mean", 0.0518)
        mrc_std = normalization_params["mrc"].get("std", 0.0794)
        original_mean = normalization_params["original"].get("mean", 0.5872)
        original_std = normalization_params["original"].get("std", 0.0407)
        
        # 정규화 파라미터 로깅
        logger.info(f"[DEBUG-NORM] 정규화 파라미터: flashrank(mean={flashrank_mean:.4f}, std={flashrank_std:.4f}), "
                   f"mrc(mean={mrc_mean:.4f}, std={mrc_std:.4f}), original(mean={original_mean:.4f}, std={original_std:.4f})")
        logger.info(f"[DEBUG-NORM] MRC 임계치: {mrc_score_threshold}")
        
        # 점수 정규화 및 결과 업데이트
        for i, (passage, mrc_result, flashrank_score) in enumerate(zip(passages, mrc_results, flashrank_scores)):
            # 원본 점수 저장 및 로깅
            if original_scores is not None and i < len(original_scores):
                # 파라미터로 전달된 original_scores 사용
                original_score = original_scores[i]
                original_score_exists = True
                original_score_raw = original_score
            else:
                # 기존 방식대로 passage에서 추출
                original_score_exists = "original_score" in passage
                original_score_raw = passage.get("original_score", "없음")
                original_score = passage.get("original_score", original_mean)
            
            # 로깅 추가 - 처음 5개와 마지막 5개 항목만
            if i < 5 or i >= len(passages) - 5:
                passage_id = passage.get('id', 'unknown')
                logger.info(f"[ORIGINAL-SCORE-DEBUG] 항목 {i} (ID {passage_id}): original_score 존재={original_score_exists}, 원시값={original_score_raw}, 사용값={original_score}")
                # 전체 passage 구조 로깅
                logger.info(f"[PASSAGE-DEBUG] 항목 {i}: 키={list(passage.keys())}")
            
            # MRC 점수 저장
            mrc_score = mrc_result['answerability']
            mrc_scores.append(mrc_score)
            
            # MRC 결과 저장 - 상위 레벨에 명확하게 저장
            passage['mrc_answer'] = mrc_result['answer']
            passage['mrc_char_ids'] = mrc_result['char_ids']
            passage['mrc_score'] = mrc_score
            passage['flashrank_score'] = flashrank_score
            passage['original_score'] = original_score
            
            # Z-점수 정규화
            flashrank_z = (flashrank_score - flashrank_mean) / flashrank_std if flashrank_std > 0 else 0
            mrc_z = (mrc_score - mrc_mean) / mrc_std if mrc_std > 0 else 0
            original_z = (original_score - original_mean) / original_std if original_std > 0 else 0
            
            # MRC 임계치 적용
            mrc_contribution = weight_mrc * mrc_z if mrc_score >= mrc_score_threshold else 0.0
            
            # 최종 점수 계산
            final_score = (weight_flashrank * flashrank_z) + mrc_contribution + (weight_original * original_z)
            
            # 로그 출력 (처음 5개와 마지막 5개만)
            if log_count < 5 or log_count >= len(passages) - 5:
                passage_id = passage.get('id', 'unknown')
                logger.info(f"[DEBUG-SCORE] ID {passage_id}: "
                           f"flashrank({flashrank_score:.4f} → z={flashrank_z:.4f}), "
                           f"mrc({mrc_score:.4f} → z={mrc_z:.4f}, contrib={mrc_contribution:.4f}), "
                           f"original({original_score:.4f} → z={original_z:.4f}), "
                           f"final={final_score:.4f}")
            
            log_count += 1
            
            # 최종 점수 저장
            passage['hybrid_score_raw'] = final_score
            
            # 최대/최소 점수 업데이트
            max_final_score = max(max_final_score, final_score)
            min_final_score = min(min_final_score, final_score)
        
        # 최종 점수 스케일링 (0~1 범위로)
        score_range = max_final_score - min_final_score
        logger.info(f"[DEBUG-SCALE] 점수 범위: min={min_final_score:.4f}, max={max_final_score:.4f}, range={score_range:.4f}")
        
        # 이론적 최대/최소값 계산
        flashrank_z_min = normalization_params["flashrank"].get("z_min", -2.62)
        flashrank_z_max = normalization_params["flashrank"].get("z_max", 1.99)
        mrc_z_min = normalization_params["mrc"].get("z_min", -0.64)
        mrc_z_max = normalization_params["mrc"].get("z_max", 4.19)
        original_z_min = normalization_params["original"].get("z_min", -3.21)
        original_z_max = normalization_params["original"].get("z_max", 2.02)
        
        # 이론적 최대/최소 점수 계산 (MRC 임계치 고려)
        theoretical_min = (weight_flashrank * flashrank_z_min + 
                          (weight_mrc * mrc_z_min if mrc_score_threshold == 0 else 0) + 
                          weight_original * original_z_min)
        
        theoretical_max = (weight_flashrank * flashrank_z_max + 
                          weight_mrc * mrc_z_max + 
                          weight_original * original_z_max)
        
        theoretical_range = theoretical_max - theoretical_min
        
        logger.info(f"[DEBUG-THEORY] 이론적 점수 범위: min={theoretical_min:.4f}, max={theoretical_max:.4f}, range={theoretical_range:.4f}")
        
        if theoretical_range > 0:
            for passage in passages:
                raw_score = passage['hybrid_score_raw']
                # 이론적 범위 기준으로 스케일링
                scaled_score = (raw_score - theoretical_min) / theoretical_range
                # 0-1 범위를 벗어날 수 있으나, 필요하다면 클램핑 가능
                scaled_score = max(0, min(1, scaled_score))  # 0-1 사이로 클램핑 (선택적)
                
                passage['hybrid_score'] = scaled_score
                passage['score'] = scaled_score  # 최종 점수로 사용
                
                # 첫 5개와 마지막 5개 결과만 로깅
                passage_id = passage.get('id', 'unknown')
                if passage_id in [p.get('id', 'unknown') for p in passages[:5]] or passage_id in [p.get('id', 'unknown') for p in passages[-5:]]:
                    logger.info(f"[DEBUG-SCALE] ID {passage_id}: raw={raw_score:.4f} → scaled={scaled_score:.4f}")
        else:
            # 이론적 범위가 0인 경우 (발생하지 않아야 함)
            logger.warning(f"[DEBUG-SCALE] 이론적 점수 범위가 0입니다! 기본값 0.5로 설정")
            for passage in passages:
                passage['hybrid_score'] = 0.5  # 기본값
                passage['score'] = 0.5
        
        # 하이브리드 점수로 정렬
        reranked_passages = sorted(passages, key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        # 정렬 후 id 값 로깅
        logger.info(f"[DEBUG] 정렬 후 id 샘플: {[p.get('id', 'N/A') for p in reranked_passages[:5]]}")
        if len(reranked_passages) > 5:
            logger.info(f"[DEBUG] 정렬 후 마지막 항목들 id: {[p.get('id', 'N/A') for p in reranked_passages[-5:]]}")
        
        # 메타데이터 중복 제거 - 각 결과 항목에서 metadata 내부의 필드를 상위 레벨로 이동하고 metadata 필드 제거
        for i, passage in enumerate(reranked_passages):
            # id 값이 있는지 확인하고 로깅
            has_id = 'id' in passage
            id_value = passage.get('id', 'N/A')
            if i < 5 or i >= len(reranked_passages) - 5:  # 처음 5개와 마지막 5개만 로깅
                logger.info(f"[DEBUG-ID] 메타데이터 처리 전 항목 {i}: id 존재={has_id}, 값={id_value}")
                
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
            
            # id 값이 있는지 다시 확인하고 로깅
            has_id_after = 'id' in passage
            if i < 5 or i >= len(reranked_passages) - 5:  # 처음 5개와 마지막 5개만 로깅
                logger.info(f"[DEBUG-ID] 메타데이터 처리 후 항목 {i}: id 존재={has_id_after}, 값={id_value}")
                
                # 필드 변화 확인
                logger.info(f"[DEBUG-FIELDS] 항목 {i} 필드: {list(passage.keys())}")
        
        # top_k 적용 (점수 목록도 함께 정렬)
        if top_k is not None and isinstance(top_k, int) and top_k > 0:
            # 원본 인덱스 정보를 유지하면서 상위 결과 선택
            logger.info(f"[DEBUG] top_k 적용 전 결과 수: {len(reranked_passages)}")
            logger.info(f"[DEBUG] top_k={top_k} 적용")
            
            # top_k 적용 전 마지막 항목 id 로깅
            if len(reranked_passages) > 0:
                logger.info(f"[DEBUG] top_k 적용 전 마지막 항목: id={reranked_passages[-1].get('id', 'N/A')}")
            
            reranked_passages = reranked_passages[:top_k]
            
            # top_k 적용 후 마지막 항목 id 로깅
            if len(reranked_passages) > 0:
                logger.info(f"[DEBUG] top_k 적용 후 마지막 항목: id={reranked_passages[-1].get('id', 'N/A')}")
            
            # 같은 순서로 mrc_scores 재정렬 (필요한 경우)
            if return_mrc_scores:
                mrc_scores = mrc_scores[:top_k]
        
        # 최종 결과의 id 값 로깅
        logger.info(f"[DEBUG] 최종 결과 id: {[p.get('id', 'N/A') for p in reranked_passages]}")
        
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