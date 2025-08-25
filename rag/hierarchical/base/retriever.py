"""
위계형 문서 검색 베이스 클래스

기존 RAG 시스템의 검색 기능을 확장하여 위계형 구조를 고려한 검색을 제공합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from pymilvus import Collection


class BaseHierarchicalRetriever(ABC):
    """위계형 문서 검색 베이스 클래스"""
    
    def __init__(self, existing_interact_manager=None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager 인스턴스 (검색 기능 재사용)
        """
        self.interact_manager = existing_interact_manager
        self.logger = logging.getLogger(__name__)
        
        # 위계형 검색 설정
        self.default_expansion_depth = 2  # 기본 확장 깊이
        self.max_expansion_depth = 5      # 최대 확장 깊이
        
    def search(self, collection_name: str, query: str, 
               search_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        통합 위계형 검색
        
        Args:
            collection_name: 검색할 컬렉션
            query: 검색 쿼리
            search_params: 검색 옵션
                - top_k: 결과 개수 (기본: 10)
                - expand_hierarchy: 위계 확장 여부 (기본: True)
                - expansion_depth: 확장 깊이 (기본: 2)
                - include_parents: 부모 문서 포함 여부 (기본: True)
                - include_children: 자식 문서 포함 여부 (기본: True)
                - hierarchy_weight: 위계 관계 가중치 (기본: 0.3)
                - filter_conditions: 추가 필터 조건
                
        Returns:
            List[Dict]: 검색 결과 (점수순 정렬)
        """
        try:
            start_time = time.time()
            self.logger.info(f"위계형 검색 시작: {query} in {collection_name}")
            
            # 기본 파라미터 설정
            params = {
                "top_k": 10,
                "expand_hierarchy": True,
                "expansion_depth": self.default_expansion_depth,
                "include_parents": True,
                "include_children": True,
                "hierarchy_weight": 0.3,
                "filter_conditions": {},
                "group_by_hierarchy": False,
                **(search_params or {})
            }
            
            # 1단계: 기본 벡터 검색
            base_results = self._vector_search(collection_name, query, params)
            
            if not base_results:
                self.logger.info("기본 검색 결과 없음")
                return []
            
            # 2단계: 위계 확장 (옵션)
            if params.get("expand_hierarchy"):
                expanded_results = self._expand_with_hierarchy(
                    collection_name, base_results, params
                )
            else:
                expanded_results = base_results
            
            # 3단계: 결과 후처리 및 재랭킹
            final_results = self._post_process_results(expanded_results, params)
            
            end_time = time.time()
            self.logger.info(f"위계형 검색 완료: {len(final_results)}개 결과, {end_time - start_time:.3f}초")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"위계형 검색 실패: {e}")
            return []
    
    def _vector_search(self, collection_name: str, query: str, 
                      params: Dict) -> List[Dict[str, Any]]:
        """
        기본 벡터 검색 (기존 시스템 활용)
        
        Args:
            collection_name: 컬렉션 이름
            query: 검색 쿼리
            params: 검색 파라미터
            
        Returns:
            List[Dict]: 기본 검색 결과
        """
        try:
            if not self.interact_manager:
                self.logger.error("InteractManager가 설정되지 않았습니다")
                return []
            
            # 기존 retrieve_data 함수 활용
            filter_conditions = params.get("filter_conditions", {})
            filter_conditions["domain"] = collection_name
            
            self.logger.info(f"벡터 검색 실행: top_k={params['top_k']}")
            
            # 기존 시스템의 retrieve_data 호출
            results = self.interact_manager.retrieve_data(
                query=query,
                top_k=params["top_k"] * 2,  # 확장을 위해 더 많이 검색
                filter_conditions=filter_conditions
            )
            
            # 결과 포맷 통일
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_result = {
                        "id": result.get("passage_uid", ""),
                        "doc_id": result.get("doc_id", ""),
                        "passage_id": result.get("passage_id", 0),
                        "text": result.get("text", ""),
                        "title": result.get("title", ""),
                        "score": result.get("score", 0.0),
                        "hierarchy_level": result.get("hierarchy_level", 0),
                        "parent_id": result.get("parent_id", ""),
                        "hierarchy_path": result.get("hierarchy_path", "/"),
                        "section_type": result.get("section_type", "content"),
                        "section_number": result.get("section_number", ""),
                        "entity": result  # 전체 엔티티 보존
                    }
                    formatted_results.append(formatted_result)
            
            self.logger.info(f"벡터 검색 완료: {len(formatted_results)}개 결과")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"벡터 검색 실패: {e}")
            return []
    
    @abstractmethod
    def _expand_with_hierarchy(self, collection_name: str, 
                              base_results: List[Dict], 
                              params: Dict) -> List[Dict[str, Any]]:
        """
        위계 관계를 이용한 검색 확장 (서브클래스에서 구현)
        
        Args:
            collection_name: 컬렉션 이름
            base_results: 기본 검색 결과
            params: 검색 파라미터
            
        Returns:
            List[Dict]: 확장된 검색 결과
        """
        pass
    
    def _find_related_documents(self, collection_name: str, reference_doc: Dict[str, Any],
                               relation_type: str = "both") -> List[Dict[str, Any]]:
        """
        특정 문서와 관련된 위계 문서들 찾기
        
        Args:
            collection_name: 컬렉션 이름
            reference_doc: 기준 문서
            relation_type: 관계 타입 ("parent", "children", "both")
            
        Returns:
            List[Dict]: 관련 문서들
        """
        try:
            collection = Collection(collection_name)
            collection.load()
            
            related_docs = []
            doc_id = reference_doc.get("doc_id", "")
            hierarchy_level = reference_doc.get("hierarchy_level", 0)
            hierarchy_path = reference_doc.get("hierarchy_path", "/")
            
            # 부모 문서 찾기
            if relation_type in ["parent", "both"]:
                parent_docs = self._find_parent_documents(
                    collection, doc_id, hierarchy_level, hierarchy_path
                )
                related_docs.extend(parent_docs)
            
            # 자식 문서 찾기
            if relation_type in ["children", "both"]:
                child_docs = self._find_child_documents(
                    collection, doc_id, hierarchy_level, hierarchy_path
                )
                related_docs.extend(child_docs)
            
            return related_docs
            
        except Exception as e:
            self.logger.error(f"관련 문서 검색 실패: {e}")
            return []
    
    def _find_parent_documents(self, collection: Collection, doc_id: str,
                              hierarchy_level: int, hierarchy_path: str) -> List[Dict[str, Any]]:
        """부모 문서들 찾기"""
        try:
            # 상위 레벨 문서 검색
            parent_level = hierarchy_level - 1
            if parent_level < 0:
                return []
            
            # 간단한 구현 예시 (실제로는 더 복잡한 쿼리 필요)
            search_params = {
                "data": [[0.0] * 1024],  # 더미 벡터 (실제로는 적절한 쿼리 필요)
                "anns_field": "text_emb",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 100,
                "expr": f"hierarchy_level == {parent_level}",
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            # 결과 처리 로직 (생략 - 실제 구현 시 필요)
            
            return []  # 임시 반환
            
        except Exception as e:
            self.logger.error(f"부모 문서 검색 실패: {e}")
            return []
    
    def _find_child_documents(self, collection: Collection, doc_id: str,
                             hierarchy_level: int, hierarchy_path: str) -> List[Dict[str, Any]]:
        """자식 문서들 찾기"""
        try:
            # 하위 레벨 문서 검색
            child_level = hierarchy_level + 1
            
            # 간단한 구현 예시 (실제로는 더 복잡한 쿼리 필요)
            search_params = {
                "data": [[0.0] * 1024],  # 더미 벡터
                "anns_field": "text_emb",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 100,
                "expr": f"hierarchy_level == {child_level}",
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            # 결과 처리 로직 (생략)
            
            return []  # 임시 반환
            
        except Exception as e:
            self.logger.error(f"자식 문서 검색 실패: {e}")
            return []
    
    def _post_process_results(self, results: List[Dict], 
                             params: Dict) -> List[Dict[str, Any]]:
        """
        결과 후처리 및 재랭킹
        
        Args:
            results: 원본 검색 결과
            params: 검색 파라미터
            
        Returns:
            List[Dict]: 처리된 최종 결과
        """
        try:
            if not results:
                return []
            
            # 1. 중복 제거
            unique_results = self._remove_duplicates(results)
            
            # 2. 위계 가중치 적용
            if params.get("hierarchy_weight", 0) > 0:
                weighted_results = self._apply_hierarchy_weights(
                    unique_results, params["hierarchy_weight"]
                )
            else:
                weighted_results = unique_results
            
            # 3. 점수순 정렬
            sorted_results = sorted(
                weighted_results, 
                key=lambda x: x.get("final_score", x.get("score", 0.0)), 
                reverse=True
            )
            
            # 4. 결과 수 제한
            top_k = params.get("top_k", 10)
            final_results = sorted_results[:top_k]
            
            # 5. 위계별 그룹핑 (옵션)
            if params.get("group_by_hierarchy"):
                final_results = self._group_by_hierarchy_level(final_results)
            
            self.logger.info(f"후처리 완료: {len(final_results)}개 최종 결과")
            return final_results
            
        except Exception as e:
            self.logger.error(f"결과 후처리 실패: {e}")
            return results  # 실패 시 원본 반환
    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        """중복 결과 제거"""
        try:
            seen_ids = set()
            unique_results = []
            
            for result in results:
                result_id = result.get("id") or result.get("passage_uid", "")
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            self.logger.info(f"중복 제거: {len(results)} -> {len(unique_results)}")
            return unique_results
            
        except Exception as e:
            self.logger.error(f"중복 제거 실패: {e}")
            return results
    
    def _apply_hierarchy_weights(self, results: List[Dict], weight: float) -> List[Dict]:
        """위계 관계 가중치 적용"""
        try:
            weighted_results = []
            
            for result in results:
                base_score = result.get("score", 0.0)
                hierarchy_level = result.get("hierarchy_level", 0)
                
                # 위계 레벨에 따른 가중치 계산 (예시)
                # 높은 레벨(상위 개념)일수록 약간의 가중치 부여
                level_bonus = weight * (1.0 / (hierarchy_level + 1))
                final_score = base_score + level_bonus
                
                result["final_score"] = final_score
                result["hierarchy_bonus"] = level_bonus
                weighted_results.append(result)
            
            return weighted_results
            
        except Exception as e:
            self.logger.error(f"가중치 적용 실패: {e}")
            return results
    
    def _group_by_hierarchy_level(self, results: List[Dict]) -> Dict[int, List[Dict]]:
        """위계 레벨별 그룹핑"""
        try:
            grouped = {}
            
            for result in results:
                level = result.get("hierarchy_level", 0)
                if level not in grouped:
                    grouped[level] = []
                grouped[level].append(result)
            
            # 레벨순으로 정렬된 딕셔너리 반환
            return dict(sorted(grouped.items()))
            
        except Exception as e:
            self.logger.error(f"그룹핑 실패: {e}")
            return {0: results}
    
    def get_document_context(self, collection_name: str, document_id: str,
                           context_depth: int = 2) -> Dict[str, Any]:
        """
        특정 문서의 위계적 컨텍스트 조회
        
        Args:
            collection_name: 컬렉션 이름
            document_id: 문서 ID
            context_depth: 컨텍스트 깊이
            
        Returns:
            Dict: 문서와 그 컨텍스트 정보
        """
        try:
            # 기본 문서 정보 조회
            base_doc = self._get_document_by_id(collection_name, document_id)
            if not base_doc:
                return {"error": f"문서를 찾을 수 없습니다: {document_id}"}
            
            # 컨텍스트 수집
            context = {
                "document": base_doc,
                "parents": [],
                "children": [],
                "siblings": []
            }
            
            # 부모/자식 문서들 수집
            for depth in range(1, context_depth + 1):
                # 구현 로직 (간소화)
                pass
            
            return context
            
        except Exception as e:
            self.logger.error(f"컨텍스트 조회 실패: {e}")
            return {"error": str(e)}
    
    def _get_document_by_id(self, collection_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """ID로 문서 조회"""
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # 간단한 구현 (실제로는 더 정확한 쿼리 필요)
            search_params = {
                "data": [[0.0] * 1024],  # 더미 벡터
                "anns_field": "text_emb",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 1,
                "expr": f'passage_uid == "{document_id}"',
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            # 결과 처리 (생략)
            
            return None  # 임시 반환
            
        except Exception as e:
            self.logger.error(f"문서 조회 실패: {e}")
            return None
