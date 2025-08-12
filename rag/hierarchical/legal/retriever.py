"""
법령 문서 검색 클래스

법령의 위계적 구조를 활용한 고도화된 검색 기능을 제공합니다.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from pymilvus import Collection

from ..base.retriever import BaseHierarchicalRetriever
from ..base.advanced_retriever import AdvancedHierarchicalRetriever


class LegalRetriever(BaseHierarchicalRetriever, AdvancedHierarchicalRetriever):
    """법령 전용 검색 클래스"""
    
    def __init__(self, existing_interact_manager=None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager 인스턴스 (검색 기능 재사용)
        """
        super().__init__(existing_interact_manager)
        self.logger = logging.getLogger(__name__)
        
        # 법령 검색 특화 설정
        self.legal_node_types = ["law", "part", "chapter", "section", "article", "paragraph", "item", "subitem"]
        self.legal_search_weights = {
            "article": 1.0,      # 조문이 가장 중요
            "paragraph": 0.9,    # 항
            "item": 0.8,         # 호
            "chapter": 0.7,      # 장
            "section": 0.6,      # 절
            "subitem": 0.5,      # 목
            "part": 0.4,         # 편
            "law": 0.3           # 법령명
        }
        
        # 법령 쿼리 패턴
        self.legal_patterns = {
            "article_ref": r"제(\d+)조(?:의(\d+))?",
            "paragraph_ref": r"제(\d+)항",
            "item_ref": r"제(\d+)호",
            "law_ref": r"「(.+?)」",
            "article_range": r"제(\d+)조부터\s*제(\d+)조까지",
        }
        
        self.logger.info("법령 검색기 초기화 완료")
    
    def search_legal_documents(self, collection_name: str, query: str,
                              search_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        법령 문서 전용 검색
        
        Args:
            collection_name: 검색할 컬렉션
            query: 검색 쿼리
            search_params: 검색 옵션
                - search_mode: "semantic", "legal_reference", "hybrid" (기본: "hybrid")
                - target_node_types: 검색할 노드 타입들 (기본: ["article", "paragraph"])
                - include_context: 상하위 조문 포함 여부 (기본: True)
                - law_type_filter: 법령 유형 필터 (예: "법률", "시행령")
                - date_range: 시행일 범위 필터
                
        Returns:
            List[Dict]: 검색 결과
        """
        try:
            self.logger.info(f"법령 검색 시작: {query}")
            
            # 기본 파라미터 설정
            params = {
                "search_mode": "hybrid",
                "target_node_types": ["article", "paragraph", "item"],
                "include_context": True,
                "law_type_filter": None,
                "date_range": None,
                "top_k": 10,
                "expand_hierarchy": True,
                "hierarchy_weight": 0.4,
                **(search_params or {})
            }
            
            # 쿼리 분석 및 전처리
            analyzed_query = self._analyze_legal_query(query)
            
            # 검색 모드에 따른 처리
            if analyzed_query["has_legal_references"] and params["search_mode"] in ["legal_reference", "hybrid"]:
                # 법조문 참조 검색
                reference_results = self._search_by_legal_references(
                    collection_name, analyzed_query, params
                )
            else:
                reference_results = []
            
            if params["search_mode"] in ["semantic", "hybrid"]:
                # 의미 기반 검색
                semantic_results = self._search_by_semantics(
                    collection_name, analyzed_query["processed_query"], params
                )
            else:
                semantic_results = []
            
            # 결과 병합 및 후처리
            combined_results = self._combine_search_results(
                reference_results, semantic_results, params
            )
            
            # 법령 특화 후처리
            final_results = self._post_process_legal_results(combined_results, params)
            
            self.logger.info(f"법령 검색 완료: {len(final_results)}개 결과")
            return final_results
            
        except Exception as e:
            self.logger.error(f"법령 검색 중 오류: {e}")
            return []
    
    def _analyze_legal_query(self, query: str) -> Dict[str, Any]:
        """법령 쿼리 분석"""
        try:
            analysis = {
                "original_query": query,
                "processed_query": query,
                "has_legal_references": False,
                "article_references": [],
                "paragraph_references": [],
                "item_references": [],
                "law_references": [],
                "article_ranges": []
            }
            
            # 조문 참조 추출
            article_matches = re.finditer(self.legal_patterns["article_ref"], query)
            for match in article_matches:
                article_num = match.group(1)
                article_sub = match.group(2)
                ref = f"제{article_num}조"
                if article_sub:
                    ref += f"의{article_sub}"
                analysis["article_references"].append(ref)
                analysis["has_legal_references"] = True
            
            # 항 참조 추출
            paragraph_matches = re.finditer(self.legal_patterns["paragraph_ref"], query)
            for match in paragraph_matches:
                paragraph_num = match.group(1)
                ref = f"제{paragraph_num}항"
                analysis["paragraph_references"].append(ref)
                analysis["has_legal_references"] = True
            
            # 호 참조 추출
            item_matches = re.finditer(self.legal_patterns["item_ref"], query)
            for match in item_matches:
                item_num = match.group(1)
                ref = f"제{item_num}호"
                analysis["item_references"].append(ref)
                analysis["has_legal_references"] = True
            
            # 법령명 참조 추출
            law_matches = re.finditer(self.legal_patterns["law_ref"], query)
            for match in law_matches:
                law_name = match.group(1)
                analysis["law_references"].append(law_name)
                analysis["has_legal_references"] = True
            
            # 조문 범위 추출
            range_matches = re.finditer(self.legal_patterns["article_range"], query)
            for match in range_matches:
                start_article = match.group(1)
                end_article = match.group(2)
                analysis["article_ranges"].append((start_article, end_article))
                analysis["has_legal_references"] = True
            
            # 참조 제거한 순수 의미 쿼리 생성
            processed_query = query
            for pattern in self.legal_patterns.values():
                processed_query = re.sub(pattern, "", processed_query)
            analysis["processed_query"] = processed_query.strip()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"쿼리 분석 중 오류: {e}")
            return {"original_query": query, "processed_query": query, "has_legal_references": False}
    
    def _search_by_legal_references(self, collection_name: str, 
                                   analyzed_query: Dict[str, Any],
                                   params: Dict) -> List[Dict[str, Any]]:
        """법조문 참조 기반 검색"""
        try:
            results = []
            collection = Collection(collection_name)
            collection.load()
            
            # 조문 참조 검색
            for article_ref in analyzed_query["article_references"]:
                article_results = self._search_by_article_reference(
                    collection, article_ref, params
                )
                results.extend(article_results)
            
            # 항 참조 검색 (현재 조문 컨텍스트에서)
            for paragraph_ref in analyzed_query["paragraph_references"]:
                paragraph_results = self._search_by_paragraph_reference(
                    collection, paragraph_ref, params
                )
                results.extend(paragraph_results)
            
            # 호 참조 검색
            for item_ref in analyzed_query["item_references"]:
                item_results = self._search_by_item_reference(
                    collection, item_ref, params
                )
                results.extend(item_results)
            
            # 법령명 참조 검색
            for law_ref in analyzed_query["law_references"]:
                law_results = self._search_by_law_reference(
                    collection, law_ref, params
                )
                results.extend(law_results)
            
            # 조문 범위 검색
            for start_article, end_article in analyzed_query["article_ranges"]:
                range_results = self._search_by_article_range(
                    collection, start_article, end_article, params
                )
                results.extend(range_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"법조문 참조 검색 중 오류: {e}")
            return []
    
    def _search_by_article_reference(self, collection: Collection, 
                                   article_ref: str, params: Dict) -> List[Dict[str, Any]]:
        """조문 참조 검색"""
        try:
            # 정확한 조문 매칭
            expr = f'article_number == "{article_ref}"'
            
            search_params = {
                "data": [[0.0] * 1024],  # 더미 벡터 (expr 기반 검색)
                "anns_field": "content_embedding",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 100,
                "expr": expr,
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            return self._format_milvus_results(results, score_boost=1.0)
            
        except Exception as e:
            self.logger.error(f"조문 참조 검색 중 오류: {e}")
            return []
    
    def _search_by_paragraph_reference(self, collection: Collection,
                                     paragraph_ref: str, params: Dict) -> List[Dict[str, Any]]:
        """항 참조 검색"""
        try:
            expr = f'paragraph_number == "{paragraph_ref}"'
            
            search_params = {
                "data": [[0.0] * 1024],
                "anns_field": "content_embedding", 
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 50,
                "expr": expr,
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            return self._format_milvus_results(results, score_boost=0.9)
            
        except Exception as e:
            self.logger.error(f"항 참조 검색 중 오류: {e}")
            return []
    
    def _search_by_item_reference(self, collection: Collection,
                                item_ref: str, params: Dict) -> List[Dict[str, Any]]:
        """호 참조 검색"""
        try:
            expr = f'item_number == "{item_ref}"'
            
            search_params = {
                "data": [[0.0] * 1024],
                "anns_field": "content_embedding",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 30,
                "expr": expr,
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            return self._format_milvus_results(results, score_boost=0.8)
            
        except Exception as e:
            self.logger.error(f"호 참조 검색 중 오류: {e}")
            return []
    
    def _search_by_law_reference(self, collection: Collection,
                               law_name: str, params: Dict) -> List[Dict[str, Any]]:
        """법령명 참조 검색"""
        try:
            expr = f'law_title like "%{law_name}%"'
            
            search_params = {
                "data": [[0.0] * 1024],
                "anns_field": "content_embedding",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 100,
                "expr": expr,
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            return self._format_milvus_results(results, score_boost=0.7)
            
        except Exception as e:
            self.logger.error(f"법령명 참조 검색 중 오류: {e}")
            return []
    
    def _search_by_article_range(self, collection: Collection,
                               start_article: str, end_article: str, 
                               params: Dict) -> List[Dict[str, Any]]:
        """조문 범위 검색"""
        try:
            # 범위 내 모든 조문 검색
            start_num = int(start_article)
            end_num = int(end_article)
            
            article_exprs = []
            for i in range(start_num, end_num + 1):
                article_exprs.append(f'article_number == "제{i}조"')
            
            expr = " or ".join(article_exprs)
            
            search_params = {
                "data": [[0.0] * 1024],
                "anns_field": "content_embedding",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 200,
                "expr": expr,
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            return self._format_milvus_results(results, score_boost=0.9)
            
        except Exception as e:
            self.logger.error(f"조문 범위 검색 중 오류: {e}")
            return []
    
    def _search_by_semantics(self, collection_name: str, query: str, 
                           params: Dict) -> List[Dict[str, Any]]:
        """의미 기반 검색"""
        try:
            # 노드 타입 필터 생성
            node_type_filter = self._build_node_type_filter(params.get("target_node_types", []))
            
            # 기존 벡터 검색 활용
            filter_conditions = params.get("filter_conditions", {})
            if node_type_filter:
                filter_conditions["node_type_filter"] = node_type_filter
            
            search_params = {
                "top_k": params.get("top_k", 10) * 2,  # 더 많이 검색해서 필터링
                "expand_hierarchy": False,  # 의미 검색에서는 확장 안함
                "filter_conditions": filter_conditions
            }
            
            results = self._vector_search(collection_name, query, search_params)
            
            # 법령 가중치 적용
            weighted_results = self._apply_legal_weights(results)
            
            return weighted_results
            
        except Exception as e:
            self.logger.error(f"의미 기반 검색 중 오류: {e}")
            return []
    
    def _build_node_type_filter(self, target_node_types: List[str]) -> Optional[str]:
        """노드 타입 필터 생성"""
        try:
            if not target_node_types:
                return None
            
            # Milvus expr 형식으로 변환
            type_conditions = [f'node_type == "{node_type}"' for node_type in target_node_types]
            return " or ".join(type_conditions)
            
        except Exception as e:
            self.logger.error(f"노드 타입 필터 생성 중 오류: {e}")
            return None
    
    def _apply_legal_weights(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """법령 특화 가중치 적용"""
        try:
            weighted_results = []
            
            for result in results:
                node_type = result.get("entity", {}).get("node_type", "content")
                base_score = result.get("score", 0.0)
                
                # 법령 노드 타입별 가중치 적용
                weight = self.legal_search_weights.get(node_type, 0.5)
                final_score = base_score * weight
                
                result["legal_score"] = final_score
                result["node_type_weight"] = weight
                weighted_results.append(result)
            
            # 법령 점수로 재정렬
            weighted_results.sort(key=lambda x: x.get("legal_score", 0.0), reverse=True)
            
            return weighted_results
            
        except Exception as e:
            self.logger.error(f"법령 가중치 적용 중 오류: {e}")
            return results
    
    def _format_milvus_results(self, raw_results, score_boost: float = 1.0) -> List[Dict[str, Any]]:
        """Milvus 검색 결과 포맷팅"""
        try:
            formatted_results = []
            
            for hits in raw_results:
                for hit in hits:
                    try:
                        result = {
                            "id": getattr(hit, 'id', ''),
                            "score": getattr(hit, 'distance', 0.0) * score_boost,
                            "entity": {}
                        }
                        
                        # entity 정보 추출
                        if hasattr(hit, 'entity') and hit.entity:
                            if isinstance(hit.entity, dict):
                                result["entity"] = hit.entity
                            
                        # 추가 필드들 추출
                        for attr in ['node_id', 'document_id', 'node_type', 'title', 'content',
                                   'hierarchy_level', 'article_number', 'paragraph_number']:
                            if hasattr(hit, attr):
                                result[attr] = getattr(hit, attr)
                            elif hasattr(hit, 'entity') and isinstance(hit.entity, dict):
                                if attr in hit.entity:
                                    result[attr] = hit.entity[attr]
                        
                        formatted_results.append(result)
                        
                    except Exception as e:
                        self.logger.warning(f"결과 포맷팅 중 오류: {e}")
                        continue
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"결과 포맷팅 중 오류: {e}")
            return []
    
    def _combine_search_results(self, reference_results: List[Dict[str, Any]],
                              semantic_results: List[Dict[str, Any]], 
                              params: Dict) -> List[Dict[str, Any]]:
        """검색 결과 병합"""
        try:
            # 참조 검색 결과에 높은 우선순위
            for result in reference_results:
                result["search_type"] = "legal_reference"
                result["final_score"] = result.get("score", 0.0) + 1.0  # 보너스 점수
            
            # 의미 검색 결과
            for result in semantic_results:
                result["search_type"] = "semantic"
                result["final_score"] = result.get("legal_score", result.get("score", 0.0))
            
            # 중복 제거 (ID 기반)
            seen_ids = set()
            combined_results = []
            
            # 참조 검색 결과 우선 추가
            for result in reference_results:
                result_id = result.get("id", "")
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    combined_results.append(result)
            
            # 의미 검색 결과 추가 (중복 제외)
            for result in semantic_results:
                result_id = result.get("id", "")
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    combined_results.append(result)
            
            # 최종 점수로 정렬
            combined_results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"결과 병합 중 오류: {e}")
            return reference_results + semantic_results
    
    def _expand_with_hierarchy(self, collection_name: str, 
                              base_results: List[Dict], 
                              params: Dict) -> List[Dict[str, Any]]:
        """법령 위계 확장"""
        try:
            if not params.get("include_context", True):
                return base_results
            
            expanded_results = base_results.copy()
            
            for result in base_results:
                # 상위 조문 (부모) 추가
                parent_results = self._find_parent_articles(collection_name, result)
                expanded_results.extend(parent_results)
                
                # 하위 조문 (자식) 추가  
                child_results = self._find_child_articles(collection_name, result)
                expanded_results.extend(child_results)
            
            # 중복 제거
            expanded_results = self._remove_duplicates(expanded_results)
            
            return expanded_results
            
        except Exception as e:
            self.logger.error(f"위계 확장 중 오류: {e}")
            return base_results
    
    def _find_parent_articles(self, collection_name: str, 
                            result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """부모 조문 찾기"""
        try:
            # 현재 결과의 부모 노드 ID 확인
            parent_node_id = result.get("entity", {}).get("parent_node_id", "")
            
            if not parent_node_id:
                return []
            
            collection = Collection(collection_name)
            collection.load()
            
            # 부모 노드 검색
            expr = f'node_id == "{parent_node_id}"'
            search_params = {
                "data": [[0.0] * 1024],
                "anns_field": "content_embedding",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 5,
                "expr": expr,
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            parent_results = self._format_milvus_results(results, score_boost=0.6)
            
            # 컨텍스트 표시
            for parent in parent_results:
                parent["relation_type"] = "parent"
                parent["context_info"] = "상위 조문"
            
            return parent_results
            
        except Exception as e:
            self.logger.error(f"부모 조문 검색 중 오류: {e}")
            return []
    
    def _find_child_articles(self, collection_name: str,
                           result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """자식 조문 찾기"""
        try:
            # 현재 결과의 노드 ID
            current_node_id = result.get("id", "") or result.get("node_id", "")
            
            if not current_node_id:
                return []
            
            collection = Collection(collection_name)
            collection.load()
            
            # 자식 노드들 검색
            expr = f'parent_node_id == "{current_node_id}"'
            search_params = {
                "data": [[0.0] * 1024],
                "anns_field": "content_embedding",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 20,
                "expr": expr,
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            child_results = self._format_milvus_results(results, score_boost=0.7)
            
            # 컨텍스트 표시
            for child in child_results:
                child["relation_type"] = "child"
                child["context_info"] = "하위 조문"
            
            return child_results
            
        except Exception as e:
            self.logger.error(f"자식 조문 검색 중 오류: {e}")
            return []
    
    def _post_process_legal_results(self, results: List[Dict[str, Any]], 
                                  params: Dict) -> List[Dict[str, Any]]:
        """법령 결과 후처리"""
        try:
            # 결과 수 제한
            top_k = params.get("top_k", 10)
            limited_results = results[:top_k]
            
            # 법령 메타데이터 보강
            for result in limited_results:
                self._enrich_legal_metadata(result)
            
            return limited_results
            
        except Exception as e:
            self.logger.error(f"법령 결과 후처리 중 오류: {e}")
            return results
    
    def _enrich_legal_metadata(self, result: Dict[str, Any]) -> None:
        """법령 메타데이터 보강"""
        try:
            entity = result.get("entity", {})
            
            # 조문 정보 요약
            legal_info = {
                "법령명": entity.get("law_title", ""),
                "조문번호": entity.get("article_number", ""),
                "항번호": entity.get("paragraph_number", ""),
                "호번호": entity.get("item_number", ""),
                "노드타입": entity.get("node_type", ""),
                "위계레벨": entity.get("hierarchy_level", 0)
            }
            
            # 빈 값 제거
            legal_info = {k: v for k, v in legal_info.items() if v}
            
            result["legal_metadata"] = legal_info
            
        except Exception as e:
            self.logger.error(f"메타데이터 보강 중 오류: {e}")
    
    def get_article_context(self, collection_name: str, article_number: str) -> Dict[str, Any]:
        """특정 조문의 전체 컨텍스트 조회"""
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # 해당 조문과 관련된 모든 노드 검색
            expr = f'article_number == "{article_number}"'
            search_params = {
                "data": [[0.0] * 1024],
                "anns_field": "content_embedding",
                "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                "limit": 100,
                "expr": expr,
                "output_fields": ["*"]
            }
            
            results = collection.search(**search_params)
            formatted_results = self._format_milvus_results(results)
            
            # 위계별로 그룹핑
            context = {
                "article": None,
                "paragraphs": [],
                "items": [],
                "subitems": [],
                "total_nodes": len(formatted_results)
            }
            
            for result in formatted_results:
                node_type = result.get("entity", {}).get("node_type", "")
                if node_type == "article":
                    context["article"] = result
                elif node_type == "paragraph":
                    context["paragraphs"].append(result)
                elif node_type == "item":
                    context["items"].append(result)
                elif node_type == "subitem":
                    context["subitems"].append(result)
            
            return context
            
        except Exception as e:
            self.logger.error(f"조문 컨텍스트 조회 중 오류: {e}")
            return {"error": str(e)}
