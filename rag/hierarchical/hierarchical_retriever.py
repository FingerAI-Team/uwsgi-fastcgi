"""
위계형 검색기

기존 RAG 검색과 완전히 호환되면서 조문 참조 기능만 추가합니다.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from pymilvus import Collection


class HierarchicalRetriever:
    """위계형 검색 클래스"""
    
    def __init__(self, existing_interact_manager=None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager 인스턴스
        """
        self.interact_manager = existing_interact_manager
        self.logger = logging.getLogger(__name__)
        
        # 법령 패턴 (간단한 버전)
        self.legal_patterns = {
            "article_ref": r"제(\d+)조(?:의(\d+))?",
            "paragraph_ref": r"제(\d+)항",
            "item_ref": r"제(\d+)호",
        }
        
        self.logger.info("✅ 간단한 위계형 검색기 초기화 완료")
    
    def search(self, collection_name: str, query: str, 
              search_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        간단한 위계형 검색
        
        Args:
            collection_name: 검색할 컬렉션
            query: 검색 쿼리
            search_params: 검색 파라미터
            
        Returns:
            List[Dict]: 검색 결과
        """
        try:
            self.logger.info(f"🔍 간단한 위계형 검색 시작: {query}")
            
            # 기본 파라미터 설정
            params = {
                "top_k": search_params.get('top_k', 10) if search_params else 10,
                **(search_params or {})
            }
            
            # 쿼리 분석
            analysis = self._analyze_query(query)
            
            results = []
            
            # 1. 조문 참조가 있으면 정확한 검색
            if analysis["has_legal_references"]:
                self.logger.info("🔍 조문 참조 검색 실행")
                results = self._search_by_legal_references(
                    collection_name, analysis, params
                )
            
            # 2. 조문 참조가 없으면 기존 벡터 검색
            if not results:
                self.logger.info("🔍 기존 벡터 검색 실행")
                results = self._vector_search(
                    collection_name, query, params
                )
            
            self.logger.info(f"✅ 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            self.logger.error(f"검색 중 오류: {e}")
            return []
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석 (간단한 버전)"""
        try:
            analysis = {
                "original_query": query,
                "has_legal_references": False,
                "article_references": [],
                "paragraph_references": [],
                "item_references": [],
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
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"쿼리 분석 중 오류: {e}")
            return {"original_query": query, "has_legal_references": False}
    
    def _search_by_legal_references(self, collection_name: str, 
                                   analysis: Dict[str, Any],
                                   params: Dict) -> List[Dict[str, Any]]:
        """조문 참조 기반 검색"""
        try:
            results = []
            collection = Collection(collection_name)
            collection.load()
            
            # 조문 참조 검색
            for article_ref in analysis["article_references"]:
                expr = f'article_number == "{article_ref}"'
                search_params = {
                    "data": [[0.0] * 1024],
                    "anns_field": "text_emb",
                    "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                    "limit": params.get("top_k", 10),
                    "expr": expr,
                    "output_fields": ["*"]
                }
                
                search_results = collection.search(**search_params)
                results.extend(self._format_results(search_results))
            
            # 항 참조 검색
            for paragraph_ref in analysis["paragraph_references"]:
                expr = f'paragraph_number == "{paragraph_ref}"'
                search_params = {
                    "data": [[0.0] * 1024],
                    "anns_field": "text_emb",
                    "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                    "limit": params.get("top_k", 10),
                    "expr": expr,
                    "output_fields": ["*"]
                }
                
                search_results = collection.search(**search_params)
                results.extend(self._format_results(search_results))
            
            # 호 참조 검색
            for item_ref in analysis["item_references"]:
                expr = f'item_number == "{item_ref}"'
                search_params = {
                    "data": [[0.0] * 1024],
                    "anns_field": "text_emb",
                    "param": {"metric_type": "COSINE", "params": {"nprobe": 16}},
                    "limit": params.get("top_k", 10),
                    "expr": expr,
                    "output_fields": ["*"]
                }
                
                search_results = collection.search(**search_params)
                results.extend(self._format_results(search_results))
            
            return results
            
        except Exception as e:
            self.logger.error(f"조문 참조 검색 중 오류: {e}")
            return []
    
    def _vector_search(self, collection_name: str, query: str, 
                      params: Dict) -> List[Dict[str, Any]]:
        """기존 벡터 검색 (InteractManager 활용)"""
        try:
            if not self.interact_manager:
                self.logger.error("InteractManager가 없습니다")
                return []
            
            # 기존 검색 파라미터로 변환
            search_params = {
                "top_k": params.get("top_k", 10),
                "filter_conditions": params.get("filter_conditions", {})
            }
            
            # 기존 검색 실행
            results = self.interact_manager.search(
                collection_name, query, search_params
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"벡터 검색 중 오류: {e}")
            return []
    
    def _format_results(self, raw_results) -> List[Dict[str, Any]]:
        """검색 결과 포맷팅"""
        try:
            formatted_results = []
            
            for hits in raw_results:
                for hit in hits:
                    # 기본 정보 추출
                    result = {
                        "id": getattr(hit, 'id', ''),
                        "score": getattr(hit, 'distance', 0.0),
                        "entity": {}
                    }
                    
                    # entity 정보 추출
                    if hasattr(hit, 'entity') and hit.entity:
                        if isinstance(hit.entity, dict):
                            result["entity"] = hit.entity
                    
                    # 기본 필드들 추출
                    basic_fields = ['passage_uid', 'doc_id', 'title', 'text', 'author', 'domain']
                    for attr in basic_fields:
                        if hasattr(hit, attr):
                            result[attr] = getattr(hit, attr)
                        elif hasattr(hit, 'entity') and isinstance(hit.entity, dict):
                            if attr in hit.entity:
                                result[attr] = hit.entity[attr]
                    
                    # === 위계형 조문 정보 추출 ===
                    article_number = None
                    paragraph_number = None
                    item_number = None
                    
                    if hasattr(hit, 'article_number'):
                        article_number = getattr(hit, 'article_number')
                    elif hasattr(hit, 'entity') and isinstance(hit.entity, dict):
                        article_number = hit.entity.get('article_number')
                    
                    if hasattr(hit, 'paragraph_number'):
                        paragraph_number = getattr(hit, 'paragraph_number')
                    elif hasattr(hit, 'entity') and isinstance(hit.entity, dict):
                        paragraph_number = hit.entity.get('paragraph_number')
                    
                    if hasattr(hit, 'item_number'):
                        item_number = getattr(hit, 'item_number')
                    elif hasattr(hit, 'entity') and isinstance(hit.entity, dict):
                        item_number = hit.entity.get('item_number')
                    
                    # 위계형 정보 구성
                    result["hierarchical_info"] = {
                        "article_number": article_number or "",
                        "paragraph_number": paragraph_number or "",
                        "item_number": item_number or "",
                        "full_reference": self._build_legal_reference(article_number, paragraph_number, item_number)
                    }
                    
                    # 기존 필드와의 호환성을 위해 개별 필드도 유지
                    result["article_number"] = article_number or ""
                    result["paragraph_number"] = paragraph_number or ""
                    result["item_number"] = item_number or ""
                    
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"결과 포맷팅 중 오류: {e}")
            return []
    
    def _build_legal_reference(self, article_number: str, paragraph_number: str, item_number: str) -> str:
        """법령 참조 문자열 구성"""
        reference_parts = []
        
        if article_number:
            reference_parts.append(article_number)
        if paragraph_number:
            reference_parts.append(paragraph_number)
        if item_number:
            reference_parts.append(item_number)
        
        return " ".join(reference_parts) if reference_parts else ""
