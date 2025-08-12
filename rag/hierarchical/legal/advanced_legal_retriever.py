"""
고급 법령 검색 시스템

단순한 유사도 검색을 넘어서 법령의 위계 구조와 법적 논리를 완전히 활용한
지능형 검색 시스템입니다.
"""

from typing import Dict, List, Any, Optional, Set
import logging
import re
from collections import defaultdict
from pymilvus import Collection, utility

from ..base.advanced_retriever import AdvancedHierarchicalRetriever


class AdvancedLegalRetriever(AdvancedHierarchicalRetriever):
    """고급 법령 검색 구현 클래스"""
    
    def __init__(self, existing_interact_manager=None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager 인스턴스
        """
        super().__init__(existing_interact_manager)
        self.logger = logging.getLogger(__name__)
        
        # 법령 특화 검색 설정
        self.legal_hierarchy_weights = {
            "chapter": 1.0,     # 장 레벨
            "section": 1.1,     # 절 레벨  
            "article": 1.3,     # 조 레벨 (가장 중요)
            "paragraph": 1.2,   # 항 레벨
            "item": 1.0,        # 호 레벨
            "subitem": 0.9      # 목 레벨
        }
        
        # 법령 키워드 시소러스
        self.legal_thesaurus = {
            "개인정보": ["개인식별정보", "개인데이터", "신상정보"],
            "수집": ["취득", "획득", "접수", "입수"],
            "동의": ["승낙", "허락", "허가", "동의서"],
            "처리": ["가공", "이용", "활용", "처리행위"],
            "제공": ["전달", "공유", "이전", "송신"],
            "위탁": ["위임", "의뢰", "대행", "위탁처리"],
            "파기": ["삭제", "폐기", "소각", "말소"],
            "정보주체": ["개인", "당사자", "본인", "해당자"],
            "개인정보처리자": ["처리자", "관리자", "취급자", "사업자"]
        }
        
        # 법령 관계 패턴
        self.legal_relation_patterns = {
            "단서조항": r"단서|다만|그러나|다만.{0,50}경우",
            "예외조항": r"예외|제외|적용하지|해당하지.{0,30}않",
            "준용조항": r"준용|적용|이에.{0,20}따라",
            "위임조항": r"위임|위탁|대통령령|부령|시행령",
            "벌칙조항": r"벌금|과태료|과징금|징역|처벌"
        }
    
    # ==================== 구체적인 법령 검색 메서드들 ====================
    
    def _get_nodes_by_id(self, collection_name: str, node_ids: List[str]) -> List[Dict[str, Any]]:
        """노드 ID로 노드들 조회"""
        try:
            if not self.interact_manager or not node_ids:
                return []
            
            # Milvus에서 node_id 기반 조회
            collection = Collection(collection_name)
            
            # node_id 필터 조건 생성
            id_filter = f"node_id in {node_ids}"
            
            # 쿼리 실행
            results = collection.query(
                expr=id_filter,
                output_fields=["*"],
                limit=len(node_ids)
            )
            
            return [dict(result) for result in results]
            
        except Exception as e:
            self.logger.error(f"노드 ID 조회 중 오류: {e}")
            return []
    
    def _get_child_nodes(self, collection_name: str, parent_id: str) -> List[Dict[str, Any]]:
        """자식 노드들 조회"""
        try:
            if not self.interact_manager or not parent_id:
                return []
            
            collection = Collection(collection_name)
            
            # 부모 ID 필터
            parent_filter = f"parent_node_id == '{parent_id}'"
            
            results = collection.query(
                expr=parent_filter,
                output_fields=["*"],
                limit=100  # 충분한 자식 노드 수
            )
            
            return [dict(result) for result in results]
            
        except Exception as e:
            self.logger.error(f"자식 노드 조회 중 오류: {e}")
            return []
    
    def _search_by_article_pattern(self, collection_name: str, pattern: str) -> List[Dict[str, Any]]:
        """조문 패턴으로 검색"""
        try:
            if not self.interact_manager:
                return []
            
            collection = Collection(collection_name)
            
            # 조문 번호 필터 (정확한 매칭)
            filters = []
            
            # "제N조" 패턴 처리
            article_match = re.search(r'제(\d+)조', pattern)
            if article_match:
                article_num = article_match.group(1)
                filters.append(f"article_number == '제{article_num}조'")
            
            # "제N항" 패턴 처리  
            paragraph_match = re.search(r'제?(\d+)항', pattern)
            if paragraph_match:
                para_num = int(paragraph_match.group(1))
                filters.append(f"paragraph_number == {para_num}")
            
            # "제N호" 패턴 처리
            item_match = re.search(r'제?(\d+)호', pattern)
            if item_match:
                item_num = int(item_match.group(1))
                filters.append(f"item_number == {item_num}")
            
            if not filters:
                return []
            
            # 조건 조합
            filter_expr = " and ".join(filters)
            
            results = collection.query(
                expr=filter_expr,
                output_fields=["*"],
                limit=50
            )
            
            return [dict(result) for result in results]
            
        except Exception as e:
            self.logger.error(f"조문 패턴 검색 중 오류: {e}")
            return []
    
    def _search_by_legal_keyword(self, collection_name: str, keyword: str) -> List[Dict[str, Any]]:
        """법령 키워드로 검색"""
        try:
            if not self.interact_manager:
                return []
            
            # 시소러스 확장
            expanded_keywords = self._expand_legal_keyword(keyword)
            
            results = []
            
            for expanded_keyword in expanded_keywords:
                # 기존 시스템의 텍스트 검색 활용
                keyword_results = self.interact_manager.retrieve_data(
                    query=expanded_keyword,
                    top_k=20,
                    filter_conditions={}
                )
                
                # 법령 키워드 필드에서도 검색
                collection = Collection(collection_name)
                
                # JSON 배열 내 키워드 검색
                json_filter = f"JSON_CONTAINS(legal_keywords, '\"{expanded_keyword}\"')"
                
                try:
                    json_results = collection.query(
                        expr=json_filter,
                        output_fields=["*"],
                        limit=20
                    )
                    
                    keyword_results.extend([dict(result) for result in json_results])
                    
                except Exception:
                    # JSON_CONTAINS가 지원되지 않을 경우 스킵
                    pass
                
                results.extend(keyword_results)
            
            return self._deduplicate_results(results)
            
        except Exception as e:
            self.logger.error(f"법령 키워드 검색 중 오류: {e}")
            return []
    
    def _search_by_legal_concept(self, collection_name: str, concept: str) -> List[Dict[str, Any]]:
        """법령 개념으로 검색"""
        try:
            if not self.interact_manager:
                return []
            
            # 개념 기반 검색 전략
            concept_results = []
            
            # 1. 직접 개념 검색
            direct_results = self.interact_manager.retrieve_data(
                query=concept,
                top_k=15,
                filter_conditions={}
            )
            concept_results.extend(direct_results)
            
            # 2. 관련 법령 조항 패턴 검색
            if concept in self.legal_thesaurus:
                for related_term in self.legal_thesaurus[concept]:
                    related_results = self.interact_manager.retrieve_data(
                        query=related_term,
                        top_k=10,
                        filter_conditions={}
                    )
                    concept_results.extend(related_results)
            
            # 3. 법령 관계 패턴 기반 검색
            relation_results = self._search_by_legal_relations(collection_name, concept)
            concept_results.extend(relation_results)
            
            return self._deduplicate_results(concept_results)
            
        except Exception as e:
            self.logger.error(f"법령 개념 검색 중 오류: {e}")
            return []
    
    def _search_by_legal_relations(self, collection_name: str, concept: str) -> List[Dict[str, Any]]:
        """법령 관계 패턴 기반 검색"""
        try:
            relation_results = []
            
            if not self.interact_manager:
                return relation_results
            
            # 각 법령 관계 패턴별로 검색
            for relation_type, pattern in self.legal_relation_patterns.items():
                # 개념 + 관계 패턴 조합 검색
                combined_query = f"{concept} {relation_type}"
                
                pattern_results = self.interact_manager.retrieve_data(
                    query=combined_query,
                    top_k=5,
                    filter_conditions={}
                )
                
                # 관계 타입 정보 추가
                for result in pattern_results:
                    result["detected_relation"] = relation_type
                    result["relation_pattern"] = pattern
                
                relation_results.extend(pattern_results)
            
            return relation_results
            
        except Exception as e:
            self.logger.error(f"법령 관계 검색 중 오류: {e}")
            return []
    
    def _expand_legal_keyword(self, keyword: str) -> List[str]:
        """법령 키워드 시소러스 확장"""
        expanded = [keyword]
        
        # 시소러스에서 확장
        if keyword in self.legal_thesaurus:
            expanded.extend(self.legal_thesaurus[keyword])
        
        # 역방향 검색 (다른 키워드의 동의어로 등록된 경우)
        for main_keyword, synonyms in self.legal_thesaurus.items():
            if keyword in synonyms and main_keyword not in expanded:
                expanded.append(main_keyword)
        
        return expanded
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 중복 제거"""
        try:
            seen_nodes = set()
            unique_results = []
            
            for result in results:
                node_id = result.get("node_id")
                if node_id and node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    unique_results.append(result)
            
            return unique_results
            
        except Exception as e:
            self.logger.error(f"중복 제거 중 오류: {e}")
            return results
    
    # ==================== 법령 특화 고급 검색 메서드들 ====================
    
    def legal_contextual_search(self, collection_name: str, query: str,
                               legal_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """법령 맥락 기반 검색"""
        try:
            self.logger.info(f"🏛️ 법령 맥락 기반 검색: '{query}'")
            
            if not legal_context:
                legal_context = {}
            
            # 맥락 정보를 활용한 필터 구성
            context_filters = {}
            
            # 특정 법령으로 검색 범위 제한
            if legal_context.get("target_law"):
                context_filters["law_number"] = [legal_context["target_law"]]
            
            # 특정 법령 유형으로 제한
            if legal_context.get("law_type"):
                context_filters["law_type"] = [legal_context["law_type"]]
            
            # 특정 도메인으로 제한
            if legal_context.get("domain"):
                context_filters["domain"] = [legal_context["domain"]]
            
            # 위계 레벨 제한
            if legal_context.get("hierarchy_levels"):
                context_filters["hierarchy_level"] = legal_context["hierarchy_levels"]
            
            # 고급 검색 실행
            search_params = {
                "top_k": legal_context.get("top_k", 15),
                "search_mode": "focused",  # 맥락이 있으므로 집중 검색
                "filter_conditions": context_filters,
                "enable_reasoning": True,
                "semantic_expansion": True,
                "explanation_mode": True
            }
            
            results = self.advanced_search(collection_name, query, search_params)
            
            # 법령 맥락 점수 추가
            for result in results:
                context_score = self._calculate_legal_context_score(result, legal_context)
                result["legal_context_score"] = context_score
                result["adjusted_score"] = result.get("final_score", 0) + context_score * 0.2
            
            # 맥락 점수로 재정렬
            results.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
            
            self.logger.info(f"   → {len(results)}개 맥락 기반 결과")
            return results
            
        except Exception as e:
            self.logger.error(f"법령 맥락 검색 중 오류: {e}")
            return []
    
    def legal_cross_reference_search(self, collection_name: str, article_number: str) -> List[Dict[str, Any]]:
        """법령 상호 참조 검색"""
        try:
            self.logger.info(f"🔗 법령 상호 참조 검색: {article_number}")
            
            cross_references = []
            
            # 1. 해당 조문 직접 검색
            target_results = self._search_by_article_pattern(collection_name, article_number)
            
            if not target_results:
                return []
            
            target_article = target_results[0]
            
            # 2. 이 조문을 참조하는 다른 조문들 검색
            reference_query = f"{article_number} 준용 적용 따라"
            referring_results = self.interact_manager.retrieve_data(
                query=reference_query,
                top_k=20,
                filter_conditions={}
            )
            
            for result in referring_results:
                result["reference_type"] = "referring_to_target"
                result["reference_strength"] = 0.8
            
            cross_references.extend(referring_results)
            
            # 3. 이 조문이 참조하는 다른 조문들 검색
            target_content = target_article.get("content", "")
            referenced_patterns = re.findall(r'제\s*(\d+)\s*조', target_content)
            
            for pattern_num in referenced_patterns:
                if pattern_num != article_number.replace("제", "").replace("조", ""):
                    referenced_article = f"제{pattern_num}조"
                    referenced_results = self._search_by_article_pattern(collection_name, referenced_article)
                    
                    for result in referenced_results:
                        result["reference_type"] = "referenced_by_target"
                        result["reference_strength"] = 0.9
                    
                    cross_references.extend(referenced_results)
            
            # 4. 동일 장/절의 관련 조문들
            target_path = target_article.get("hierarchy_path", "")
            if target_path:
                path_parts = target_path.split("/")
                if len(path_parts) >= 2:
                    chapter_section_path = "/".join(path_parts[:2])
                    
                    collection = Collection(collection_name)
                    related_filter = f"hierarchy_path like '{chapter_section_path}%'"
                    
                    try:
                        related_results = collection.query(
                            expr=related_filter,
                            output_fields=["*"],
                            limit=30
                        )
                        
                        for result in related_results:
                            result_dict = dict(result)
                            if result_dict.get("node_id") != target_article.get("node_id"):
                                result_dict["reference_type"] = "structural_relation"
                                result_dict["reference_strength"] = 0.6
                                cross_references.append(result_dict)
                    
                    except Exception:
                        pass
            
            # 중복 제거 및 정렬
            unique_references = self._deduplicate_results(cross_references)
            unique_references.sort(key=lambda x: x.get("reference_strength", 0), reverse=True)
            
            self.logger.info(f"   → {len(unique_references)}개 상호 참조 결과")
            return unique_references
            
        except Exception as e:
            self.logger.error(f"상호 참조 검색 중 오류: {e}")
            return []
    
    def legal_precedent_search(self, collection_name: str, legal_issue: str) -> List[Dict[str, Any]]:
        """법령 선례/판례 기반 검색"""
        try:
            self.logger.info(f"⚖️ 법령 선례 기반 검색: '{legal_issue}'")
            
            precedent_results = []
            
            # 1. 핵심 법적 쟁점 추출
            legal_keywords = self._extract_legal_issues(legal_issue)
            
            # 2. 각 쟁점별 관련 조문 검색
            for keyword in legal_keywords:
                keyword_results = self._search_by_legal_keyword(collection_name, keyword)
                
                for result in keyword_results:
                    result["legal_issue"] = keyword
                    result["precedent_relevance"] = self._calculate_precedent_relevance(result, legal_issue)
                
                precedent_results.extend(keyword_results)
            
            # 3. 유사한 법적 구조 패턴 검색
            structure_results = self._search_by_legal_structure_pattern(collection_name, legal_issue)
            precedent_results.extend(structure_results)
            
            # 4. 선례 관련성 점수로 정렬
            unique_results = self._deduplicate_results(precedent_results)
            unique_results.sort(key=lambda x: x.get("precedent_relevance", 0), reverse=True)
            
            self.logger.info(f"   → {len(unique_results)}개 선례 기반 결과")
            return unique_results
            
        except Exception as e:
            self.logger.error(f"선례 기반 검색 중 오류: {e}")
            return []
    
    # ==================== 보조 메서드들 ====================
    
    def _calculate_legal_context_score(self, result: Dict, legal_context: Dict) -> float:
        """법령 맥락 점수 계산"""
        try:
            score = 0.0
            
            # 법령 타입 일치
            if legal_context.get("law_type") == result.get("law_type"):
                score += 0.3
            
            # 도메인 일치
            if legal_context.get("domain") == result.get("domain"):
                score += 0.2
            
            # 위계 레벨 적합성
            target_levels = legal_context.get("hierarchy_levels", [])
            if target_levels and result.get("hierarchy_level") in target_levels:
                score += 0.2
            
            # 특정 법령 일치
            if legal_context.get("target_law") == result.get("law_number"):
                score += 0.3
            
            return min(score, 1.0)
            
        except Exception:
            return 0.0
    
    def _extract_legal_issues(self, legal_issue: str) -> List[str]:
        """법적 쟁점 추출"""
        issues = []
        
        # 일반적인 법적 쟁점 패턴
        issue_patterns = [
            "권리", "의무", "책임", "손해", "배상", "구제", "제재",
            "허가", "인가", "승인", "신고", "등록", "면허",
            "위반", "처벌", "과태료", "과징금", "벌칙",
            "수집", "이용", "제공", "처리", "보관", "파기",
            "동의", "고지", "통지", "공개", "열람", "정정"
        ]
        
        for pattern in issue_patterns:
            if pattern in legal_issue:
                issues.append(pattern)
        
        return issues
    
    def _calculate_precedent_relevance(self, result: Dict, legal_issue: str) -> float:
        """선례 관련성 점수 계산"""
        try:
            relevance = 0.0
            content = result.get("content", "")
            
            # 내용 유사도
            content_similarity = self._calculate_semantic_similarity(legal_issue, content)
            relevance += content_similarity * 0.6
            
            # 법령 타입 가중치
            law_type = result.get("law_type", "")
            if law_type == "법률":
                relevance += 0.3
            elif law_type == "시행령":
                relevance += 0.2
            elif law_type == "규칙":
                relevance += 0.1
            
            # 위계 레벨 가중치
            hierarchy_level = result.get("hierarchy_level", 0)
            if hierarchy_level == 1:  # 조문 레벨
                relevance += 0.1
            
            return min(relevance, 1.0)
            
        except Exception:
            return 0.0
    
    def _search_by_legal_structure_pattern(self, collection_name: str, legal_issue: str) -> List[Dict[str, Any]]:
        """법령 구조 패턴 기반 검색"""
        try:
            structure_results = []
            
            # 법령 구조 패턴 감지
            detected_patterns = []
            
            for pattern_name, pattern_regex in self.legal_relation_patterns.items():
                if re.search(pattern_regex, legal_issue, re.IGNORECASE):
                    detected_patterns.append(pattern_name)
            
            # 패턴별 검색 실행
            for pattern in detected_patterns:
                pattern_query = f"{legal_issue} {pattern}"
                
                pattern_results = self.interact_manager.retrieve_data(
                    query=pattern_query,
                    top_k=10,
                    filter_conditions={}
                )
                
                for result in pattern_results:
                    result["detected_structure_pattern"] = pattern
                    result["structure_confidence"] = 0.7
                
                structure_results.extend(pattern_results)
            
            return structure_results
            
        except Exception as e:
            self.logger.error(f"구조 패턴 검색 중 오류: {e}")
            return []
