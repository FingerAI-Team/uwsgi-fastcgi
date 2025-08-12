"""
위계 컨텍스트 강화 시스템

검색 결과에 법령의 위계 구조를 활용한 풍부한 컨텍스트를 제공합니다.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque

from ..base.advanced_retriever import AdvancedHierarchicalRetriever


class HierarchicalContextEnhancer:
    """위계형 컨텍스트 강화 시스템"""
    
    def __init__(self, retriever_instance: Optional[AdvancedHierarchicalRetriever] = None):
        """
        Args:
            retriever_instance: 위계형 검색기 인스턴스
        """
        self.retriever = retriever_instance
        self.logger = logging.getLogger(__name__)
        
        # 컨텍스트 설정
        self.context_config = {
            "max_parent_levels": 3,      # 최대 상위 레벨
            "max_child_levels": 2,       # 최대 하위 레벨  
            "max_siblings": 5,           # 최대 형제 노드
            "max_adjacent_articles": 2,   # 최대 인접 조문
            "enable_cross_references": True,
            "enable_related_provisions": True
        }
        
        # 법령 관계 유형
        self.relation_types = {
            "parent": "상위 조항",
            "child": "하위 조항", 
            "sibling": "동일 레벨 조항",
            "adjacent": "인접 조문",
            "reference": "참조 조항",
            "related": "관련 조항"
        }
    
    def enhance_results_with_context(self, results: List[Dict[str, Any]], 
                                   collection_name: str) -> List[Dict[str, Any]]:
        """
        검색 결과에 위계 컨텍스트 추가
        
        Args:
            results: 검색 결과 리스트
            collection_name: 컬렉션 이름
            
        Returns:
            List[Dict]: 컨텍스트가 강화된 결과
        """
        try:
            if not results:
                return results
            
            self.logger.info(f"🌳 위계 컨텍스트 강화 시작: {len(results)}개 결과")
            
            enhanced_results = []
            
            for result in results:
                try:
                    # 개별 결과 컨텍스트 강화
                    enhanced_result = self._enhance_single_result(result, collection_name)
                    enhanced_results.append(enhanced_result)
                    
                except Exception as e:
                    self.logger.error(f"결과 컨텍스트 강화 중 오류: {e}")
                    enhanced_results.append(result)  # 원본 결과 유지
            
            # 결과 간 관계 분석 및 추가
            enhanced_results = self._analyze_inter_result_relations(enhanced_results)
            
            self.logger.info(f"✅ 위계 컨텍스트 강화 완료: {len(enhanced_results)}개 결과")
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"위계 컨텍스트 강화 중 오류: {e}")
            return results
    
    def _enhance_single_result(self, result: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """개별 결과 컨텍스트 강화"""
        try:
            enhanced_result = result.copy()
            
            # 기본 위계 정보
            node_id = result.get("node_id")
            hierarchy_path = result.get("hierarchy_path", "")
            hierarchy_level = result.get("hierarchy_level", 0)
            parent_node_id = result.get("parent_node_id")
            
            if not node_id:
                return enhanced_result
            
            # 위계 컨텍스트 구성
            hierarchy_context = {
                "current_position": {
                    "node_id": node_id,
                    "path": hierarchy_path,
                    "level": hierarchy_level,
                    "type": result.get("node_type", "")
                },
                "parent_context": None,
                "children_context": [],
                "siblings_context": [],
                "adjacent_articles": [],
                "cross_references": []
            }
            
            # 상위 컨텍스트 수집
            if parent_node_id:
                hierarchy_context["parent_context"] = self._get_parent_context(
                    collection_name, parent_node_id, hierarchy_level
                )
            
            # 하위 컨텍스트 수집
            children = self._get_children_context(collection_name, node_id)
            hierarchy_context["children_context"] = children
            
            # 형제 컨텍스트 수집
            if parent_node_id:
                siblings = self._get_siblings_context(collection_name, parent_node_id, node_id)
                hierarchy_context["siblings_context"] = siblings
            
            # 인접 조문 컨텍스트 수집
            adjacent_articles = self._get_adjacent_articles_context(result, collection_name)
            hierarchy_context["adjacent_articles"] = adjacent_articles
            
            # 상호 참조 컨텍스트 수집
            if self.context_config["enable_cross_references"]:
                cross_refs = self._get_cross_references_context(result, collection_name)
                hierarchy_context["cross_references"] = cross_refs
            
            # 컨텍스트 요약 생성
            context_summary = self._generate_context_summary(hierarchy_context)
            
            # 결과에 컨텍스트 정보 추가
            enhanced_result["hierarchy_context"] = hierarchy_context
            enhanced_result["context_summary"] = context_summary
            enhanced_result["enriched_content"] = self._create_enriched_content(result, hierarchy_context)
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"개별 결과 컨텍스트 강화 중 오류: {e}")
            return result
    
    def _get_parent_context(self, collection_name: str, parent_node_id: str, 
                          current_level: int) -> Optional[Dict[str, Any]]:
        """상위 컨텍스트 조회"""
        try:
            if not self.retriever:
                return None
            
            # 상위 노드들을 레벨별로 수집
            parent_chain = []
            current_parent_id = parent_node_id
            level = current_level - 1
            
            for _ in range(self.context_config["max_parent_levels"]):
                if not current_parent_id or level < 0:
                    break
                
                parent_nodes = self.retriever._get_nodes_by_id(collection_name, [current_parent_id])
                if not parent_nodes:
                    break
                
                parent_node = parent_nodes[0]
                parent_info = {
                    "node_id": parent_node.get("node_id"),
                    "title": parent_node.get("title", ""),
                    "content": parent_node.get("content", "")[:200],  # 요약용
                    "level": level,
                    "node_type": parent_node.get("node_type", ""),
                    "hierarchy_path": parent_node.get("hierarchy_path", "")
                }
                parent_chain.append(parent_info)
                
                current_parent_id = parent_node.get("parent_node_id")
                level -= 1
            
            return {
                "chain": parent_chain,
                "immediate_parent": parent_chain[0] if parent_chain else None,
                "root_parent": parent_chain[-1] if parent_chain else None
            }
            
        except Exception as e:
            self.logger.error(f"상위 컨텍스트 조회 중 오류: {e}")
            return None
    
    def _get_children_context(self, collection_name: str, node_id: str) -> List[Dict[str, Any]]:
        """하위 컨텍스트 조회"""
        try:
            if not self.retriever:
                return []
            
            children = self.retriever._get_child_nodes(collection_name, node_id)
            
            children_context = []
            for child in children:
                child_info = {
                    "node_id": child.get("node_id"),
                    "title": child.get("title", ""),
                    "content": child.get("content", "")[:150],
                    "node_type": child.get("node_type", ""),
                    "article_number": child.get("article_number", ""),
                    "paragraph_number": child.get("paragraph_number", 0)
                }
                children_context.append(child_info)
            
            # 최대 개수 제한
            return children_context[:10]
            
        except Exception as e:
            self.logger.error(f"하위 컨텍스트 조회 중 오류: {e}")
            return []
    
    def _get_siblings_context(self, collection_name: str, parent_node_id: str, 
                            current_node_id: str) -> List[Dict[str, Any]]:
        """형제 컨텍스트 조회"""
        try:
            if not self.retriever:
                return []
            
            siblings = self.retriever._get_child_nodes(collection_name, parent_node_id)
            
            siblings_context = []
            for sibling in siblings:
                if sibling.get("node_id") != current_node_id:
                    sibling_info = {
                        "node_id": sibling.get("node_id"),
                        "title": sibling.get("title", ""),
                        "content": sibling.get("content", "")[:100],
                        "node_type": sibling.get("node_type", ""),
                        "article_number": sibling.get("article_number", ""),
                        "paragraph_number": sibling.get("paragraph_number", 0)
                    }
                    siblings_context.append(sibling_info)
            
            # 최대 개수 제한
            return siblings_context[:self.context_config["max_siblings"]]
            
        except Exception as e:
            self.logger.error(f"형제 컨텍스트 조회 중 오류: {e}")
            return []
    
    def _get_adjacent_articles_context(self, result: Dict[str, Any], 
                                     collection_name: str) -> List[Dict[str, Any]]:
        """인접 조문 컨텍스트 조회"""
        try:
            article_number = result.get("article_number", "")
            law_title = result.get("law_title", "")
            
            if not article_number or not self.retriever:
                return []
            
            # 조문 번호에서 숫자 추출
            article_match = re.search(r'제(\d+)조', article_number)
            if not article_match:
                return []
            
            current_num = int(article_match.group(1))
            adjacent_articles = []
            
            # 인접 조문들 (±1, ±2)
            for offset in [-2, -1, 1, 2]:
                adjacent_num = current_num + offset
                if adjacent_num > 0:
                    adjacent_article = f"제{adjacent_num}조"
                    
                    # 해당 조문 검색 (같은 법령 내에서)
                    filter_conditions = {
                        "article_number": [adjacent_article],
                        "law_title": [law_title]
                    }
                    
                    # 실제로는 Milvus 쿼리 필요 (간단히 구현)
                    # adjacent_results = self._search_by_filter(collection_name, filter_conditions)
                    
                    # 임시로 기본 정보만 제공
                    adjacent_info = {
                        "article_number": adjacent_article,
                        "offset": offset,
                        "relation": "인접 조문",
                        "available": False  # 실제 검색 시 업데이트
                    }
                    adjacent_articles.append(adjacent_info)
            
            return adjacent_articles[:self.context_config["max_adjacent_articles"]]
            
        except Exception as e:
            self.logger.error(f"인접 조문 컨텍스트 조회 중 오류: {e}")
            return []
    
    def _get_cross_references_context(self, result: Dict[str, Any], 
                                    collection_name: str) -> List[Dict[str, Any]]:
        """상호 참조 컨텍스트 조회"""
        try:
            content = result.get("content", "")
            cross_references = []
            
            # 내용에서 다른 조문 참조 찾기
            reference_patterns = re.findall(r'제\s*(\d+)\s*조', content)
            
            for ref_num in reference_patterns:
                ref_article = f"제{ref_num}조"
                
                # 자기 자신 제외
                if ref_article != result.get("article_number", ""):
                    cross_ref_info = {
                        "referenced_article": ref_article,
                        "reference_type": "조문 참조",
                        "context": self._extract_reference_context(content, ref_article)
                    }
                    cross_references.append(cross_ref_info)
            
            # 준용, 적용 등의 관계 찾기
            if "준용" in content:
                cross_references.append({
                    "reference_type": "준용 관계",
                    "context": self._extract_around_keyword(content, "준용", 50)
                })
            
            if "적용" in content:
                cross_references.append({
                    "reference_type": "적용 관계", 
                    "context": self._extract_around_keyword(content, "적용", 50)
                })
            
            return cross_references[:5]  # 최대 5개
            
        except Exception as e:
            self.logger.error(f"상호 참조 컨텍스트 조회 중 오류: {e}")
            return []
    
    def _extract_reference_context(self, content: str, reference: str, window: int = 100) -> str:
        """참조 주변 문맥 추출"""
        try:
            index = content.find(reference)
            if index == -1:
                return ""
            
            start = max(0, index - window//2)
            end = min(len(content), index + len(reference) + window//2)
            
            context = content[start:end]
            if start > 0:
                context = "..." + context
            if end < len(content):
                context = context + "..."
            
            return context
            
        except Exception as e:
            return ""
    
    def _extract_around_keyword(self, content: str, keyword: str, window: int = 50) -> str:
        """키워드 주변 문맥 추출"""
        try:
            index = content.find(keyword)
            if index == -1:
                return ""
            
            start = max(0, index - window)
            end = min(len(content), index + len(keyword) + window)
            
            return content[start:end]
            
        except Exception as e:
            return ""
    
    def _analyze_inter_result_relations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 간 관계 분석"""
        try:
            # 결과 간 관계 매트릭스 구성
            for i, result in enumerate(results):
                result["inter_relations"] = []
                
                for j, other_result in enumerate(results):
                    if i != j:
                        relation = self._analyze_two_results_relation(result, other_result)
                        if relation:
                            result["inter_relations"].append({
                                "related_result_index": j,
                                "relation_type": relation["type"],
                                "relation_strength": relation["strength"],
                                "relation_description": relation["description"]
                            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"결과 간 관계 분석 중 오류: {e}")
            return results
    
    def _analyze_two_results_relation(self, result1: Dict, result2: Dict) -> Optional[Dict[str, Any]]:
        """두 결과 간 관계 분석"""
        try:
            # 같은 법령인지 확인
            if result1.get("law_title") != result2.get("law_title"):
                return None
            
            article1 = result1.get("article_number", "")
            article2 = result2.get("article_number", "")
            
            # 같은 조문인지 확인
            if article1 == article2:
                return {
                    "type": "same_article",
                    "strength": 0.9,
                    "description": f"동일 조문 ({article1}) 내 다른 항"
                }
            
            # 인접 조문인지 확인
            if article1 and article2:
                match1 = re.search(r'제(\d+)조', article1)
                match2 = re.search(r'제(\d+)조', article2)
                
                if match1 and match2:
                    num1, num2 = int(match1.group(1)), int(match2.group(1))
                    distance = abs(num1 - num2)
                    
                    if distance == 1:
                        return {
                            "type": "adjacent_article",
                            "strength": 0.7,
                            "description": f"인접 조문 ({article1} ↔ {article2})"
                        }
                    elif distance <= 3:
                        return {
                            "type": "nearby_article", 
                            "strength": 0.5,
                            "description": f"근접 조문 ({article1} ↔ {article2})"
                        }
            
            # 위계 관계 확인
            path1 = result1.get("hierarchy_path", "")
            path2 = result2.get("hierarchy_path", "")
            
            if path1 and path2:
                if path1 in path2:
                    return {
                        "type": "parent_child",
                        "strength": 0.8,
                        "description": "상하위 관계"
                    }
                elif path2 in path1:
                    return {
                        "type": "child_parent",
                        "strength": 0.8,
                        "description": "하상위 관계"
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"두 결과 관계 분석 중 오류: {e}")
            return None
    
    def _generate_context_summary(self, hierarchy_context: Dict[str, Any]) -> Dict[str, str]:
        """컨텍스트 요약 생성"""
        try:
            summary = {
                "position": "",
                "structure": "",
                "relations": "",
                "navigation": ""
            }
            
            current = hierarchy_context["current_position"]
            
            # 현재 위치 요약
            summary["position"] = f"{current['type']} 레벨 {current['level']}: {current['path']}"
            
            # 구조 요약
            parent_count = len(hierarchy_context.get("parent_context", {}).get("chain", []))
            children_count = len(hierarchy_context.get("children_context", []))
            siblings_count = len(hierarchy_context.get("siblings_context", []))
            
            summary["structure"] = f"상위 {parent_count}단계, 하위 {children_count}개, 동급 {siblings_count}개"
            
            # 관계 요약
            relations = []
            if hierarchy_context.get("adjacent_articles"):
                relations.append(f"인접 조문 {len(hierarchy_context['adjacent_articles'])}개")
            if hierarchy_context.get("cross_references"):
                relations.append(f"상호 참조 {len(hierarchy_context['cross_references'])}개")
            
            summary["relations"] = ", ".join(relations) if relations else "관계 정보 없음"
            
            # 탐색 정보
            navigation_hints = []
            if parent_count > 0:
                navigation_hints.append("상위 조항 확인 가능")
            if children_count > 0:
                navigation_hints.append("하위 세부 조항 확인 가능")
            
            summary["navigation"] = ", ".join(navigation_hints) if navigation_hints else ""
            
            return summary
            
        except Exception as e:
            self.logger.error(f"컨텍스트 요약 생성 중 오류: {e}")
            return {"error": "요약 생성 실패"}
    
    def _create_enriched_content(self, result: Dict[str, Any], 
                               hierarchy_context: Dict[str, Any]) -> Dict[str, str]:
        """풍부한 컨텐츠 생성"""
        try:
            enriched = {
                "main_content": result.get("content", ""),
                "contextual_intro": "",
                "related_content": "",
                "navigation_info": ""
            }
            
            # 컨텍스트 도입부
            current_pos = hierarchy_context["current_position"]
            enriched["contextual_intro"] = f"[{current_pos['path']}] {current_pos['type']}"
            
            # 관련 내용
            related_parts = []
            
            # 상위 컨텍스트
            parent_context = hierarchy_context.get("parent_context")
            if parent_context and parent_context.get("immediate_parent"):
                parent = parent_context["immediate_parent"]
                related_parts.append(f"상위 조항: {parent['title']}")
            
            # 하위 컨텍스트
            children = hierarchy_context.get("children_context", [])
            if children:
                child_titles = [child.get("title", "제목 없음") for child in children[:3]]
                related_parts.append(f"하위 조항: {', '.join(child_titles)}")
            
            enriched["related_content"] = " | ".join(related_parts)
            
            # 탐색 정보
            nav_info = []
            if hierarchy_context.get("siblings_context"):
                nav_info.append(f"동급 조항 {len(hierarchy_context['siblings_context'])}개")
            if hierarchy_context.get("adjacent_articles"):
                nav_info.append(f"인접 조문 {len(hierarchy_context['adjacent_articles'])}개")
            
            enriched["navigation_info"] = " • ".join(nav_info)
            
            return enriched
            
        except Exception as e:
            self.logger.error(f"풍부한 컨텐츠 생성 중 오류: {e}")
            return {"main_content": result.get("content", "")}


# 전역 인스턴스
_context_enhancer = None

def get_context_enhancer(retriever_instance=None) -> HierarchicalContextEnhancer:
    """전역 컨텍스트 강화 인스턴스 조회"""
    global _context_enhancer
    if _context_enhancer is None:
        _context_enhancer = HierarchicalContextEnhancer(retriever_instance)
    return _context_enhancer
