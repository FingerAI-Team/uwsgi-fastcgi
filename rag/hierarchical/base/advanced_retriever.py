"""
고급 위계형 검색 시스템

단순한 유사도 검색이 아닌, 위계형 데이터의 특성을 완전히 활용한 
지능형 검색 엔진입니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
import time
import numpy as np
from collections import defaultdict, deque
from pymilvus import Collection


class AdvancedHierarchicalRetriever(ABC):
    """고급 위계형 검색 베이스 클래스"""
    
    def __init__(self, existing_interact_manager=None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager 인스턴스
        """
        self.interact_manager = existing_interact_manager
        self.logger = logging.getLogger(__name__)
        
        # 고급 검색 설정
        self.search_strategies = {
            "direct_matching": 0.4,      # 직접 매칭 가중치
            "hierarchy_reasoning": 0.3,   # 위계 추론 가중치  
            "semantic_expansion": 0.2,    # 의미적 확장 가중치
            "structural_traversal": 0.1   # 구조적 탐색 가중치
        }
        
        # 위계형 탐색 설정
        self.max_traversal_depth = 5
        self.semantic_similarity_threshold = 0.7
        self.hierarchy_boost_factor = 1.5
        
    def advanced_search(self, collection_name: str, query: str, 
                       search_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        고급 위계형 검색 수행
        
        검색 전략:
        1. 📍 직접 매칭: 쿼리와 직접 관련된 노드들 
        2. 🧠 위계 추론: 부모-자식 관계를 통한 추론적 검색
        3. 🌐 의미적 확장: 유사한 개념 및 관련 법령 조항들
        4. 🌳 구조적 탐색: 법령 구조를 따라가며 연관 조문 발견
        """
        try:
            start_time = time.time()
            
            if not search_params:
                search_params = {}
            
            # 고급 검색 파라미터
            params = {
                "top_k": search_params.get("top_k", 15),
                "search_mode": search_params.get("search_mode", "comprehensive"), # comprehensive/focused/exploratory
                "enable_reasoning": search_params.get("enable_reasoning", True),
                "semantic_expansion": search_params.get("semantic_expansion", True),
                "structural_traversal": search_params.get("structural_traversal", True),
                "hierarchy_boost": search_params.get("hierarchy_boost", True),
                "diversity_factor": search_params.get("diversity_factor", 0.3),
                "filter_conditions": search_params.get("filter_conditions", {}),
                "explanation_mode": search_params.get("explanation_mode", False)
            }
            
            self.logger.info(f"🚀 고급 위계형 검색 시작: '{query}' (모드: {params['search_mode']})")
            
            # 🔍 1. 다중 전략 병렬 검색
            search_results = self._execute_multi_strategy_search(collection_name, query, params)
            
            # 🧠 2. 위계형 추론 실행
            if params["enable_reasoning"]:
                reasoning_results = self._hierarchical_reasoning_search(collection_name, query, search_results, params)
                search_results["reasoning"] = reasoning_results
            
            # 🌐 3. 의미적 확장 검색
            if params["semantic_expansion"]:
                expansion_results = self._semantic_expansion_search(collection_name, query, search_results, params)
                search_results["expansion"] = expansion_results
            
            # 🌳 4. 구조적 탐색 실행
            if params["structural_traversal"]:
                traversal_results = self._structural_traversal_search(collection_name, search_results, params)
                search_results["traversal"] = traversal_results
            
            # 📊 5. 결과 융합 및 스코어링
            fused_results = self._fuse_multi_strategy_results(search_results, params)
            
            # 🎯 6. 위계형 다양성 확보
            diverse_results = self._ensure_hierarchical_diversity(fused_results, params)
            
            # 📈 7. 최종 랭킹 및 설명 생성
            final_results = self._final_ranking_with_explanation(diverse_results, query, params)
            
            end_time = time.time()
            self.logger.info(f"✅ 고급 위계형 검색 완료: {len(final_results)}개 결과, {end_time - start_time:.3f}초")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"고급 위계형 검색 중 오류: {e}")
            return []
    
    def _execute_multi_strategy_search(self, collection_name: str, query: str, 
                                     params: Dict) -> Dict[str, List[Dict]]:
        """다중 검색 전략 병렬 실행"""
        try:
            results = {}
            
            # 📍 1. 직접 매칭 검색
            self.logger.info("📍 직접 매칭 검색 실행...")
            direct_results = self._direct_matching_search(collection_name, query, params)
            results["direct"] = direct_results
            self.logger.info(f"   → {len(direct_results)}개 직접 매칭 결과")
            
            # 🎯 2. 조문 번호 및 키워드 기반 검색
            keyword_results = self._keyword_based_search(collection_name, query, params)
            results["keyword"] = keyword_results
            self.logger.info(f"   → {len(keyword_results)}개 키워드 매칭 결과")
            
            # 📊 3. 메타데이터 필터링 검색
            metadata_results = self._metadata_filtered_search(collection_name, query, params)
            results["metadata"] = metadata_results
            self.logger.info(f"   → {len(metadata_results)}개 메타데이터 매칭 결과")
            
            return results
            
        except Exception as e:
            self.logger.error(f"다중 전략 검색 중 오류: {e}")
            return {}
    
    def _direct_matching_search(self, collection_name: str, query: str, 
                              params: Dict) -> List[Dict[str, Any]]:
        """직접 매칭 검색 (기존 벡터 검색 활용)"""
        try:
            if not self.interact_manager:
                return []
            
            # 기존 retrieve_data 활용하되 더 많은 결과 요청
            results = self.interact_manager.retrieve_data(
                query=query,
                top_k=params["top_k"] * 3,  # 다양성을 위해 3배 더 검색
                filter_conditions=params["filter_conditions"]
            )
            
            # 각 결과에 전략 태그 추가
            for result in results:
                result["search_strategy"] = "direct_matching"
                result["strategy_confidence"] = result.get("score", 0.0)
            
            return results
            
        except Exception as e:
            self.logger.error(f"직접 매칭 검색 중 오류: {e}")
            return []
    
    def _keyword_based_search(self, collection_name: str, query: str, 
                            params: Dict) -> List[Dict[str, Any]]:
        """키워드 및 조문 번호 기반 검색"""
        try:
            if not self.interact_manager or not hasattr(self.interact_manager, 'vectorenv'):
                return []
            
            results = []
            
            # 🔤 1. 조문 번호 패턴 검색
            article_patterns = self._extract_article_patterns(query)
            if article_patterns:
                for pattern in article_patterns:
                    article_results = self._search_by_article_pattern(collection_name, pattern)
                    for result in article_results:
                        result["search_strategy"] = "article_matching"
                        result["strategy_confidence"] = 0.9  # 조문 매칭은 높은 신뢰도
                        result["matched_pattern"] = pattern
                    results.extend(article_results)
            
            # 🏷️ 2. 법령 키워드 검색
            legal_keywords = self._extract_legal_keywords(query)
            if legal_keywords:
                for keyword in legal_keywords:
                    keyword_results = self._search_by_legal_keyword(collection_name, keyword)
                    for result in keyword_results:
                        result["search_strategy"] = "keyword_matching"
                        result["strategy_confidence"] = 0.7
                        result["matched_keyword"] = keyword
                    results.extend(keyword_results)
            
            return self._deduplicate_results(results)
            
        except Exception as e:
            self.logger.error(f"키워드 기반 검색 중 오류: {e}")
            return []
    
    def _metadata_filtered_search(self, collection_name: str, query: str, 
                                params: Dict) -> List[Dict[str, Any]]:
        """메타데이터 필터링 기반 검색"""
        try:
            if not self.interact_manager:
                return []
            
            # 쿼리에서 법령 타입, 도메인 등 추출
            metadata_filters = self._extract_metadata_hints(query)
            
            if not metadata_filters:
                return []
            
            # 메타데이터 조건으로 검색
            enhanced_filters = {**params["filter_conditions"], **metadata_filters}
            
            results = self.interact_manager.retrieve_data(
                query=query,
                top_k=params["top_k"] * 2,
                filter_conditions=enhanced_filters
            )
            
            for result in results:
                result["search_strategy"] = "metadata_filtering"
                result["strategy_confidence"] = 0.6
                result["applied_filters"] = metadata_filters
            
            return results
            
        except Exception as e:
            self.logger.error(f"메타데이터 필터링 검색 중 오류: {e}")
            return []
    
    def _hierarchical_reasoning_search(self, collection_name: str, query: str,
                                     base_results: Dict, params: Dict) -> List[Dict[str, Any]]:
        """위계형 추론 검색"""
        try:
            self.logger.info("🧠 위계형 추론 검색 실행...")
            
            reasoning_results = []
            processed_nodes = set()
            
            # 기존 검색 결과에서 위계 관계 분석
            for strategy_name, results in base_results.items():
                for result in results:
                    node_id = result.get("node_id")
                    if not node_id or node_id in processed_nodes:
                        continue
                    
                    processed_nodes.add(node_id)
                    
                    # 🔼 상위 개념 추론
                    parent_reasoning = self._reason_from_parents(collection_name, result, query)
                    reasoning_results.extend(parent_reasoning)
                    
                    # 🔽 하위 개념 추론  
                    child_reasoning = self._reason_from_children(collection_name, result, query)
                    reasoning_results.extend(child_reasoning)
                    
                    # ↔️ 동등 개념 추론
                    sibling_reasoning = self._reason_from_siblings(collection_name, result, query)
                    reasoning_results.extend(sibling_reasoning)
            
            # 추론 결과 정리
            unique_results = self._deduplicate_results(reasoning_results)
            
            self.logger.info(f"   → {len(unique_results)}개 추론 결과 발견")
            return unique_results
            
        except Exception as e:
            self.logger.error(f"위계형 추론 검색 중 오류: {e}")
            return []
    
    def _reason_from_parents(self, collection_name: str, node: Dict, query: str) -> List[Dict[str, Any]]:
        """부모 노드로부터의 추론"""
        try:
            results = []
            parent_id = node.get("parent_node_id")
            
            if not parent_id:
                return results
            
            # 부모 노드 조회
            parent_nodes = self._get_nodes_by_id(collection_name, [parent_id])
            
            for parent in parent_nodes:
                # 부모 노드의 다른 자식들이 쿼리와 관련될 수 있음
                siblings = self._get_child_nodes(collection_name, parent_id)
                
                for sibling in siblings:
                    if sibling.get("node_id") != node.get("node_id"):
                        # 의미적 유사도 확인
                        similarity = self._calculate_semantic_similarity(query, sibling.get("content", ""))
                        
                        if similarity > self.semantic_similarity_threshold:
                            sibling["search_strategy"] = "parent_reasoning"
                            sibling["strategy_confidence"] = similarity * 0.8  # 추론이므로 약간 낮춤
                            sibling["reasoning_path"] = f"부모({parent_id})를 통한 추론"
                            results.append(sibling)
            
            return results
            
        except Exception as e:
            self.logger.error(f"부모 추론 중 오류: {e}")
            return []
    
    def _reason_from_children(self, collection_name: str, node: Dict, query: str) -> List[Dict[str, Any]]:
        """자식 노드로부터의 추론"""
        try:
            results = []
            node_id = node.get("node_id")
            
            # 자식 노드들 조회
            children = self._get_child_nodes(collection_name, node_id)
            
            for child in children:
                # 자식이 쿼리와 관련 있으면 부모의 다른 자식들도 관련될 수 있음
                similarity = self._calculate_semantic_similarity(query, child.get("content", ""))
                
                if similarity > self.semantic_similarity_threshold:
                    # 이 자식과 같은 레벨의 다른 자식들 검색
                    child_parent = child.get("parent_node_id")
                    if child_parent:
                        other_children = self._get_child_nodes(collection_name, child_parent)
                        
                        for other_child in other_children:
                            if other_child.get("node_id") != child.get("node_id"):
                                other_child["search_strategy"] = "child_reasoning"
                                other_child["strategy_confidence"] = similarity * 0.7
                                other_child["reasoning_path"] = f"자식({child.get('node_id')})을 통한 추론"
                                results.append(other_child)
            
            return results
            
        except Exception as e:
            self.logger.error(f"자식 추론 중 오류: {e}")
            return []
    
    def _reason_from_siblings(self, collection_name: str, node: Dict, query: str) -> List[Dict[str, Any]]:
        """형제 노드로부터의 추론"""
        try:
            results = []
            parent_id = node.get("parent_node_id")
            
            if not parent_id:
                return results
            
            # 형제 노드들 조회
            siblings = self._get_child_nodes(collection_name, parent_id)
            
            for sibling in siblings:
                if sibling.get("node_id") != node.get("node_id"):
                    # 형제 노드와 쿼리의 의미적 관련성 확인
                    similarity = self._calculate_semantic_similarity(query, sibling.get("content", ""))
                    
                    if similarity > self.semantic_similarity_threshold:
                        sibling["search_strategy"] = "sibling_reasoning"
                        sibling["strategy_confidence"] = similarity * 0.75
                        sibling["reasoning_path"] = f"형제({node.get('node_id')})를 통한 추론"
                        results.append(sibling)
            
            return results
            
        except Exception as e:
            self.logger.error(f"형제 추론 중 오류: {e}")
            return []
    
    def _semantic_expansion_search(self, collection_name: str, query: str,
                                 base_results: Dict, params: Dict) -> List[Dict[str, Any]]:
        """의미적 확장 검색"""
        try:
            self.logger.info("🌐 의미적 확장 검색 실행...")
            
            expansion_results = []
            
            # 🔄 1. 쿼리 의미 확장
            expanded_queries = self._expand_query_semantically(query)
            
            for expanded_query in expanded_queries:
                if expanded_query != query:  # 원본 쿼리 제외
                    expanded_results = self.interact_manager.retrieve_data(
                        query=expanded_query,
                        top_k=params["top_k"],
                        filter_conditions=params["filter_conditions"]
                    )
                    
                    for result in expanded_results:
                        result["search_strategy"] = "semantic_expansion"
                        result["strategy_confidence"] = result.get("score", 0.0) * 0.8
                        result["expanded_query"] = expanded_query
                    
                    expansion_results.extend(expanded_results)
            
            # 🏷️ 2. 법령 개념 확장
            legal_concepts = self._extract_legal_concepts(query)
            for concept in legal_concepts:
                related_concepts = self._get_related_legal_concepts(concept)
                
                for related_concept in related_concepts:
                    concept_results = self._search_by_legal_concept(collection_name, related_concept)
                    
                    for result in concept_results:
                        result["search_strategy"] = "concept_expansion"
                        result["strategy_confidence"] = 0.6
                        result["original_concept"] = concept
                        result["expanded_concept"] = related_concept
                    
                    expansion_results.extend(concept_results)
            
            unique_results = self._deduplicate_results(expansion_results)
            self.logger.info(f"   → {len(unique_results)}개 의미 확장 결과")
            
            return unique_results
            
        except Exception as e:
            self.logger.error(f"의미적 확장 검색 중 오류: {e}")
            return []
    
    def _structural_traversal_search(self, collection_name: str, 
                                   base_results: Dict, params: Dict) -> List[Dict[str, Any]]:
        """구조적 탐색 검색"""
        try:
            self.logger.info("🌳 구조적 탐색 검색 실행...")
            
            traversal_results = []
            visited_nodes = set()
            
            # 기존 결과에서 시작하여 구조적 탐색
            for strategy_name, results in base_results.items():
                for result in results:
                    if result.get("node_id") in visited_nodes:
                        continue
                    
                    # 🌊 BFS 방식으로 위계 구조 탐색
                    traversal_nodes = self._bfs_hierarchy_traversal(
                        collection_name, 
                        result.get("node_id"), 
                        max_depth=self.max_traversal_depth
                    )
                    
                    for node in traversal_nodes:
                        if node.get("node_id") not in visited_nodes:
                            node["search_strategy"] = "structural_traversal"
                            node["strategy_confidence"] = max(0.3, 1.0 / (node.get("traversal_distance", 1) + 1))
                            node["traversal_origin"] = result.get("node_id")
                            
                            traversal_results.append(node)
                            visited_nodes.add(node.get("node_id"))
            
            self.logger.info(f"   → {len(traversal_results)}개 구조 탐색 결과")
            return traversal_results
            
        except Exception as e:
            self.logger.error(f"구조적 탐색 검색 중 오류: {e}")
            return []
    
    def _bfs_hierarchy_traversal(self, collection_name: str, start_node_id: str, 
                               max_depth: int = 3) -> List[Dict[str, Any]]:
        """BFS 방식 위계 구조 탐색"""
        try:
            results = []
            visited = set([start_node_id])
            queue = deque([(start_node_id, 0)])  # (node_id, depth)
            
            while queue and len(results) < 50:  # 최대 50개로 제한
                current_node_id, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                # 인접 노드들 (부모, 자식, 형제) 수집
                adjacent_nodes = []
                
                # 현재 노드 정보 조회
                current_nodes = self._get_nodes_by_id(collection_name, [current_node_id])
                if not current_nodes:
                    continue
                
                current_node = current_nodes[0]
                
                # 부모 노드
                parent_id = current_node.get("parent_node_id")
                if parent_id and parent_id not in visited:
                    parent_nodes = self._get_nodes_by_id(collection_name, [parent_id])
                    for parent in parent_nodes:
                        parent["traversal_distance"] = depth + 1
                        parent["traversal_relation"] = "parent"
                        adjacent_nodes.append(parent)
                        queue.append((parent_id, depth + 1))
                        visited.add(parent_id)
                
                # 자식 노드들
                children = self._get_child_nodes(collection_name, current_node_id)
                for child in children:
                    child_id = child.get("node_id")
                    if child_id and child_id not in visited:
                        child["traversal_distance"] = depth + 1
                        child["traversal_relation"] = "child"
                        adjacent_nodes.append(child)
                        queue.append((child_id, depth + 1))
                        visited.add(child_id)
                
                # 형제 노드들
                if parent_id:
                    siblings = self._get_child_nodes(collection_name, parent_id)
                    for sibling in siblings:
                        sibling_id = sibling.get("node_id")
                        if sibling_id and sibling_id not in visited and sibling_id != current_node_id:
                            sibling["traversal_distance"] = depth + 1
                            sibling["traversal_relation"] = "sibling"
                            adjacent_nodes.append(sibling)
                            queue.append((sibling_id, depth + 1))
                            visited.add(sibling_id)
                
                results.extend(adjacent_nodes)
            
            return results
            
        except Exception as e:
            self.logger.error(f"BFS 탐색 중 오류: {e}")
            return []
    
    def _fuse_multi_strategy_results(self, all_results: Dict, params: Dict) -> List[Dict[str, Any]]:
        """다중 전략 결과 융합"""
        try:
            self.logger.info("📊 다중 전략 결과 융합 중...")
            
            # 모든 결과를 하나로 합치기
            fused_results = []
            node_scores = defaultdict(list)  # node_id -> [(strategy, confidence, result)]
            
            for strategy_name, results in all_results.items():
                strategy_weight = self.search_strategies.get(strategy_name, 0.1)
                
                for result in results:
                    node_id = result.get("node_id")
                    if node_id:
                        confidence = result.get("strategy_confidence", 0.5)
                        weighted_score = confidence * strategy_weight
                        
                        node_scores[node_id].append((strategy_name, weighted_score, result))
            
            # 노드별 종합 점수 계산
            for node_id, score_list in node_scores.items():
                if not score_list:
                    continue
                
                # 최고 점수 결과를 대표로 선택
                best_strategy, best_score, best_result = max(score_list, key=lambda x: x[1])
                
                # 다중 전략에서 발견된 경우 보너스
                strategy_count = len(set(item[0] for item in score_list))
                diversity_bonus = min(0.2, strategy_count * 0.05)
                
                # 종합 점수 계산
                total_score = sum(item[1] for item in score_list) + diversity_bonus
                
                # 결과에 메타정보 추가
                best_result["final_score"] = total_score
                best_result["primary_strategy"] = best_strategy
                best_result["supporting_strategies"] = [item[0] for item in score_list if item[0] != best_strategy]
                best_result["strategy_diversity"] = strategy_count
                
                fused_results.append(best_result)
            
            # 점수순 정렬
            fused_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            
            self.logger.info(f"   → {len(fused_results)}개 노드로 융합 완료")
            return fused_results
            
        except Exception as e:
            self.logger.error(f"결과 융합 중 오류: {e}")
            return []
    
    def _ensure_hierarchical_diversity(self, results: List[Dict], params: Dict) -> List[Dict[str, Any]]:
        """위계형 다양성 확보"""
        try:
            diversity_factor = params.get("diversity_factor", 0.3)
            
            if diversity_factor <= 0:
                return results[:params["top_k"]]
            
            self.logger.info("🎯 위계형 다양성 확보 중...")
            
            diverse_results = []
            used_hierarchy_paths = set()
            level_counts = defaultdict(int)
            
            # 1차: 고득점 + 다양성 균형
            for result in results:
                hierarchy_path = result.get("hierarchy_path", "")
                hierarchy_level = result.get("hierarchy_level", 0)
                
                # 경로 중복 확인
                path_similarity = self._calculate_path_similarity(hierarchy_path, used_hierarchy_paths)
                level_penalty = level_counts[hierarchy_level] * 0.1
                
                # 다양성 점수 계산
                diversity_score = 1.0 - path_similarity - level_penalty
                
                # 최종 점수 = 원본 점수 + 다양성 보너스
                final_score = result.get("final_score", 0) + (diversity_score * diversity_factor)
                result["diversity_adjusted_score"] = final_score
                
                diverse_results.append(result)
                used_hierarchy_paths.add(hierarchy_path)
                level_counts[hierarchy_level] += 1
            
            # 다양성 조정 점수로 재정렬
            diverse_results.sort(key=lambda x: x.get("diversity_adjusted_score", 0), reverse=True)
            
            final_count = min(params["top_k"], len(diverse_results))
            self.logger.info(f"   → {final_count}개 다양성 확보 결과")
            
            return diverse_results[:final_count]
            
        except Exception as e:
            self.logger.error(f"다양성 확보 중 오류: {e}")
            return results[:params["top_k"]]
    
    def _final_ranking_with_explanation(self, results: List[Dict], query: str, 
                                      params: Dict) -> List[Dict[str, Any]]:
        """최종 랭킹 및 설명 생성"""
        try:
            self.logger.info("📈 최종 랭킹 및 설명 생성 중...")
            
            final_results = []
            
            for i, result in enumerate(results):
                # 순위 정보 추가
                result["rank"] = i + 1
                result["total_candidates"] = len(results)
                
                # 검색 설명 생성
                if params.get("explanation_mode", False):
                    explanation = self._generate_search_explanation(result, query)
                    result["search_explanation"] = explanation
                
                # 위계 컨텍스트 요약
                hierarchy_context = self._generate_hierarchy_context_summary(result)
                result["hierarchy_context"] = hierarchy_context
                
                final_results.append(result)
            
            self.logger.info(f"✅ 최종 {len(final_results)}개 결과 랭킹 완료")
            return final_results
            
        except Exception as e:
            self.logger.error(f"최종 랭킹 중 오류: {e}")
            return results
    
    # ==================== 헬퍼 메서드들 ====================
    
    def _extract_article_patterns(self, query: str) -> List[str]:
        """조문 번호 패턴 추출"""
        import re
        patterns = []
        
        # "제N조", "제N항", "제N호" 등의 패턴
        article_patterns = re.findall(r'제\s*(\d+)\s*조', query)
        paragraph_patterns = re.findall(r'제?\s*(\d+)\s*항', query) 
        item_patterns = re.findall(r'제?\s*(\d+)\s*호', query)
        
        patterns.extend([f"제{num}조" for num in article_patterns])
        patterns.extend([f"제{num}항" for num in paragraph_patterns])
        patterns.extend([f"제{num}호" for num in item_patterns])
        
        return patterns
    
    def _extract_legal_keywords(self, query: str) -> List[str]:
        """법령 전문 키워드 추출"""
        legal_keywords = []
        
        # 일반적인 법령 키워드들
        common_legal_terms = [
            "개인정보", "처리", "수집", "이용", "제공", "동의", "정보주체",
            "처리목적", "보유기간", "파기", "위탁", "제3자", "국외이전",
            "권리", "의무", "손해배상", "과태료", "과징금", "벌칙"
        ]
        
        for term in common_legal_terms:
            if term in query:
                legal_keywords.append(term)
        
        return legal_keywords
    
    def _extract_metadata_hints(self, query: str) -> Dict[str, Any]:
        """쿼리에서 메타데이터 힌트 추출"""
        metadata_hints = {}
        
        # 법령 유형 힌트
        if "법률" in query or "법" in query:
            metadata_hints["law_type"] = ["법률"]
        elif "시행령" in query or "령" in query:
            metadata_hints["law_type"] = ["시행령"]
        elif "규칙" in query:
            metadata_hints["law_type"] = ["규칙"]
        
        # 도메인 힌트
        if "개인정보" in query:
            metadata_hints["domain"] = ["개인정보보호"]
        elif "정보통신" in query:
            metadata_hints["domain"] = ["정보통신"]
        elif "데이터" in query:
            metadata_hints["domain"] = ["데이터"]
        
        return metadata_hints
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """간단한 의미적 유사도 계산 (실제로는 임베딩 기반으로 구현)"""
        if not text1 or not text2:
            return 0.0
        
        # 간단한 Jaccard 유사도로 근사
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _expand_query_semantically(self, query: str) -> List[str]:
        """쿼리 의미적 확장"""
        expanded_queries = [query]
        
        # 간단한 동의어 확장 (실제로는 더 정교한 시스템 필요)
        synonyms = {
            "수집": ["취득", "획득", "접수"],
            "동의": ["승낙", "허락", "허가"],
            "개인정보": ["개인식별정보", "개인데이터"],
            "처리": ["가공", "이용", "활용"]
        }
        
        for original, synonym_list in synonyms.items():
            if original in query:
                for synonym in synonym_list:
                    expanded_query = query.replace(original, synonym)
                    if expanded_query != query:
                        expanded_queries.append(expanded_query)
        
        return expanded_queries
    
    def _extract_legal_concepts(self, query: str) -> List[str]:
        """법령 개념 추출"""
        concepts = []
        
        # 주요 법령 개념들
        legal_concepts = [
            "개인정보처리자", "정보주체", "법정대리인", "개인정보보호책임자",
            "처리목적", "보유기간", "안전성확보조치", "개인정보영향평가"
        ]
        
        for concept in legal_concepts:
            if concept in query:
                concepts.append(concept)
        
        return concepts
    
    def _get_related_legal_concepts(self, concept: str) -> List[str]:
        """관련 법령 개념 조회"""
        concept_relations = {
            "개인정보처리자": ["정보주체", "개인정보보호책임자", "수탁자"],
            "정보주체": ["개인정보처리자", "법정대리인", "대리인"],
            "수집": ["이용", "제공", "위탁", "보관"],
            "동의": ["철회", "거부", "선택"]
        }
        
        return concept_relations.get(concept, [])
    
    def _calculate_path_similarity(self, path: str, used_paths: Set[str]) -> float:
        """경로 유사도 계산"""
        if not used_paths:
            return 0.0
        
        max_similarity = 0.0
        path_parts = path.split('/')
        
        for used_path in used_paths:
            used_parts = used_path.split('/')
            
            # 공통 부분 계산
            common_parts = 0
            for i in range(min(len(path_parts), len(used_parts))):
                if path_parts[i] == used_parts[i]:
                    common_parts += 1
                else:
                    break
            
            similarity = common_parts / max(len(path_parts), len(used_parts))
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _generate_search_explanation(self, result: Dict, query: str) -> Dict[str, str]:
        """검색 결과 설명 생성"""
        explanation = {
            "primary_reason": "",
            "supporting_evidence": [],
            "hierarchy_relevance": "",
            "confidence_level": ""
        }
        
        primary_strategy = result.get("primary_strategy", "unknown")
        confidence = result.get("final_score", 0.0)
        
        # 주요 검색 이유
        if primary_strategy == "direct_matching":
            explanation["primary_reason"] = "쿼리와 직접적인 의미 유사성이 높음"
        elif primary_strategy == "article_matching":
            explanation["primary_reason"] = f"조문 번호 '{result.get('matched_pattern', '')}' 직접 매칭"
        elif primary_strategy == "keyword_matching":
            explanation["primary_reason"] = f"핵심 키워드 '{result.get('matched_keyword', '')}' 매칭"
        elif primary_strategy == "parent_reasoning":
            explanation["primary_reason"] = "상위 조항과의 관계를 통한 추론적 매칭"
        elif primary_strategy == "sibling_reasoning":
            explanation["primary_reason"] = "동일 레벨 조항과의 관계를 통한 추론적 매칭"
        
        # 신뢰도 레벨
        if confidence >= 0.8:
            explanation["confidence_level"] = "매우 높음"
        elif confidence >= 0.6:
            explanation["confidence_level"] = "높음"
        elif confidence >= 0.4:
            explanation["confidence_level"] = "보통"
        else:
            explanation["confidence_level"] = "낮음"
        
        # 위계 관련성
        hierarchy_path = result.get("hierarchy_path", "")
        if hierarchy_path:
            explanation["hierarchy_relevance"] = f"법령 구조상 위치: {hierarchy_path}"
        
        return explanation
    
    def _generate_hierarchy_context_summary(self, result: Dict) -> Dict[str, Any]:
        """위계 컨텍스트 요약 생성"""
        context = {
            "current_level": result.get("hierarchy_level", 0),
            "full_path": result.get("hierarchy_path", ""),
            "node_type": result.get("node_type", "unknown"),
            "has_children": result.get("child_count", 0) > 0,
            "has_siblings": False,  # 실제로는 DB 조회 필요
            "legal_significance": ""
        }
        
        # 법적 중요도 판단
        node_type = result.get("node_type", "")
        if node_type == "article":
            context["legal_significance"] = "조문 레벨 (핵심 법령 조항)"
        elif node_type == "paragraph":
            context["legal_significance"] = "항 레벨 (구체적 법령 내용)"
        elif node_type == "chapter":
            context["legal_significance"] = "장 레벨 (법령 대분류)"
        else:
            context["legal_significance"] = "기타 레벨"
        
        return context
    
    # ==================== 추상 메서드들 (서브클래스에서 구현) ====================
    
    @abstractmethod
    def _get_nodes_by_id(self, collection_name: str, node_ids: List[str]) -> List[Dict[str, Any]]:
        """노드 ID로 노드들 조회"""
        pass
    
    @abstractmethod  
    def _get_child_nodes(self, collection_name: str, parent_id: str) -> List[Dict[str, Any]]:
        """자식 노드들 조회"""
        pass
    
    @abstractmethod
    def _search_by_article_pattern(self, collection_name: str, pattern: str) -> List[Dict[str, Any]]:
        """조문 패턴으로 검색"""
        pass
    
    @abstractmethod
    def _search_by_legal_keyword(self, collection_name: str, keyword: str) -> List[Dict[str, Any]]:
        """법령 키워드로 검색"""
        pass
    
    @abstractmethod
    def _search_by_legal_concept(self, collection_name: str, concept: str) -> List[Dict[str, Any]]:
        """법령 개념으로 검색"""
        pass
    
    @abstractmethod
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 중복 제거"""
        pass
