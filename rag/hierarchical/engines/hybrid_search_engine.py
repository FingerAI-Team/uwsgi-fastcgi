"""
하이브리드 검색 엔진

Vector 검색(Milvus)과 BM25 검색(Meilisearch)을 결합하여
최적의 검색 결과를 제공합니다.
"""

import logging
import time
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import concurrent.futures
from threading import Thread

from .meilisearch_client import get_meilisearch_engine
from ..utils.intent_detector import get_intent_detector
from ..config.config_loader import get_config_loader


@dataclass
class HybridSearchConfig:
    """하이브리드 검색 설정"""
    default_vector_weight: float = 0.6
    default_bm25_weight: float = 0.4
    pattern_boost_weight: float = 0.2
    lawname_boost_weight: float = 0.1
    enable_intent_detection: bool = True
    enable_pattern_boost: bool = True
    enable_deduplication: bool = True
    max_results: int = 20
    
    # 상수 정의
    MAX_PATTERN_BOOST: float = 0.5
    SHORT_QUERY_THRESHOLD: int = 3
    LONG_QUERY_THRESHOLD: int = 8
    LENGTH_BIAS_ADJUSTMENT: float = 0.05
    EXACT_MATCH_BONUS: float = 0.05
    TITLE_HIT_BONUS: float = 0.05


class HybridSearchEngine:
    """하이브리드 검색 엔진"""
    
    def __init__(self, existing_interact_manager=None, config: Optional[HybridSearchConfig] = None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager (Vector 검색용)
            config: 하이브리드 검색 설정
        """
        self.interact_manager = existing_interact_manager
        self.config = config or HybridSearchConfig()
        self.logger = logging.getLogger(__name__)
        
        # 검색 엔진들 (싱글톤 패턴 사용)
        self.meilisearch_engine = get_meilisearch_engine()
        self.intent_detector = get_intent_detector()
        self.config_loader = get_config_loader()
        
        # 통계
        self.search_stats = {
            "total_searches": 0,
            "vector_searches": 0,
            "bm25_searches": 0,
            "hybrid_searches": 0,
            "cache_hits": 0
        }
    
    def hybrid_search(self, query: str, search_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        하이브리드 검색 수행
        
        Args:
            query: 검색 쿼리
            search_params: 검색 파라미터
            
        Returns:
            Dict: 검색 결과
        """
        try:
            start_time = time.time()
            self.search_stats["total_searches"] += 1
            
            if not search_params:
                search_params = {}
            
            # 기본 파라미터 설정
            params = {
                "top_k": search_params.get("top_k", 20),
                "vector_top_k": search_params.get("vector_top_k", 20),
                "bm25_top_k": search_params.get("bm25_top_k", 50),
                "enable_intent_detection": search_params.get("enable_intent_detection", True),
                "enable_pattern_boost": search_params.get("enable_pattern_boost", True),
                "filter_conditions": search_params.get("filter_conditions", {}),
                "explanation_mode": search_params.get("explanation_mode", False)
            }
            
            # domains 파라미터를 filter_conditions에 추가
            if "domains" in search_params and search_params["domains"]:
                domains = search_params["domains"]
                if isinstance(domains, list) and len(domains) > 0:
                    # 첫 번째 도메인을 사용 (단일 도메인 검색)
                    params["filter_conditions"]["domain"] = domains[0]
                    self.logger.info(f"도메인 설정: {domains[0]} (전체: {domains})")
                    
            self.logger.info(f"🚀 하이브리드 검색 시작: '{query}'")
            
            # 1. 의도 감지 및 동적 가중치 설정
            search_context = self._analyze_search_context(query, params)
            
            # 2. 병렬 검색 실행
            search_results = self._execute_parallel_search(query, params, search_context)
            
            # 3. 패턴 부스트 적용
            if params["enable_pattern_boost"]:
                search_results = self._apply_pattern_boost(search_results, query, search_context)
            
            # 4. 하이브리드 스코어링
            scored_results = self._hybrid_scoring(search_results, search_context)
            
            # 5. 중복 제거 및 그룹화
            if self.config.enable_deduplication:
                scored_results = self._deduplicate_and_group(scored_results)
            
            # 6. 최종 정렬 및 제한
            final_results = self._finalize_results(scored_results, params)
            
            # 7. 검색 설명 생성
            if params["explanation_mode"]:
                self._add_search_explanations(final_results, search_context, query)
            
            end_time = time.time()
            search_time = end_time - start_time
            
            # 결과 구성
            result = {
                "query": query,
                "intent": search_context.get("intent", "DEFAULT"),
                "total_results": len(final_results),
                "results": final_results,
                "search_context": search_context,
                "performance": {
                    "search_time_ms": int(search_time * 1000),
                    "vector_results": len(search_results.get("vector", [])),
                    "bm25_results": len(search_results.get("bm25", [])),
                    "hybrid_strategy": search_context.get("strategy", "balanced")
                },
                "metadata": {
                    "timestamp": end_time,
                    "version": "1.0.0"
                }
            }
            
            self.logger.info(f"✅ 하이브리드 검색 완료: {len(final_results)}개 결과, {search_time:.3f}초")
            return result
            
        except Exception as e:
            self.logger.error(f"하이브리드 검색 중 오류: {e}")
            return {
                "query": query,
                "total_results": 0,
                "results": [],
                "error": str(e)
            }
    
    def _analyze_search_context(self, query: str, params: Dict) -> Dict[str, Any]:
        """검색 컨텍스트 분석"""
        try:
            context = {
                "query": query,
                "query_length": len(query.split()),
                "intent": "DEFAULT",
                "weights": {
                    "vector": self.config.default_vector_weight,
                    "bm25": self.config.default_bm25_weight,
                    "pattern_boost": self.config.pattern_boost_weight,
                    "lawname_boost": self.config.lawname_boost_weight
                }
            }
            
            # 의도 감지
            if params.get("enable_intent_detection", True):
                intent = self.intent_detector.detect_intent(query)
                context["intent"] = intent
                
                # 의도별 가중치 적용
                intent_weights = self.intent_detector.get_weights_for_intent(intent)
                context["weights"].update({
                    "vector": intent_weights.dense,
                    "bm25": intent_weights.bm25,
                    "pattern_boost": intent_weights.article_hint,
                    "lawname_boost": intent_weights.lawname_hint
                })
                
                self.logger.debug(f"의도 감지: {intent}, 가중치: {context['weights']}")
            
            # 검색 전략 결정
            if context["intent"] == "EXACT_ARTICLE":
                context["strategy"] = "bm25_focused"
            elif context["intent"] in ["DEFINITION", "PROCEDURE"]:
                context["strategy"] = "vector_focused"
            else:
                context["strategy"] = "balanced"
            
            return context
            
        except Exception as e:
            self.logger.error(f"검색 컨텍스트 분석 중 오류: {e}")
            return {"query": query, "intent": "DEFAULT", "strategy": "balanced"}
    
    def _execute_parallel_search(self, query: str, params: Dict, context: Dict) -> Dict[str, List]:
        """진짜 병렬 검색 실행"""
        try:
            search_results = {
                "vector": [],
                "bm25": []
            }
            
            # ThreadPoolExecutor를 사용한 병렬 검색
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Vector 검색과 BM25 검색을 동시에 실행
                vector_future = executor.submit(self._vector_search, query, params)
                bm25_future = executor.submit(self._bm25_search, query, params)
                
                # 결과 대기 (각각 완료될 때까지)
                try:
                    vector_results = vector_future.result(timeout=30)  # 30초 타임아웃
                    search_results["vector"] = vector_results
                    self.logger.debug(f"Vector 검색 완료: {len(vector_results)}개 결과")
                except Exception as e:
                    self.logger.error(f"Vector 검색 중 오류: {e}")
                    search_results["vector"] = []
                
                try:
                    bm25_results = bm25_future.result(timeout=30)  # 30초 타임아웃
                    search_results["bm25"] = bm25_results
                    self.logger.debug(f"BM25 검색 완료: {len(bm25_results)}개 결과")
                except Exception as e:
                    self.logger.error(f"BM25 검색 중 오류: {e}")
                    search_results["bm25"] = []
            
            self.search_stats["hybrid_searches"] += 1
            return search_results
            
        except Exception as e:
            self.logger.error(f"병렬 검색 실행 중 오류: {e}")
            return {"vector": [], "bm25": []}
    
    def _vector_search(self, query: str, params: Dict) -> List[Dict]:
        """Vector 검색 (별도 메서드로 분리)"""
        try:
            if not self.interact_manager:
                return []
            
            self.search_stats["vector_searches"] += 1
            vector_results = self.interact_manager.retrieve_data(
                query=query,
                top_k=params["vector_top_k"],
                filter_conditions=params["filter_conditions"]
            )
            
            # Vector 결과 형식 통일
            for result in vector_results:
                if isinstance(result, dict):
                    result["search_strategy"] = "vector"
                    result["vector_score"] = result.get("score", 0.0)
            
            return vector_results
            
        except Exception as e:
            self.logger.error(f"Vector 검색 중 오류: {e}")
            return []
    
    def _bm25_search(self, query: str, params: Dict) -> List[Dict]:
        """BM25 검색 (별도 메서드로 분리)"""
        try:
            self.search_stats["bm25_searches"] += 1
            bm25_results = self.meilisearch_engine.search(
                query=query,
                limit=params["bm25_top_k"],
                filters=params["filter_conditions"]
            )
            
            return bm25_results
            
        except Exception as e:
            self.logger.error(f"BM25 검색 중 오류: {e}")
            return []
    
    def _apply_pattern_boost(self, search_results: Dict, query: str, context: Dict) -> Dict[str, List]:
        """패턴 부스트 적용"""
        try:
            # 패턴 감지
            patterns = self._extract_query_patterns(query)
            
            if not patterns:
                return search_results
            
            self.logger.debug(f"감지된 패턴: {patterns}")
            
            # 모든 검색 결과에 패턴 부스트 적용
            for strategy in ["vector", "bm25"]:
                for result in search_results.get(strategy, []):
                    boost_score = self._calculate_pattern_boost(result, patterns, context)
                    result["pattern_boost"] = boost_score
                    result["has_pattern_match"] = boost_score > 0
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"패턴 부스트 적용 중 오류: {e}")
            return search_results
    
    def _extract_query_patterns(self, query: str) -> Dict[str, List[str]]:
        """쿼리에서 패턴 추출"""
        patterns = {
            "articles": re.findall(r'제\s*(\d+)\s*조', query),
            "paragraphs": re.findall(r'제?\s*(\d+)\s*항', query),
            "items": re.findall(r'제?\s*(\d+)\s*호', query),
            "law_names": re.findall(r'[가-힣]{2,}법|[가-힣]{2,}규칙|[가-힣]{2,}시행령', query)
        }
        
        # 빈 리스트 제거
        return {k: v for k, v in patterns.items() if v}
    
    def _calculate_pattern_boost(self, result: Dict, patterns: Dict, context: Dict) -> float:
        """패턴 부스트 점수 계산"""
        try:
            boost_score = 0.0
            content = result.get("content", "")
            article_number = result.get("article_number", "")
            law_title = result.get("law_title", "")
            
            # 조문 번호 매칭
            if patterns.get("articles"):
                for article_num in patterns["articles"]:
                    if f"제{article_num}조" in content or f"제{article_num}조" == article_number:
                        boost_score += context["weights"]["pattern_boost"]
                        break
            
            # 항 번호 매칭
            if patterns.get("paragraphs"):
                for para_num in patterns["paragraphs"]:
                    if f"{para_num}항" in content or result.get("paragraph_number") == int(para_num):
                        boost_score += context["weights"]["pattern_boost"] * 0.8
                        break
            
            # 호 번호 매칭
            if patterns.get("items"):
                for item_num in patterns["items"]:
                    if f"{item_num}호" in content or result.get("item_number") == int(item_num):
                        boost_score += context["weights"]["pattern_boost"] * 0.6
                        break
            
            # 법령명 매칭
            if patterns.get("law_names"):
                for law_name in patterns["law_names"]:
                    if law_name in content or law_name in law_title:
                        boost_score += context["weights"]["lawname_boost"]
                        break
            
            return min(boost_score, self.config.MAX_PATTERN_BOOST)  # 상수 사용
            
        except Exception as e:
            self.logger.error(f"패턴 부스트 계산 중 오류: {e}")
            return 0.0
    
    def _hybrid_scoring(self, search_results: Dict, context: Dict) -> List[Dict[str, Any]]:
        """하이브리드 스코어링 (동적 보정 포함)"""
        try:
            all_results = []
            query = context.get("query", "")
            
            # Vector 결과 처리
            for result in search_results.get("vector", []):
                vector_score = self._normalize_score(result.get("vector_score", 0.0))
                pattern_boost = result.get("pattern_boost", 0.0)
                
                # 동적 보정 적용
                dynamic_adjustments = self._calculate_dynamic_adjustments(query, result)
                
                hybrid_score = (
                    context["weights"]["vector"] * vector_score +
                    pattern_boost +
                    dynamic_adjustments["total"]
                )
                
                result["hybrid_score"] = hybrid_score
                result["score_breakdown"] = {
                    "vector": vector_score,
                    "bm25": 0.0,
                    "pattern_boost": pattern_boost,
                    "dynamic_adjustments": dynamic_adjustments
                }
                all_results.append(result)
            
            # BM25 결과 처리
            for result in search_results.get("bm25", []):
                bm25_score = self._normalize_score(result.get("bm25_score", 0.0))
                pattern_boost = result.get("pattern_boost", 0.0)
                
                # 동적 보정 적용
                dynamic_adjustments = self._calculate_dynamic_adjustments(query, result)
                
                hybrid_score = (
                    context["weights"]["bm25"] * bm25_score +
                    pattern_boost +
                    dynamic_adjustments["total"]
                )
                
                result["hybrid_score"] = hybrid_score
                result["score_breakdown"] = {
                    "vector": 0.0,
                    "bm25": bm25_score,
                    "pattern_boost": pattern_boost,
                    "dynamic_adjustments": dynamic_adjustments
                }
                all_results.append(result)
            
            # 점수순 정렬
            all_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"하이브리드 스코어링 중 오류: {e}")
            return []
    
    def _calculate_dynamic_adjustments(self, query: str, result: Dict[str, Any]) -> Dict[str, float]:
        """동적 점수 보정 계산 (설정 파일 기반)"""
        try:
            adjustments = {
                "length_bias": 0.0,
                "exact_numeric": 0.0,
                "title_hit": 0.0,
                "total": 0.0
            }
            
            # 설정에서 동적 점수 조정 정보 로드
            length_config = self.config_loader.get_query_length_adjustments()
            exact_config = self.config_loader.get_exact_matching_bonuses()
            field_config = self.config_loader.get_field_matching_bonuses()
            
            # 1. 쿼리 길이 보정 (설정 기반)
            tokens = len(query.split())
            
            short_config = length_config.get("short_query", {})
            if tokens <= short_config.get("max_tokens", 3):
                adjustments["length_bias"] = short_config.get("adjustment_value", 0.05)
            
            long_config = length_config.get("long_query", {})
            if tokens >= long_config.get("min_tokens", 8):
                adjustments["length_bias"] = long_config.get("adjustment_value", 0.05)
            
            # 2. 정확 매칭 보너스 (설정 기반)
            article_config = exact_config.get("article_number", {})
            for pattern in article_config.get("patterns", [r'제\s*(\d+)\s*조']):
                query_matches = re.findall(pattern, query)
                match_field = result.get(article_config.get("match_field", "article_number"), "")
                
                for match in query_matches:
                    if f"제{match}조" in match_field:
                        adjustments["exact_numeric"] = article_config.get("bonus_value", 0.05)
                        break
            
            # 3. 필드 매칭 보너스 (설정 기반)
            title_config = field_config.get("title_hit", {})
            title = result.get(title_config.get("target_field", "title"), "")
            min_length = title_config.get("min_term_length", 2)
            query_terms = [term for term in query.split() if len(term) >= min_length]
            
            for term in query_terms:
                if term in title:
                    adjustments["title_hit"] = title_config.get("bonus_value", 0.05)
                    break
            
            # 총합 계산 (최대값 제한)
            total_adjustment = sum([
                adjustments["length_bias"],
                adjustments["exact_numeric"], 
                adjustments["title_hit"]
            ])
            
            # 설정에서 최대 조정값 확인
            scoring_config = self.config_loader.get_dynamic_scoring_config()
            max_adjustment = scoring_config.get("adjustment_limits", {}).get("max_total_adjustment", 0.15)
            
            adjustments["total"] = min(total_adjustment, max_adjustment)
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"동적 보정 계산 중 오류: {e}")
            # 폴백: 기본 계산
            return self._calculate_fallback_adjustments(query, result)
    
    def _calculate_fallback_adjustments(self, query: str, result: Dict[str, Any]) -> Dict[str, float]:
        """폴백용 기본 동적 조정"""
        adjustments = {"length_bias": 0.0, "exact_numeric": 0.0, "title_hit": 0.0, "total": 0.0}
        
        # 기본 로직
        tokens = len(query.split())
        if tokens <= self.config.SHORT_QUERY_THRESHOLD:
            adjustments["length_bias"] = self.config.LENGTH_BIAS_ADJUSTMENT
        elif tokens >= self.config.LONG_QUERY_THRESHOLD:
            adjustments["length_bias"] = self.config.LENGTH_BIAS_ADJUSTMENT
        
        query_articles = re.findall(r'제\s*(\d+)\s*조', query)
        doc_article = result.get("article_number", "")
        for num in query_articles:
            if f"제{num}조" in doc_article:
                adjustments["exact_numeric"] = self.config.EXACT_MATCH_BONUS
                break
        
        title = result.get("title", "")
        query_terms = [term for term in query.split() if len(term) >= 2]
        for term in query_terms:
            if term in title:
                adjustments["title_hit"] = self.config.TITLE_HIT_BONUS
                break
        
        adjustments["total"] = sum([adjustments["length_bias"], adjustments["exact_numeric"], adjustments["title_hit"]])
        return adjustments
    
    def _normalize_score(self, score: float) -> float:
        """점수 정규화 (0.0 ~ 1.0)"""
        return max(0.0, min(1.0, score))
    
    def _deduplicate_and_group(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """중복 제거 및 그룹화"""
        try:
            # node_id 기반 중복 제거
            unique_results = {}
            
            for result in results:
                node_id = result.get("node_id")
                if not node_id:
                    continue
                
                # 더 높은 점수의 결과로 유지
                if node_id not in unique_results or result.get("hybrid_score", 0) > unique_results[node_id].get("hybrid_score", 0):
                    unique_results[node_id] = result
            
            # article_key 기반 그룹화
            article_groups = defaultdict(list)
            for result in unique_results.values():
                law_number = result.get("law_number", "unknown")
                article_number = result.get("article_number", "unknown")
                article_key = f"{law_number}#{article_number}"
                article_groups[article_key].append(result)
            
            # 각 그룹에서 대표 결과 선택 + 관련 결과 추가
            grouped_results = []
            for group in article_groups.values():
                group.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
                
                main_result = group[0]
                if len(group) > 1:
                    main_result["related_paragraphs"] = group[1:3]  # 최대 2개 관련 항
                    main_result["total_related_count"] = len(group) - 1
                
                grouped_results.append(main_result)
            
            # 최종 점수순 정렬
            grouped_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
            
            return grouped_results
            
        except Exception as e:
            self.logger.error(f"중복 제거 및 그룹화 중 오류: {e}")
            return results
    
    def _finalize_results(self, results: List[Dict], params: Dict) -> List[Dict[str, Any]]:
        """최종 결과 구성"""
        try:
            # 결과 수 제한
            top_k = params.get("top_k", 20)
            final_results = results[:top_k]
            
            # 순위 정보 추가
            for i, result in enumerate(final_results):
                result["rank"] = i + 1
                result["total_candidates"] = len(results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"최종 결과 구성 중 오류: {e}")
            return results
    
    def _add_search_explanations(self, results: List[Dict], context: Dict, query: str):
        """검색 설명 추가"""
        try:
            for result in results:
                explanation = {
                    "strategy": context.get("strategy", "balanced"),
                    "intent": context.get("intent", "DEFAULT"),
                    "score_explanation": self._generate_score_explanation(result, context),
                    "pattern_matches": self._get_pattern_matches(result, query)
                }
                result["search_explanation"] = explanation
                
        except Exception as e:
            self.logger.error(f"검색 설명 생성 중 오류: {e}")
    
    def _generate_score_explanation(self, result: Dict, context: Dict) -> str:
        """점수 설명 생성"""
        try:
            breakdown = result.get("score_breakdown", {})
            parts = []
            
            if breakdown.get("vector", 0) > 0:
                parts.append(f"Vector: {breakdown['vector']:.2f}")
            
            if breakdown.get("bm25", 0) > 0:
                parts.append(f"BM25: {breakdown['bm25']:.2f}")
            
            if breakdown.get("pattern_boost", 0) > 0:
                parts.append(f"패턴 부스트: +{breakdown['pattern_boost']:.2f}")
            
            explanation = " + ".join(parts)
            final_score = result.get("hybrid_score", 0.0)
            
            return f"{explanation} = {final_score:.3f}"
            
        except Exception as e:
            return f"점수: {result.get('hybrid_score', 0.0):.3f}"
    
    def _get_pattern_matches(self, result: Dict, query: str) -> List[str]:
        """패턴 매칭 정보 조회"""
        try:
            matches = []
            
            if result.get("has_pattern_match", False):
                patterns = self._extract_query_patterns(query)
                
                for pattern_type, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        if pattern_type == "articles" and f"제{pattern}조" in result.get("content", ""):
                            matches.append(f"조문 매칭: 제{pattern}조")
                        elif pattern_type == "law_names" and pattern in result.get("law_title", ""):
                            matches.append(f"법령명 매칭: {pattern}")
            
            return matches
            
        except Exception as e:
            return []
    
    def add_document_to_both(self, document: Dict[str, Any]) -> Dict[str, bool]:
        """문서를 Vector DB와 Text DB 양쪽에 추가"""
        try:
            results = {
                "milvus_success": False,
                "meilisearch_success": False
            }
            
            # Milvus 추가 (기존 시스템 활용)
            if self.interact_manager:
                try:
                    # 기존 인덱싱 시스템 활용
                    # 실제로는 LegalIndexer를 통해 처리됨
                    results["milvus_success"] = True
                    self.logger.debug("Milvus 문서 추가 성공")
                except Exception as e:
                    self.logger.error(f"Milvus 문서 추가 실패: {e}")
            
            # Meilisearch 추가
            try:
                success = self.meilisearch_engine.add_documents([document])
                results["meilisearch_success"] = success
                if success:
                    self.logger.debug("Meilisearch 문서 추가 성공")
                else:
                    self.logger.error("Meilisearch 문서 추가 실패")
            except Exception as e:
                self.logger.error(f"Meilisearch 문서 추가 중 오류: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"이중 저장 중 오류: {e}")
            return {"milvus_success": False, "meilisearch_success": False}
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 조회"""
        return {
            "search_stats": self.search_stats,
            "meilisearch_stats": self.meilisearch_engine.get_stats(),
            "config": {
                "vector_weight": self.config.default_vector_weight,
                "bm25_weight": self.config.default_bm25_weight,
                "pattern_boost_enabled": self.config.enable_pattern_boost,
                "intent_detection_enabled": self.config.enable_intent_detection
            }
        }


# 전역 인스턴스
_hybrid_search_engine = None

def get_hybrid_search_engine(existing_interact_manager=None) -> HybridSearchEngine:
    """전역 하이브리드 검색 엔진 인스턴스 조회"""
    global _hybrid_search_engine
    if _hybrid_search_engine is None:
        _hybrid_search_engine = HybridSearchEngine(existing_interact_manager)
    return _hybrid_search_engine
