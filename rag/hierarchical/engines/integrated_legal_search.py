"""
통합 법령 검색 시스템

모든 고급 기능을 통합한 완전한 법령 검색 시스템입니다.
Vector + BM25 하이브리드 검색, 고급 재랭킹, 위계 컨텍스트, 결과 설명을 모두 제공합니다.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .hybrid_search_engine import get_hybrid_search_engine
from .advanced_reranker import get_advanced_reranker
from .context_enhancer import get_context_enhancer
from .result_explainer import get_result_explainer
from .meilisearch_client import get_meilisearch_engine


@dataclass
class IntegratedSearchConfig:
    """통합 검색 설정"""
    enable_hybrid_search: bool = True
    enable_advanced_reranking: bool = True
    enable_context_enhancement: bool = True
    enable_result_explanation: bool = True
    enable_performance_monitoring: bool = True
    max_processing_time: float = 30.0  # 최대 처리 시간 (초)


class IntegratedLegalSearchSystem:
    """통합 법령 검색 시스템"""
    
    def __init__(self, existing_interact_manager=None, config: Optional[IntegratedSearchConfig] = None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager 인스턴스
            config: 통합 검색 설정
        """
        self.interact_manager = existing_interact_manager
        self.config = config or IntegratedSearchConfig()
        self.logger = logging.getLogger(__name__)
        
        # 검색 엔진 컴포넌트들
        self.hybrid_engine = None
        self.reranker = None
        self.context_enhancer = None
        self.result_explainer = None
        self.meilisearch_engine = None
        
        # 초기화
        self._initialize_components()
        
        # 성능 통계
        self.performance_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "average_response_time": 0.0,
            "component_timings": {
                "hybrid_search": 0.0,
                "reranking": 0.0,
                "context_enhancement": 0.0,
                "result_explanation": 0.0
            }
        }
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            if self.config.enable_hybrid_search:
                self.hybrid_engine = get_hybrid_search_engine(self.interact_manager)
                self.logger.info("하이브리드 검색 엔진 초기화 완료")
            
            if self.config.enable_advanced_reranking:
                self.reranker = get_advanced_reranker()
                self.logger.info("고급 재랭킹 시스템 초기화 완료")
            
            if self.config.enable_context_enhancement:
                # retriever 인스턴스 필요 시 전달
                self.context_enhancer = get_context_enhancer()
                self.logger.info("위계 컨텍스트 강화 시스템 초기화 완료")
            
            if self.config.enable_result_explanation:
                self.result_explainer = get_result_explainer()
                self.logger.info("결과 설명 시스템 초기화 완료")
            
            # Meilisearch 엔진
            self.meilisearch_engine = get_meilisearch_engine()
            
            self.logger.info("🚀 통합 법령 검색 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 중 오류: {e}")
            raise
    
    def search(self, query: str, search_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        통합 법령 검색 수행
        
        Args:
            query: 검색 쿼리
            search_params: 검색 파라미터
                - top_k: 결과 개수 (기본: 15)
                - collection_name: 컬렉션 이름 (기본: "legal_documents")
                - enable_explanation: 설명 생성 여부 (기본: True)
                - enable_context: 컨텍스트 강화 여부 (기본: True)
                - filter_conditions: 필터 조건
                
        Returns:
            Dict: 통합 검색 결과
        """
        try:
            start_time = time.time()
            self.performance_stats["total_searches"] += 1
            
            if not search_params:
                search_params = {}
            
            # 기본 파라미터 설정
            params = {
                "top_k": search_params.get("top_k", 15),
                "collection_name": search_params.get("collection_name", "legal_documents"),
                "enable_explanation": search_params.get("enable_explanation", True),
                "enable_context": search_params.get("enable_context", True),
                "filter_conditions": search_params.get("filter_conditions", {}),
                "explanation_mode": search_params.get("explanation_mode", True)
            }
            
            self.logger.info(f"🏛️ 통합 법령 검색 시작: '{query}'")
            
            # 검색 결과 변수 초기화
            search_result = None
            component_timings = {}
            
            # 1단계: 하이브리드 검색
            if self.config.enable_hybrid_search and self.hybrid_engine:
                step_start = time.time()
                
                hybrid_params = {
                    "top_k": params["top_k"] * 2,  # 재랭킹을 위해 더 많이 검색
                    "enable_intent_detection": True,
                    "enable_pattern_boost": True,
                    "filter_conditions": params["filter_conditions"],
                    "explanation_mode": params["explanation_mode"]
                }
                
                search_result = self.hybrid_engine.hybrid_search(query, hybrid_params)
                component_timings["hybrid_search"] = time.time() - step_start
                
                self.logger.info(f"   ✅ 하이브리드 검색 완료: {search_result.get('total_results', 0)}개 결과")
            else:
                # 폴백: 기본 벡터 검색
                search_result = self._fallback_vector_search(query, params)
                component_timings["hybrid_search"] = 0.0
            
            if not search_result or not search_result.get("results"):
                return self._create_empty_result(query, "검색 결과가 없습니다")
            
            results = search_result["results"]
            search_context = search_result.get("search_context", {})
            
            # 2단계: 고급 재랭킹
            if self.config.enable_advanced_reranking and self.reranker and len(results) > 1:
                step_start = time.time()
                
                results = self.reranker.rerank_results(results, query, search_context)
                component_timings["reranking"] = time.time() - step_start
                
                self.logger.info(f"   ✅ 고급 재랭킹 완료")
            else:
                component_timings["reranking"] = 0.0
            
            # 결과 수 제한
            results = results[:params["top_k"]]
            
            # 3단계: 위계 컨텍스트 강화
            if self.config.enable_context_enhancement and self.context_enhancer and params["enable_context"]:
                step_start = time.time()
                
                collection_name = params["collection_name"]
                results = self.context_enhancer.enhance_results_with_context(results, collection_name)
                component_timings["context_enhancement"] = time.time() - step_start
                
                self.logger.info(f"   ✅ 위계 컨텍스트 강화 완료")
            else:
                component_timings["context_enhancement"] = 0.0
            
            # 4단계: 결과 설명 생성
            if self.config.enable_result_explanation and self.result_explainer and params["enable_explanation"]:
                step_start = time.time()
                
                results = self.result_explainer.explain_search_results(results, query, search_context)
                component_timings["result_explanation"] = time.time() - step_start
                
                self.logger.info(f"   ✅ 결과 설명 생성 완료")
            else:
                component_timings["result_explanation"] = 0.0
            
            # 최종 결과 구성
            total_time = time.time() - start_time
            
            final_result = {
                "query": query,
                "intent": search_context.get("intent", "DEFAULT"),
                "strategy": search_context.get("strategy", "balanced"),
                "total_results": len(results),
                "results": results,
                "search_metadata": {
                    "search_type": "integrated_legal_search",
                    "components_used": self._get_enabled_components(),
                    "processing_pipeline": [
                        "hybrid_search",
                        "advanced_reranking", 
                        "context_enhancement",
                        "result_explanation"
                    ],
                    "search_context": search_context
                },
                "performance": {
                    "total_time_ms": int(total_time * 1000),
                    "component_timings_ms": {k: int(v * 1000) for k, v in component_timings.items()},
                    "average_per_result_ms": int((total_time / len(results)) * 1000) if results else 0
                },
                "quality_indicators": {
                    "has_exact_matches": any(r.get("has_pattern_match", False) for r in results),
                    "has_hierarchy_context": any(r.get("hierarchy_context") for r in results),
                    "has_explanations": any(r.get("explanations") for r in results),
                    "diversity_score": self._calculate_diversity_score(results)
                },
                "metadata": {
                    "timestamp": time.time(),
                    "version": "1.0.0",
                    "system": "integrated_legal_search"
                }
            }
            
            # 성능 통계 업데이트
            self._update_performance_stats(total_time, component_timings)
            self.performance_stats["successful_searches"] += 1
            
            self.logger.info(f"🎉 통합 법령 검색 완료: {len(results)}개 결과, {total_time:.3f}초")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"통합 검색 중 오류: {e}")
            return self._create_error_result(query, str(e))
    
    def _fallback_vector_search(self, query: str, params: Dict) -> Dict[str, Any]:
        """폴백 벡터 검색"""
        try:
            if not self.interact_manager:
                return None
            
            results = self.interact_manager.retrieve_data(
                query=query,
                top_k=params["top_k"],
                filter_conditions=params["filter_conditions"]
            )
            
            return {
                "query": query,
                "total_results": len(results),
                "results": results,
                "search_context": {"strategy": "vector_fallback"}
            }
            
        except Exception as e:
            self.logger.error(f"폴백 검색 중 오류: {e}")
            return None
    
    def _get_enabled_components(self) -> List[str]:
        """활성화된 컴포넌트 목록"""
        components = []
        
        if self.config.enable_hybrid_search:
            components.append("hybrid_search")
        if self.config.enable_advanced_reranking:
            components.append("advanced_reranking")
        if self.config.enable_context_enhancement:
            components.append("context_enhancement")
        if self.config.enable_result_explanation:
            components.append("result_explanation")
        
        return components
    
    def _calculate_diversity_score(self, results: List[Dict]) -> float:
        """결과 다양성 점수 계산"""
        try:
            if not results:
                return 0.0
            
            # 법령별 분포
            laws = set()
            # 위계 레벨별 분포  
            levels = set()
            # 조문별 분포
            articles = set()
            
            for result in results:
                law_title = result.get("law_title")
                if law_title:
                    laws.add(law_title)
                
                hierarchy_level = result.get("hierarchy_level")
                if hierarchy_level is not None:
                    levels.add(hierarchy_level)
                
                article_number = result.get("article_number")
                if article_number:
                    articles.add(article_number)
            
            # 다양성 점수 (0.0 ~ 1.0)
            law_diversity = min(len(laws) / 3.0, 1.0)  # 최대 3개 법령
            level_diversity = min(len(levels) / 3.0, 1.0)  # 최대 3개 레벨
            article_diversity = min(len(articles) / len(results), 1.0)
            
            return (law_diversity + level_diversity + article_diversity) / 3.0
            
        except Exception as e:
            self.logger.error(f"다양성 점수 계산 중 오류: {e}")
            return 0.0
    
    def _update_performance_stats(self, total_time: float, component_timings: Dict[str, float]):
        """성능 통계 업데이트"""
        try:
            # 평균 응답 시간 업데이트
            current_avg = self.performance_stats["average_response_time"]
            total_searches = self.performance_stats["total_searches"]
            
            self.performance_stats["average_response_time"] = (
                (current_avg * (total_searches - 1) + total_time) / total_searches
            )
            
            # 컴포넌트별 타이밍 업데이트
            for component, timing in component_timings.items():
                current_timing = self.performance_stats["component_timings"].get(component, 0.0)
                self.performance_stats["component_timings"][component] = (
                    (current_timing * (total_searches - 1) + timing) / total_searches
                )
            
        except Exception as e:
            self.logger.error(f"성능 통계 업데이트 중 오류: {e}")
    
    def _create_empty_result(self, query: str, message: str) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            "query": query,
            "total_results": 0,
            "results": [],
            "message": message,
            "search_metadata": {
                "search_type": "integrated_legal_search",
                "status": "no_results"
            },
            "performance": {
                "total_time_ms": 0
            },
            "metadata": {
                "timestamp": time.time(),
                "system": "integrated_legal_search"
            }
        }
    
    def _create_error_result(self, query: str, error: str) -> Dict[str, Any]:
        """오류 결과 생성"""
        return {
            "query": query,
            "total_results": 0,
            "results": [],
            "error": error,
            "search_metadata": {
                "search_type": "integrated_legal_search",
                "status": "error"
            },
            "performance": {
                "total_time_ms": 0
            },
            "metadata": {
                "timestamp": time.time(),
                "system": "integrated_legal_search"
            }
        }
    
    def add_document(self, document: Dict[str, Any]) -> Dict[str, bool]:
        """문서 추가 (이중 저장)"""
        try:
            self.logger.info(f"📄 문서 추가: {document.get('node_id', 'unknown')}")
            
            results = {
                "milvus_success": False,
                "meilisearch_success": False
            }
            
            # Meilisearch에 추가
            if self.meilisearch_engine:
                try:
                    success = self.meilisearch_engine.add_documents([document])
                    results["meilisearch_success"] = success
                    if success:
                        self.logger.debug("Meilisearch 문서 추가 성공")
                except Exception as e:
                    self.logger.error(f"Meilisearch 문서 추가 중 오류: {e}")
            
            # Milvus 추가는 기존 인덱싱 시스템에서 처리
            # (LegalIndexer 등을 통해)
            results["milvus_success"] = True  # 기존 시스템에서 처리된다고 가정
            
            return results
            
        except Exception as e:
            self.logger.error(f"문서 추가 중 오류: {e}")
            return {"milvus_success": False, "meilisearch_success": False}
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            status = {
                "system": "integrated_legal_search",
                "status": "healthy",
                "components": {},
                "performance": self.performance_stats.copy(),
                "configuration": {
                    "hybrid_search_enabled": self.config.enable_hybrid_search,
                    "reranking_enabled": self.config.enable_advanced_reranking,
                    "context_enhancement_enabled": self.config.enable_context_enhancement,
                    "result_explanation_enabled": self.config.enable_result_explanation
                }
            }
            
            # 각 컴포넌트 상태 확인
            if self.meilisearch_engine:
                try:
                    meilisearch_status = self.meilisearch_engine.health_check()
                    status["components"]["meilisearch"] = meilisearch_status
                except Exception as e:
                    status["components"]["meilisearch"] = {"status": "unhealthy", "error": str(e)}
            
            if self.hybrid_engine:
                try:
                    hybrid_stats = self.hybrid_engine.get_search_stats()
                    status["components"]["hybrid_engine"] = {"status": "healthy", "stats": hybrid_stats}
                except Exception as e:
                    status["components"]["hybrid_engine"] = {"status": "unhealthy", "error": str(e)}
            
            # 전체 상태 판정
            component_statuses = [comp.get("status") for comp in status["components"].values()]
            if "unhealthy" in component_statuses:
                status["status"] = "degraded"
            elif not component_statuses:
                status["status"] = "minimal"
            
            return status
            
        except Exception as e:
            self.logger.error(f"시스템 상태 조회 중 오류: {e}")
            return {
                "system": "integrated_legal_search",
                "status": "error",
                "error": str(e)
            }
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 조회"""
        return {
            "performance_stats": self.performance_stats.copy(),
            "component_stats": {
                "hybrid_engine": self.hybrid_engine.get_search_stats() if self.hybrid_engine else {},
                "meilisearch": self.meilisearch_engine.get_stats() if self.meilisearch_engine else {}
            },
            "system_health": self.get_system_status()
        }


# 전역 인스턴스
_integrated_search_system = None

def get_integrated_search_system(existing_interact_manager=None) -> IntegratedLegalSearchSystem:
    """전역 통합 검색 시스템 인스턴스 조회"""
    global _integrated_search_system
    if _integrated_search_system is None:
        _integrated_search_system = IntegratedLegalSearchSystem(existing_interact_manager)
    return _integrated_search_system
