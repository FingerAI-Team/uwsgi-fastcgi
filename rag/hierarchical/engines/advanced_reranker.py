"""
고급 재랭킹 시스템

하이브리드 검색 결과를 법령의 위계 구조와 관련성을 고려하여
최적으로 재정렬합니다.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from ..config.config_loader import get_config_loader


@dataclass
class RerankerConfig:
    """재랭킹 설정"""
    enable_adjacent_articles: bool = True
    enable_hierarchy_boost: bool = True
    enable_legal_structure_analysis: bool = True
    adjacent_article_boost: float = 0.1
    hierarchy_boost_factor: float = 0.15
    diversity_penalty_factor: float = 0.05
    max_same_article_results: int = 3


class AdvancedLegalReranker:
    """고급 법령 재랭킹 시스템"""
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Args:
            config: 재랭킹 설정
        """
        self.config = config or RerankerConfig()
        self.logger = logging.getLogger(__name__)
        self.config_loader = get_config_loader()
        
        # 법령 구조 가중치
        self.legal_structure_weights = {
            "article": 1.0,      # 조문
            "paragraph": 0.9,    # 항
            "item": 0.8,         # 호
            "subitem": 0.7,      # 목
            "chapter": 0.85,     # 장
            "section": 0.85      # 절
        }
        
        # 법령 관계 패턴
        self.legal_relations = {
            "reference": r"제\s*\d+\s*조|제\s*\d+\s*항|제\s*\d+\s*호",
            "exception": r"다만|단서|예외|제외",
            "application": r"준용|적용|이에\s*따라",
            "definition": r"라고?\s*한다|의미한다|정의",
            "procedure": r"절차|방법|기준|요건"
        }
    
    def rerank_results(self, results: List[Dict[str, Any]], query: str, 
                      search_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        고급 재랭킹 수행
        
        Args:
            results: 하이브리드 검색 결과
            query: 원본 쿼리
            search_context: 검색 컨텍스트
            
        Returns:
            List[Dict]: 재랭킹된 결과
        """
        try:
            if not results:
                return results
            
            self.logger.info(f"🎯 고급 재랭킹 시작: {len(results)}개 결과")
            
            # 1. 법령 구조 분석
            structure_analysis = self._analyze_legal_structure(results, query)
            
            # 2. 위계 관계 부스트
            if self.config.enable_hierarchy_boost:
                results = self._apply_hierarchy_boost(results, structure_analysis)
            
            # 3. 인접 조문 부스트
            if self.config.enable_adjacent_articles:
                results = self._apply_adjacent_article_boost(results, query)
            
            # 4. 법령 관계 분석 및 부스트
            if self.config.enable_legal_structure_analysis:
                results = self._apply_legal_relation_boost(results, query, structure_analysis)
            
            # 5. 다양성 확보
            results = self._ensure_result_diversity(results)
            
            # 6. 최종 스코어 계산 및 정렬
            results = self._calculate_final_scores(results, search_context)
            
            self.logger.info(f"✅ 고급 재랭킹 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            self.logger.error(f"고급 재랭킹 중 오류: {e}")
            return results
    
    def _analyze_legal_structure(self, results: List[Dict], query: str) -> Dict[str, Any]:
        """법령 구조 분석"""
        try:
            analysis = {
                "dominant_law": None,
                "dominant_articles": [],
                "hierarchy_distribution": defaultdict(int),
                "query_patterns": self._extract_comprehensive_patterns(query)
            }
            
            # 법령별, 조문별 분포 분석
            law_counts = defaultdict(int)
            article_counts = defaultdict(int)
            
            for result in results:
                law_title = result.get("law_title", "")
                article_number = result.get("article_number", "")
                hierarchy_level = result.get("hierarchy_level", 0)
                
                if law_title:
                    law_counts[law_title] += 1
                
                if article_number:
                    article_counts[article_number] += 1
                
                analysis["hierarchy_distribution"][hierarchy_level] += 1
            
            # 주요 법령 및 조문 결정
            if law_counts:
                analysis["dominant_law"] = max(law_counts, key=law_counts.get)
            
            if article_counts:
                # 상위 3개 조문
                sorted_articles = sorted(article_counts.items(), key=lambda x: x[1], reverse=True)
                analysis["dominant_articles"] = [art for art, count in sorted_articles[:3]]
            
            self.logger.debug(f"법령 구조 분석: {analysis}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"법령 구조 분석 중 오류: {e}")
            return {"query_patterns": {}}
    
    def _extract_comprehensive_patterns(self, query: str) -> Dict[str, List[str]]:
        """포괄적 패턴 추출"""
        try:
            patterns = {
                "articles": re.findall(r'제\s*(\d+)\s*조', query),
                "paragraphs": re.findall(r'제?\s*(\d+)\s*항', query),
                "items": re.findall(r'제?\s*(\d+)\s*호', query),
                "subitems": re.findall(r'([가-힣])\s*목', query),
                "chapters": re.findall(r'제\s*(\d+)\s*장', query),
                "sections": re.findall(r'제\s*(\d+)\s*절', query),
                "law_names": re.findall(r'[가-힣]{2,}법|[가-힣]{2,}규칙|[가-힣]{2,}시행령', query),
                "legal_terms": self._extract_legal_terms(query)
            }
            
            return {k: v for k, v in patterns.items() if v}
            
        except Exception as e:
            self.logger.error(f"패턴 추출 중 오류: {e}")
            return {}
    
    def _extract_legal_terms(self, query: str) -> List[str]:
        """법령 전문 용어 추출"""
        legal_terms = [
            "개인정보", "정보주체", "개인정보처리자", "개인정보보호책임자",
            "수집", "이용", "제공", "위탁", "파기", "동의", "고지",
            "안전성확보조치", "개인정보영향평가", "손해배상", "과태료"
        ]
        
        found_terms = []
        for term in legal_terms:
            if term in query:
                found_terms.append(term)
        
        return found_terms
    
    def _apply_hierarchy_boost(self, results: List[Dict], analysis: Dict) -> List[Dict[str, Any]]:
        """위계 관계 부스트 적용"""
        try:
            query_patterns = analysis.get("query_patterns", {})
            
            for result in results:
                hierarchy_boost = 0.0
                
                # 노드 타입별 기본 가중치
                node_type = result.get("node_type", "")
                base_weight = self.legal_structure_weights.get(node_type, 0.5)
                
                # 패턴 매칭 부스트
                if query_patterns.get("articles"):
                    article_number = result.get("article_number", "")
                    for pattern in query_patterns["articles"]:
                        if f"제{pattern}조" == article_number:
                            hierarchy_boost += 0.3
                            break
                
                if query_patterns.get("paragraphs"):
                    paragraph_number = result.get("paragraph_number", 0)
                    for pattern in query_patterns["paragraphs"]:
                        if int(pattern) == paragraph_number:
                            hierarchy_boost += 0.2
                            break
                
                # 주요 법령 부스트
                if analysis.get("dominant_law") == result.get("law_title"):
                    hierarchy_boost += 0.1
                
                # 최종 위계 부스트 계산
                total_hierarchy_boost = (base_weight + hierarchy_boost) * self.config.hierarchy_boost_factor
                
                # 기존 점수에 부스트 추가
                current_score = result.get("hybrid_score", 0.0)
                result["hierarchy_boost"] = total_hierarchy_boost
                result["boosted_score"] = current_score + total_hierarchy_boost
                
            return results
            
        except Exception as e:
            self.logger.error(f"위계 부스트 적용 중 오류: {e}")
            return results
    
    def _apply_adjacent_article_boost(self, results: List[Dict], query: str) -> List[Dict[str, Any]]:
        """인접 조문 부스트 적용"""
        try:
            # 쿼리에서 조문 번호 추출
            target_articles = re.findall(r'제\s*(\d+)\s*조', query)
            
            if not target_articles:
                return results
            
            target_nums = [int(num) for num in target_articles]
            
            for result in results:
                adjacent_boost = 0.0
                article_number = result.get("article_number", "")
                
                if article_number:
                    # 조문 번호 추출
                    article_match = re.search(r'제(\d+)조', article_number)
                    if article_match:
                        result_num = int(article_match.group(1))
                        
                        # 인접 조문 확인 (±1, ±2)
                        for target_num in target_nums:
                            distance = abs(result_num - target_num)
                            if distance == 1:
                                adjacent_boost += self.config.adjacent_article_boost
                            elif distance == 2:
                                adjacent_boost += self.config.adjacent_article_boost * 0.5
                
                result["adjacent_boost"] = adjacent_boost
                if adjacent_boost > 0:
                    current_score = result.get("boosted_score", result.get("hybrid_score", 0.0))
                    result["boosted_score"] = current_score + adjacent_boost
            
            return results
            
        except Exception as e:
            self.logger.error(f"인접 조문 부스트 적용 중 오류: {e}")
            return results
    
    def _apply_legal_relation_boost(self, results: List[Dict], query: str, 
                                  analysis: Dict) -> List[Dict[str, Any]]:
        """법령 관계 부스트 적용"""
        try:
            for result in results:
                relation_boost = 0.0
                content = result.get("content", "")
                
                # 각 법령 관계 패턴별 부스트
                for relation_type, pattern in self.legal_relations.items():
                    if re.search(pattern, content) and re.search(pattern, query):
                        if relation_type == "reference":
                            relation_boost += 0.15
                        elif relation_type == "exception":
                            relation_boost += 0.1
                        elif relation_type == "application":
                            relation_boost += 0.12
                        elif relation_type == "definition":
                            relation_boost += 0.08
                        elif relation_type == "procedure":
                            relation_boost += 0.05
                
                # 법령 용어 매칭 부스트
                legal_terms = analysis.get("query_patterns", {}).get("legal_terms", [])
                for term in legal_terms:
                    if term in content:
                        relation_boost += 0.03
                
                result["relation_boost"] = relation_boost
                if relation_boost > 0:
                    current_score = result.get("boosted_score", result.get("hybrid_score", 0.0))
                    result["boosted_score"] = current_score + relation_boost
            
            return results
            
        except Exception as e:
            self.logger.error(f"법령 관계 부스트 적용 중 오류: {e}")
            return results
    
    def _ensure_result_diversity(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """결과 다양성 확보"""
        try:
            # 조문별 결과 수 제한
            article_counts = defaultdict(int)
            diverse_results = []
            
            for result in results:
                article_number = result.get("article_number", "")
                
                if not article_number or article_counts[article_number] < self.config.max_same_article_results:
                    diverse_results.append(result)
                    article_counts[article_number] += 1
                else:
                    # 다양성 패널티 적용
                    diversity_penalty = self.config.diversity_penalty_factor * article_counts[article_number]
                    current_score = result.get("boosted_score", result.get("hybrid_score", 0.0))
                    result["diversity_penalty"] = diversity_penalty
                    result["boosted_score"] = max(0.0, current_score - diversity_penalty)
                    diverse_results.append(result)
            
            return diverse_results
            
        except Exception as e:
            self.logger.error(f"다양성 확보 중 오류: {e}")
            return results
    
    def _calculate_final_scores(self, results: List[Dict], context: Dict) -> List[Dict[str, Any]]:
        """최종 점수 계산 및 정렬"""
        try:
            for result in results:
                # 모든 부스트 요소 합산
                base_score = result.get("hybrid_score", 0.0)
                hierarchy_boost = result.get("hierarchy_boost", 0.0)
                adjacent_boost = result.get("adjacent_boost", 0.0)
                relation_boost = result.get("relation_boost", 0.0)
                diversity_penalty = result.get("diversity_penalty", 0.0)
                
                final_score = base_score + hierarchy_boost + adjacent_boost + relation_boost - diversity_penalty
                result["final_rerank_score"] = max(0.0, final_score)
                
                # 점수 구성 요소 상세 정보
                result["score_components"] = {
                    "base_hybrid": base_score,
                    "hierarchy_boost": hierarchy_boost,
                    "adjacent_boost": adjacent_boost,
                    "relation_boost": relation_boost,
                    "diversity_penalty": diversity_penalty,
                    "final": result["final_rerank_score"]
                }
            
            # 최종 점수로 정렬
            results.sort(key=lambda x: x.get("final_rerank_score", 0.0), reverse=True)
            
            # 순위 업데이트
            for i, result in enumerate(results):
                result["final_rank"] = i + 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"최종 점수 계산 중 오류: {e}")
            return results
    
    def get_reranking_explanation(self, result: Dict[str, Any]) -> Dict[str, str]:
        """재랭킹 설명 생성"""
        try:
            components = result.get("score_components", {})
            explanation = {
                "ranking_strategy": "고급 법령 재랭킹",
                "base_score": f"하이브리드 점수: {components.get('base_hybrid', 0.0):.3f}",
                "boosts_applied": [],
                "penalties_applied": []
            }
            
            # 부스트 요소들
            if components.get("hierarchy_boost", 0) > 0:
                explanation["boosts_applied"].append(
                    f"위계 부스트: +{components['hierarchy_boost']:.3f}"
                )
            
            if components.get("adjacent_boost", 0) > 0:
                explanation["boosts_applied"].append(
                    f"인접 조문 부스트: +{components['adjacent_boost']:.3f}"
                )
            
            if components.get("relation_boost", 0) > 0:
                explanation["boosts_applied"].append(
                    f"법령 관계 부스트: +{components['relation_boost']:.3f}"
                )
            
            # 패널티 요소들
            if components.get("diversity_penalty", 0) > 0:
                explanation["penalties_applied"].append(
                    f"다양성 패널티: -{components['diversity_penalty']:.3f}"
                )
            
            explanation["final_explanation"] = (
                f"최종 점수: {components.get('final', 0.0):.3f} "
                f"(순위: {result.get('final_rank', 'N/A')})"
            )
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"재랭킹 설명 생성 중 오류: {e}")
            return {"error": "설명 생성 실패"}


# 전역 인스턴스
_advanced_reranker = None

def get_advanced_reranker() -> AdvancedLegalReranker:
    """전역 고급 재랭킹 인스턴스 조회"""
    global _advanced_reranker
    if _advanced_reranker is None:
        _advanced_reranker = AdvancedLegalReranker()
    return _advanced_reranker