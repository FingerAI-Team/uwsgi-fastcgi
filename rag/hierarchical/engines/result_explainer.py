"""
검색 결과 설명 및 하이라이트 시스템

사용자가 검색 결과를 이해할 수 있도록 상세한 설명과 
하이라이트를 제공합니다.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class HighlightConfig:
    """하이라이트 설정"""
    enable_query_highlight: bool = True
    enable_pattern_highlight: bool = True
    enable_legal_term_highlight: bool = True
    max_highlight_length: int = 300
    context_window: int = 50


class SearchResultExplainer:
    """검색 결과 설명 및 하이라이트 시스템"""
    
    def __init__(self, config: Optional[HighlightConfig] = None):
        """
        Args:
            config: 하이라이트 설정
        """
        self.config = config or HighlightConfig()
        self.logger = logging.getLogger(__name__)
        
        # 법령 전문 용어 사전
        self.legal_terms = {
            "개인정보": ["개인식별정보", "개인데이터", "신상정보"],
            "개인정보처리자": ["처리자", "관리자", "취급자"],
            "정보주체": ["개인", "당사자", "본인"],
            "수집": ["취득", "획득", "접수"],
            "동의": ["승낙", "허락", "허가"],
            "처리": ["가공", "이용", "활용"],
            "제공": ["전달", "공유", "이전"],
            "위탁": ["위임", "의뢰", "대행"],
            "파기": ["삭제", "폐기", "소각"]
        }
        
        # 하이라이트 스타일
        self.highlight_styles = {
            "query_match": {"tag": "query", "class": "query-highlight"},
            "pattern_match": {"tag": "pattern", "class": "pattern-highlight"},
            "legal_term": {"tag": "term", "class": "legal-term-highlight"},
            "article_ref": {"tag": "article", "class": "article-highlight"},
            "emphasis": {"tag": "emphasis", "class": "emphasis-highlight"}
        }
    
    def explain_search_results(self, results: List[Dict[str, Any]], 
                             query: str, search_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        검색 결과에 상세 설명 추가
        
        Args:
            results: 검색 결과 리스트
            query: 원본 검색 쿼리
            search_context: 검색 컨텍스트
            
        Returns:
            List[Dict]: 설명이 추가된 검색 결과
        """
        try:
            if not results:
                return results
            
            self.logger.info(f"📖 검색 결과 설명 생성 시작: {len(results)}개 결과")
            
            explained_results = []
            
            for i, result in enumerate(results):
                try:
                    explained_result = self._explain_single_result(result, query, search_context, i)
                    explained_results.append(explained_result)
                    
                except Exception as e:
                    self.logger.error(f"개별 결과 설명 생성 중 오류: {e}")
                    explained_results.append(result)  # 원본 유지
            
            # 전체 결과 요약 설명 추가
            self._add_overall_explanation(explained_results, query, search_context)
            
            self.logger.info(f"✅ 검색 결과 설명 생성 완료")
            return explained_results
            
        except Exception as e:
            self.logger.error(f"검색 결과 설명 생성 중 오류: {e}")
            return results
    
    def _explain_single_result(self, result: Dict[str, Any], query: str, 
                             context: Dict[str, Any], rank: int) -> Dict[str, Any]:
        """개별 결과 설명 생성"""
        try:
            explained_result = result.copy()
            
            # 검색 매칭 설명
            matching_explanation = self._generate_matching_explanation(result, query, context)
            
            # 점수 구성 설명
            score_explanation = self._generate_score_explanation(result, context)
            
            # 하이라이트된 내용 생성
            highlighted_content = self._generate_highlighted_content(result, query)
            
            # 법령 구조 설명
            structure_explanation = self._generate_structure_explanation(result)
            
            # 관련성 설명
            relevance_explanation = self._generate_relevance_explanation(result, query)
            
            # 추천 이유 설명
            recommendation_reason = self._generate_recommendation_reason(result, rank, context)
            
            # 설명 정보 통합
            explained_result["explanations"] = {
                "matching": matching_explanation,
                "scoring": score_explanation,
                "structure": structure_explanation,
                "relevance": relevance_explanation,
                "recommendation": recommendation_reason
            }
            
            # 하이라이트 정보
            explained_result["highlights"] = highlighted_content
            
            # 사용자 친화적 요약
            explained_result["user_summary"] = self._generate_user_friendly_summary(
                result, matching_explanation, rank
            )
            
            return explained_result
            
        except Exception as e:
            self.logger.error(f"개별 결과 설명 생성 중 오류: {e}")
            return result
    
    def _generate_matching_explanation(self, result: Dict[str, Any], 
                                     query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """매칭 설명 생성"""
        try:
            explanation = {
                "match_type": [],
                "match_details": [],
                "confidence_factors": []
            }
            
            content = result.get("content", "")
            title = result.get("title", "")
            
            # 쿼리 용어 직접 매칭 확인
            query_terms = query.split()
            direct_matches = []
            
            for term in query_terms:
                if term in content or term in title:
                    direct_matches.append(term)
            
            if direct_matches:
                explanation["match_type"].append("직접 키워드 매칭")
                explanation["match_details"].append(f"매칭 키워드: {', '.join(direct_matches)}")
                explanation["confidence_factors"].append("키워드 정확 매칭으로 높은 신뢰도")
            
            # 패턴 매칭 확인
            if result.get("has_pattern_match", False):
                explanation["match_type"].append("법령 패턴 매칭")
                
                pattern_matches = []
                article_number = result.get("article_number", "")
                if article_number and article_number in query:
                    pattern_matches.append(f"조문 번호: {article_number}")
                
                if pattern_matches:
                    explanation["match_details"].append(f"패턴 매칭: {', '.join(pattern_matches)}")
                    explanation["confidence_factors"].append("법령 구조 패턴 매칭으로 정확성 확보")
            
            # 의미적 매칭 확인
            search_strategy = result.get("search_strategy", "")
            if "vector" in search_strategy:
                explanation["match_type"].append("의미적 유사성 매칭")
                explanation["match_details"].append("AI 임베딩을 통한 의미적 관련성 발견")
                explanation["confidence_factors"].append("컨텍스트 이해를 통한 포괄적 매칭")
            
            # BM25 매칭 확인
            if "bm25" in search_strategy:
                explanation["match_type"].append("통계적 키워드 매칭")
                explanation["match_details"].append("단어 빈도 및 문서 빈도 기반 관련성")
                explanation["confidence_factors"].append("검증된 정보 검색 알고리즘 활용")
            
            # 법령 용어 매칭 확인
            legal_term_matches = []
            for main_term, synonyms in self.legal_terms.items():
                if main_term in query:
                    for synonym in synonyms:
                        if synonym in content:
                            legal_term_matches.append(f"{main_term}↔{synonym}")
                            break
            
            if legal_term_matches:
                explanation["match_type"].append("법령 용어 확장 매칭")
                explanation["match_details"].append(f"용어 확장: {', '.join(legal_term_matches)}")
                explanation["confidence_factors"].append("법령 전문 용어 사전 기반 확장")
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"매칭 설명 생성 중 오류: {e}")
            return {"match_type": ["오류"], "match_details": [], "confidence_factors": []}
    
    def _generate_score_explanation(self, result: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """점수 구성 설명 생성"""
        try:
            explanation = {
                "total_score": result.get("final_rerank_score", result.get("hybrid_score", 0.0)),
                "components": [],
                "breakdown": {},
                "ranking_factors": []
            }
            
            # 점수 구성 요소 분석
            score_components = result.get("score_components", {})
            
            if score_components.get("base_hybrid", 0) > 0:
                explanation["components"].append({
                    "name": "기본 하이브리드 점수",
                    "value": score_components["base_hybrid"],
                    "description": "Vector 검색과 BM25 검색의 결합 점수"
                })
            
            if score_components.get("hierarchy_boost", 0) > 0:
                explanation["components"].append({
                    "name": "위계 구조 부스트",
                    "value": score_components["hierarchy_boost"],
                    "description": "법령의 위계 구조 적합성에 따른 가산점"
                })
            
            if score_components.get("pattern_boost", 0) > 0:
                explanation["components"].append({
                    "name": "패턴 매칭 부스트",
                    "value": score_components["pattern_boost"],
                    "description": "조문 번호 등 정확한 패턴 매칭에 따른 가산점"
                })
            
            if score_components.get("relation_boost", 0) > 0:
                explanation["components"].append({
                    "name": "법령 관계 부스트",
                    "value": score_components["relation_boost"],
                    "description": "법령 간 참조 관계 등에 따른 가산점"
                })
            
            # 랭킹 요인 설명
            intent = context.get("intent", "DEFAULT")
            if intent == "EXACT_ARTICLE":
                explanation["ranking_factors"].append("정확한 조문 검색 의도로 키워드 매칭 우선")
            elif intent == "DEFINITION":
                explanation["ranking_factors"].append("정의 검색 의도로 의미적 유사성 우선")
            elif intent == "PROCEDURE":
                explanation["ranking_factors"].append("절차 검색 의도로 실무 조항 우선")
            
            final_rank = result.get("final_rank", result.get("rank", 0))
            if final_rank <= 3:
                explanation["ranking_factors"].append("상위 결과로 높은 관련성 보장")
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"점수 설명 생성 중 오류: {e}")
            return {"total_score": 0.0, "components": [], "ranking_factors": []}
    
    def _generate_highlighted_content(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """하이라이트된 내용 생성"""
        try:
            content = result.get("content", "")
            title = result.get("title", "")
            
            highlights = {
                "content": content,
                "title": title,
                "highlighted_content": content,
                "highlighted_title": title,
                "highlight_spans": []
            }
            
            if not self.config.enable_query_highlight:
                return highlights
            
            # 쿼리 용어 하이라이트
            query_terms = query.split()
            highlighted_content = content
            highlighted_title = title
            
            for term in query_terms:
                if len(term) >= 2:  # 2글자 이상만 하이라이트
                    # 내용 하이라이트
                    pattern = re.compile(f'({re.escape(term)})', re.IGNORECASE)
                    highlighted_content = pattern.sub(
                        r'<mark class="query-highlight">\1</mark>', 
                        highlighted_content
                    )
                    
                    # 제목 하이라이트
                    highlighted_title = pattern.sub(
                        r'<mark class="query-highlight">\1</mark>',
                        highlighted_title
                    )
                    
                    # 하이라이트 위치 기록
                    for match in pattern.finditer(content):
                        highlights["highlight_spans"].append({
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
                            "type": "query_term"
                        })
            
            # 조문 번호 하이라이트
            if self.config.enable_pattern_highlight:
                article_pattern = re.compile(r'(제\s*\d+\s*조)', re.IGNORECASE)
                highlighted_content = article_pattern.sub(
                    r'<mark class="article-highlight">\1</mark>',
                    highlighted_content
                )
            
            # 법령 전문 용어 하이라이트
            if self.config.enable_legal_term_highlight:
                for main_term in self.legal_terms.keys():
                    if main_term in query:
                        term_pattern = re.compile(f'({re.escape(main_term)})', re.IGNORECASE)
                        highlighted_content = term_pattern.sub(
                            r'<mark class="legal-term-highlight">\1</mark>',
                            highlighted_content
                        )
            
            highlights["highlighted_content"] = highlighted_content
            highlights["highlighted_title"] = highlighted_title
            
            # 하이라이트 요약 생성
            highlights["summary"] = self._generate_highlight_summary(highlights["highlight_spans"], content)
            
            return highlights
            
        except Exception as e:
            self.logger.error(f"하이라이트 생성 중 오류: {e}")
            return {"content": result.get("content", ""), "title": result.get("title", "")}
    
    def _generate_highlight_summary(self, highlight_spans: List[Dict], content: str) -> List[Dict[str, str]]:
        """하이라이트 요약 생성"""
        try:
            summary = []
            
            for span in highlight_spans[:3]:  # 상위 3개만
                start = span["start"]
                end = span["end"]
                
                # 컨텍스트 윈도우 적용
                context_start = max(0, start - self.config.context_window)
                context_end = min(len(content), end + self.config.context_window)
                
                context = content[context_start:context_end]
                
                # 앞뒤 생략 표시
                if context_start > 0:
                    context = "..." + context
                if context_end < len(content):
                    context = context + "..."
                
                summary.append({
                    "highlighted_term": span["text"],
                    "context": context,
                    "position": f"위치 {start}-{end}"
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"하이라이트 요약 생성 중 오류: {e}")
            return []
    
    def _generate_structure_explanation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """법령 구조 설명 생성"""
        try:
            explanation = {
                "position": "",
                "hierarchy": "",
                "navigation": "",
                "legal_significance": ""
            }
            
            # 현재 위치 설명
            hierarchy_path = result.get("hierarchy_path", "")
            hierarchy_level = result.get("hierarchy_level", 0)
            node_type = result.get("node_type", "")
            
            if hierarchy_path:
                explanation["position"] = f"법령 구조상 위치: {hierarchy_path}"
            
            # 위계 레벨 설명
            level_descriptions = {
                0: "최상위 (법령/장)",
                1: "상위 (장/절)",
                2: "중위 (조문)",
                3: "하위 (항)",
                4: "세부 (호)",
                5: "상세 (목)"
            }
            
            explanation["hierarchy"] = f"위계 레벨 {hierarchy_level} - {level_descriptions.get(hierarchy_level, '기타')}"
            
            # 법적 중요도 설명
            if node_type == "article":
                explanation["legal_significance"] = "법령의 핵심 조문으로 높은 법적 효력"
            elif node_type == "paragraph":
                explanation["legal_significance"] = "조문의 구체적 내용을 규정하는 항"
            elif node_type == "chapter":
                explanation["legal_significance"] = "법령의 대분류를 나타내는 장"
            elif node_type == "section":
                explanation["legal_significance"] = "장 하위의 중분류를 나타내는 절"
            else:
                explanation["legal_significance"] = "법령의 세부 내용"
            
            # 탐색 가이드
            navigation_guides = []
            
            if hierarchy_level > 0:
                navigation_guides.append("상위 조항에서 전체 맥락 확인 가능")
            
            # 관련 조항이 있는지 확인
            if result.get("related_paragraphs"):
                related_count = len(result["related_paragraphs"])
                navigation_guides.append(f"동일 조문 내 관련 항 {related_count}개 추가 확인 가능")
            
            explanation["navigation"] = " | ".join(navigation_guides) if navigation_guides else "단독 조항"
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"구조 설명 생성 중 오류: {e}")
            return {"position": "구조 정보 없음"}
    
    def _generate_relevance_explanation(self, result: Dict[str, Any], query: str) -> Dict[str, str]:
        """관련성 설명 생성"""
        try:
            explanation = {
                "direct_relevance": "",
                "contextual_relevance": "",
                "practical_relevance": ""
            }
            
            content = result.get("content", "")
            
            # 직접적 관련성
            query_terms = query.split()
            matched_terms = [term for term in query_terms if term in content]
            
            if matched_terms:
                explanation["direct_relevance"] = f"검색어 '{', '.join(matched_terms)}'와 직접 관련"
            else:
                explanation["direct_relevance"] = "의미적 유사성을 통한 관련성"
            
            # 맥락적 관련성
            article_number = result.get("article_number", "")
            law_title = result.get("law_title", "")
            
            context_factors = []
            if article_number:
                context_factors.append(f"{article_number} 조문")
            if law_title:
                context_factors.append(f"{law_title}")
            
            if context_factors:
                explanation["contextual_relevance"] = f"{' '.join(context_factors)} 내에서의 관련성"
            
            # 실무적 관련성
            practical_keywords = ["절차", "방법", "요건", "기준", "의무", "권리"]
            found_practical = [kw for kw in practical_keywords if kw in content]
            
            if found_practical:
                explanation["practical_relevance"] = f"실무 관련 키워드 포함: {', '.join(found_practical)}"
            else:
                explanation["practical_relevance"] = "법리적 관련성"
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"관련성 설명 생성 중 오류: {e}")
            return {"direct_relevance": "관련성 분석 불가"}
    
    def _generate_recommendation_reason(self, result: Dict[str, Any], 
                                      rank: int, context: Dict[str, Any]) -> Dict[str, str]:
        """추천 이유 설명 생성"""
        try:
            explanation = {
                "ranking_reason": "",
                "strength": "",
                "usage_suggestion": ""
            }
            
            # 순위별 추천 이유
            if rank == 0:  # 1위
                explanation["ranking_reason"] = "가장 높은 관련성과 정확성을 보여 최우선 추천"
                explanation["strength"] = "매우 강한 추천"
            elif rank <= 2:  # 2-3위
                explanation["ranking_reason"] = "높은 관련성을 보여 우선 검토 권장"
                explanation["strength"] = "강한 추천"
            elif rank <= 4:  # 4-5위
                explanation["ranking_reason"] = "상당한 관련성을 보여 참고용으로 검토 권장"
                explanation["strength"] = "보통 추천"
            else:
                explanation["ranking_reason"] = "부분적 관련성을 보여 추가 참고용"
                explanation["strength"] = "참고용 추천"
            
            # 사용 제안
            intent = context.get("intent", "DEFAULT")
            node_type = result.get("node_type", "")
            
            usage_suggestions = []
            
            if intent == "EXACT_ARTICLE" and node_type == "article":
                usage_suggestions.append("정확한 조문 확인")
            elif intent == "DEFINITION":
                usage_suggestions.append("용어 정의 확인")
            elif intent == "PROCEDURE":
                usage_suggestions.append("절차 및 방법 확인")
            
            if result.get("related_paragraphs"):
                usage_suggestions.append("관련 세부 조항 함께 검토")
            
            if result.get("hierarchy_context"):
                usage_suggestions.append("상하위 조항과 연계하여 해석")
            
            explanation["usage_suggestion"] = " | ".join(usage_suggestions) if usage_suggestions else "전체 내용 검토"
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"추천 이유 생성 중 오류: {e}")
            return {"ranking_reason": "추천 이유 분석 불가"}
    
    def _generate_user_friendly_summary(self, result: Dict[str, Any], 
                                      matching_explanation: Dict[str, Any], 
                                      rank: int) -> Dict[str, str]:
        """사용자 친화적 요약 생성"""
        try:
            summary = {
                "one_line": "",
                "key_points": [],
                "action_guide": ""
            }
            
            # 한 줄 요약
            article_number = result.get("article_number", "")
            title = result.get("title", "")
            match_types = matching_explanation.get("match_type", [])
            
            if article_number and title:
                primary_match = match_types[0] if match_types else "관련성"
                summary["one_line"] = f"{article_number} {title} - {primary_match} 확인됨"
            else:
                summary["one_line"] = f"법령 조항 - {'정확한 매칭' if rank == 0 else '관련 내용'} 발견"
            
            # 핵심 포인트
            key_points = []
            
            if "직접 키워드 매칭" in match_types:
                key_points.append("🎯 검색어와 정확히 일치")
            
            if "법령 패턴 매칭" in match_types:
                key_points.append("📋 조문 번호 정확 매칭")
            
            if "의미적 유사성 매칭" in match_types:
                key_points.append("🧠 AI 기반 의미 분석")
            
            if result.get("final_rerank_score", 0) > 0.8:
                key_points.append("⭐ 높은 관련성 점수")
            
            if result.get("related_paragraphs"):
                key_points.append(f"📚 관련 조항 {len(result['related_paragraphs'])}개 추가")
            
            summary["key_points"] = key_points
            
            # 행동 가이드
            if rank == 0:
                summary["action_guide"] = "✅ 우선적으로 검토하세요"
            elif rank <= 2:
                summary["action_guide"] = "📖 중요한 내용이니 반드시 확인하세요"
            else:
                summary["action_guide"] = "🔍 참고용으로 검토해보세요"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"사용자 친화적 요약 생성 중 오류: {e}")
            return {"one_line": "검색 결과", "key_points": [], "action_guide": "검토 필요"}
    
    def _add_overall_explanation(self, results: List[Dict[str, Any]], 
                               query: str, context: Dict[str, Any]):
        """전체 결과 요약 설명 추가"""
        try:
            if not results:
                return
            
            overall_explanation = {
                "search_summary": {
                    "total_results": len(results),
                    "search_strategy": context.get("strategy", "balanced"),
                    "intent_detected": context.get("intent", "DEFAULT"),
                    "query_complexity": "simple" if len(query.split()) <= 3 else "complex"
                },
                "result_distribution": {},
                "search_tips": [],
                "related_searches": []
            }
            
            # 결과 분포 분석
            law_distribution = defaultdict(int)
            level_distribution = defaultdict(int)
            
            for result in results:
                law_title = result.get("law_title", "기타")
                hierarchy_level = result.get("hierarchy_level", 0)
                
                law_distribution[law_title] += 1
                level_distribution[hierarchy_level] += 1
            
            overall_explanation["result_distribution"] = {
                "by_law": dict(law_distribution),
                "by_level": dict(level_distribution)
            }
            
            # 검색 팁 생성
            tips = []
            intent = context.get("intent", "DEFAULT")
            
            if intent == "EXACT_ARTICLE":
                tips.append("💡 정확한 조문 검색 시 조문 번호를 포함하면 더 정확한 결과를 얻을 수 있습니다")
            elif intent == "DEFINITION":
                tips.append("💡 용어 정의 검색 시 '정의' 또는 '의미' 키워드를 추가하면 도움됩니다")
            
            if len(results) > 10:
                tips.append("💡 결과가 많을 때는 검색어를 더 구체적으로 입력해보세요")
            
            overall_explanation["search_tips"] = tips
            
            # 관련 검색 제안
            related_searches = []
            
            # 주요 법령 기반 제안
            main_law = max(law_distribution, key=law_distribution.get) if law_distribution else None
            if main_law and main_law != "기타":
                related_searches.append(f"{main_law} 전체 조문")
            
            # 검색어 기반 제안
            query_terms = query.split()
            if len(query_terms) > 1:
                for term in query_terms:
                    if len(term) > 1:
                        related_searches.append(f"{term} 관련 규정")
            
            overall_explanation["related_searches"] = related_searches[:3]
            
            # 첫 번째 결과에 전체 설명 추가
            if results:
                results[0]["overall_explanation"] = overall_explanation
            
        except Exception as e:
            self.logger.error(f"전체 결과 설명 추가 중 오류: {e}")


# 전역 인스턴스
_result_explainer = None

def get_result_explainer() -> SearchResultExplainer:
    """전역 결과 설명기 인스턴스 조회"""
    global _result_explainer
    if _result_explainer is None:
        _result_explainer = SearchResultExplainer()
    return _result_explainer
