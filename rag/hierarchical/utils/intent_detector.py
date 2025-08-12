"""
룰 기반 의도 파악 시스템

학습 없이 간단한 패턴으로 검색 의도를 파악하여
동적 가중치를 적용합니다.
"""

import re
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from ..config.config_loader import get_config_loader


@dataclass
class IntentWeights:
    """의도별 가중치 설정"""
    dense: float
    bm25: float
    article_hint: float
    lawname_hint: float
    sibling_bonus: float


class LegalIntentDetector:
    """법령 검색 의도 감지기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_loader = get_config_loader()
        
        # 설정 파일에서 가중치 로드
        self.intent_weights = self._load_weights_from_config()
        
        # 설정 파일에서 의도 감지 패턴 로드
        self.intent_patterns = self._load_patterns_from_config()
    
    def detect_intent(self, query: str) -> str:
        """검색 의도 감지"""
        try:
            query_normalized = self._normalize_query(query)
            
            # 우선순위: EXACT_ARTICLE > SANCTION > PROCEDURE > DEFINITION > DEFAULT
            for intent in ["EXACT_ARTICLE", "SANCTION", "PROCEDURE", "DEFINITION"]:
                if self._matches_intent(query_normalized, intent):
                    self.logger.debug(f"의도 감지: {intent} for query: {query}")
                    return intent
            
            self.logger.debug(f"기본 의도 적용: DEFAULT for query: {query}")
            return "DEFAULT"
            
        except Exception as e:
            self.logger.error(f"의도 감지 중 오류: {e}")
            return "DEFAULT"
    
    def get_weights_for_intent(self, intent: str) -> IntentWeights:
        """의도별 가중치 조회"""
        return self.intent_weights.get(intent, self.intent_weights["DEFAULT"])
    
    def calculate_dynamic_score(self, features: Dict[str, Any], intent: str) -> float:
        """동적 점수 계산"""
        try:
            weights = self.get_weights_for_intent(intent)
            
            # 기본 점수 (Dense + BM25)
            dense_score = self._normalize_score(features.get("dense_score", 0.0))
            bm25_score = self._normalize_score(features.get("bm25_score", 0.0))
            
            base_score = weights.dense * dense_score + weights.bm25 * bm25_score
            
            # 패턴 보너스
            pattern_bonus = 0.0
            if features.get("has_article_hint", False):
                pattern_bonus += weights.article_hint
            
            if features.get("has_lawname_hint", False):
                pattern_bonus += weights.lawname_hint
            
            # 형제 노드 보너스 (과적합 방지)
            sibling_hits = features.get("sibling_hits", 0)
            sibling_bonus = weights.sibling_bonus * min(sibling_hits / 3.0, 1.0)
            
            # 추가 동적 신호
            length_bias = self._calculate_length_bias(features.get("query_length", 0))
            numeric_bonus = features.get("exact_numeric_bonus", 0.0)
            title_bonus = features.get("title_hit_bonus", 0.0)
            
            final_score = (base_score + pattern_bonus + sibling_bonus + 
                          length_bias + numeric_bonus + title_bonus)
            
            return min(final_score, 1.0)  # 1.0으로 캡핑
            
        except Exception as e:
            self.logger.error(f"동적 점수 계산 중 오류: {e}")
            return features.get("dense_score", 0.0)
    
    def _normalize_query(self, query: str) -> str:
        """쿼리 정규화"""
        # 공백 정규화
        normalized = re.sub(r'\s+', ' ', query.strip())
        return normalized
    
    def _matches_intent(self, query: str, intent: str) -> bool:
        """의도 매칭 확인"""
        intent_config = self.intent_patterns.get(intent, {})
        
        # 정규식 패턴 확인
        for pattern in intent_config.get("patterns", []):
            if re.search(pattern, query):
                return True
        
        # 키워드 확인
        for keyword in intent_config.get("keywords", []):
            if keyword in query:
                return True
        
        return False
    
    def _normalize_score(self, score: float) -> float:
        """점수 정규화 (0.0 ~ 1.0)"""
        return max(0.0, min(1.0, score))
    
    def _calculate_length_bias(self, query_length: int) -> float:
        """쿼리 길이 기반 편향 보정"""
        if query_length <= 3:
            return 0.05  # 짧은 쿼리: BM25 선호
        elif query_length >= 8:
            return -0.05  # 긴 쿼리: Dense 선호 (음수로 BM25 가중치 감소)
        else:
            return 0.0
    
    def extract_query_features(self, query: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """쿼리-문서 간 특징 추출"""
        try:
            features = {
                "query_length": len(query.split()),
                "has_article_hint": False,
                "has_lawname_hint": False,
                "exact_numeric_bonus": 0.0,
                "title_hit_bonus": 0.0
            }
            
            content = document.get("content", "")
            title = document.get("title", "")
            
            # 조문 패턴 매칭
            article_patterns = re.findall(r'제\s*(\d+)\s*조', query)
            for pattern in article_patterns:
                if f"제{pattern}조" in content:
                    features["has_article_hint"] = True
                    break
            
            # 법령명 매칭
            law_patterns = re.findall(r'[가-힣]{2,}법|[가-힣]{2,}규칙|[가-힣]{2,}시행령', query)
            for pattern in law_patterns:
                if pattern in content or pattern in document.get("law_title", ""):
                    features["has_lawname_hint"] = True
                    break
            
            # 정확 숫자 매칭 보너스
            query_numbers = set(re.findall(r'\d+', query))
            content_numbers = set(re.findall(r'\d+', content))
            if query_numbers & content_numbers:  # 교집합이 있으면
                features["exact_numeric_bonus"] = 0.05
            
            # 제목 히트 보너스
            query_words = set(query.split())
            title_words = set(title.split()) if title else set()
            if query_words & title_words:  # 교집합이 있으면
                features["title_hit_bonus"] = 0.05
            
            return features
            
        except Exception as e:
            self.logger.error(f"특징 추출 중 오류: {e}")
            return {"query_length": len(query.split())}
    
    def get_intent_statistics(self) -> Dict[str, Any]:
        """의도별 통계 조회"""
        return {
            "available_intents": list(self.intent_weights.keys()),
            "weight_ranges": {
                "dense": f"{min(w.dense for w in self.intent_weights.values()):.2f} ~ {max(w.dense for w in self.intent_weights.values()):.2f}",
                "bm25": f"{min(w.bm25 for w in self.intent_weights.values()):.2f} ~ {max(w.bm25 for w in self.intent_weights.values()):.2f}"
            },
            "pattern_counts": {
                intent: len(config.get("patterns", [])) + len(config.get("keywords", []))
                for intent, config in self.intent_patterns.items()
            }
        }
    
    def _load_weights_from_config(self) -> Dict[str, IntentWeights]:
        """설정 파일에서 가중치 로드"""
        try:
            weight_presets = self.config_loader.get_weight_presets()
            loaded_weights = {}
            
            for intent, config in weight_presets.items():
                loaded_weights[intent] = IntentWeights(
                    dense=config.get("dense_weight", 0.6),
                    bm25=config.get("bm25_weight", 0.4),
                    article_hint=config.get("article_hint_bonus", 0.1),
                    lawname_hint=config.get("lawname_hint_bonus", 0.1),
                    sibling_bonus=config.get("sibling_bonus", 0.05)
                )
            
            self.logger.info(f"설정에서 {len(loaded_weights)}개 의도 가중치 로드 완료")
            return loaded_weights
            
        except Exception as e:
            self.logger.error(f"가중치 설정 로드 실패, 기본값 사용: {e}")
            return self._get_default_weights()
    
    def _load_patterns_from_config(self) -> Dict[str, Dict]:
        """설정 파일에서 의도 감지 패턴 로드"""
        try:
            intent_keywords = self.config_loader.get_intent_keywords()
            loaded_patterns = {}
            
            for intent, config in intent_keywords.items():
                loaded_patterns[intent] = {
                    "patterns": config.get("regex_patterns", []),
                    "keywords": config.get("keywords", [])
                }
            
            self.logger.info(f"설정에서 {len(loaded_patterns)}개 의도 패턴 로드 완료")
            return loaded_patterns
            
        except Exception as e:
            self.logger.error(f"패턴 설정 로드 실패, 기본값 사용: {e}")
            return self._get_default_patterns()
    
    def _get_default_weights(self) -> Dict[str, IntentWeights]:
        """기본 가중치 (폴백용)"""
        return {
            "EXACT_ARTICLE": IntentWeights(0.45, 0.55, 0.25, 0.15, 0.00),
            "SANCTION": IntentWeights(0.55, 0.45, 0.10, 0.10, 0.05),
            "PROCEDURE": IntentWeights(0.60, 0.40, 0.05, 0.05, 0.10),
            "DEFINITION": IntentWeights(0.65, 0.35, 0.05, 0.05, 0.10),
            "DEFAULT": IntentWeights(0.60, 0.40, 0.10, 0.10, 0.05)
        }
    
    def _get_default_patterns(self) -> Dict[str, Dict]:
        """기본 패턴 (폴백용)"""
        return {
            "EXACT_ARTICLE": {
                "patterns": [r"제\s*\d+\s*조", r"\d+\s*항", r"[가-힣]{2,}법"],
                "keywords": []
            },
            "SANCTION": {
                "patterns": [],
                "keywords": ["벌칙", "과태료", "처벌", "벌금"]
            },
            "PROCEDURE": {
                "patterns": [],
                "keywords": ["절차", "방법", "요건", "신고"]
            },
            "DEFINITION": {
                "patterns": [r"제2조"],
                "keywords": ["정의", "란", "의미"]
            }
        }


# 전역 인스턴스
_intent_detector = None

def get_intent_detector() -> LegalIntentDetector:
    """전역 의도 감지기 인스턴스 조회"""
    global _intent_detector
    if _intent_detector is None:
        _intent_detector = LegalIntentDetector()
    return _intent_detector
