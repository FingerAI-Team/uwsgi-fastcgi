"""
위계형 RAG 시스템 설정 로더

모든 하드코딩된 설정값들을 JSON 파일에서 로드하여 관리합니다.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class HierarchicalConfigLoader:
    """위계형 RAG 설정 로더"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: 설정 파일 디렉토리 경로
        """
        self.logger = logging.getLogger(__name__)
        
        if config_dir is None:
            # 현재 파일 기준으로 config 디렉토리 경로 설정
            current_dir = Path(__file__).parent
            self.config_dir = current_dir
        else:
            self.config_dir = Path(config_dir)
        
        # 설정 파일들
        self.config_files = {
            "hierarchical": "hierarchical_config.json",
            "thesaurus": "legal_thesaurus.json", 
            "patterns": "legal_patterns.json",
            "intent_keywords": "intent_keywords.json",
            "dynamic_scoring": "dynamic_scoring.json",
            "weight_presets": "weight_presets.json",
            "date_extraction": "date_extraction_patterns.json"
        }
        
        # 로드된 설정 캐시
        self._config_cache = {}
        
        # 설정 로드
        self._load_all_configs()
    
    def _load_all_configs(self):
        """모든 설정 파일 로드"""
        try:
            for config_name, filename in self.config_files.items():
                config_path = self.config_dir / filename
                
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self._config_cache[config_name] = json.load(f)
                    self.logger.info(f"설정 파일 로드 완료: {filename}")
                else:
                    self.logger.warning(f"설정 파일 없음: {filename}")
                    self._config_cache[config_name] = {}
            
            self.logger.info(f"총 {len(self._config_cache)}개 설정 파일 로드 완료")
            
        except Exception as e:
            self.logger.error(f"설정 파일 로드 중 오류: {e}")
            # 기본값으로 폴백
            self._load_default_configs()
    
    def _load_default_configs(self):
        """기본 설정값 로드 (파일이 없을 경우)"""
        self.logger.info("기본 설정값으로 초기화")
        
        self._config_cache = {
            "hierarchical": self._get_default_hierarchical_config(),
            "thesaurus": self._get_default_thesaurus_config(),
            "patterns": self._get_default_patterns_config()
        }
    
    # ==================== 설정 조회 메서드들 ====================
    
    def get_search_strategies(self) -> Dict[str, float]:
        """검색 전략별 가중치 조회"""
        return self._config_cache.get("hierarchical", {}).get("search_settings", {}).get("strategies", {
            "direct_matching": 0.4,
            "hierarchy_reasoning": 0.3,
            "semantic_expansion": 0.2,
            "structural_traversal": 0.1
        })
    
    def get_confidence_thresholds(self) -> Dict[str, float]:
        """신뢰도 임계값 조회"""
        return self._config_cache.get("hierarchical", {}).get("search_settings", {}).get("confidence_thresholds", {
            "semantic_similarity": 0.7,
            "article_matching": 0.9,
            "keyword_matching": 0.7
        })
    
    def get_legal_hierarchy_weights(self) -> Dict[str, float]:
        """법령 위계별 가중치 조회"""
        return self._config_cache.get("hierarchical", {}).get("legal_hierarchy_weights", {
            "article": 1.3,
            "paragraph": 1.2,
            "chapter": 1.0,
            "section": 1.1
        })
    
    def get_law_type_weights(self) -> Dict[str, float]:
        """법령 유형별 가중치 조회"""
        return self._config_cache.get("hierarchical", {}).get("law_type_weights", {
            "법률": 0.3,
            "시행령": 0.2,
            "규칙": 0.1
        })
    
    def get_legal_thesaurus(self) -> Dict[str, Any]:
        """법령 시소러스 조회"""
        return self._config_cache.get("thesaurus", {}).get("legal_keywords", {})
    
    def get_legal_concepts(self) -> Dict[str, Any]:
        """법령 개념 관계 조회"""
        return self._config_cache.get("thesaurus", {}).get("legal_concepts", {})
    
    def get_legal_relation_patterns(self) -> Dict[str, Any]:
        """법령 관계 패턴 조회"""
        return self._config_cache.get("thesaurus", {}).get("legal_relation_patterns", {})
    
    def get_text_patterns(self) -> Dict[str, Any]:
        """텍스트 패턴 조회"""
        return self._config_cache.get("patterns", {}).get("text_patterns", {})
    
    def get_hierarchy_patterns(self) -> Dict[str, Any]:
        """위계 구조 패턴 조회"""
        return self._config_cache.get("patterns", {}).get("hierarchy_patterns", {})
    
    def get_metadata_detection_patterns(self) -> Dict[str, Any]:
        """메타데이터 감지 패턴 조회"""
        return self._config_cache.get("patterns", {}).get("metadata_detection", {})
    
    def get_search_limits(self) -> Dict[str, Any]:
        """검색 제한 설정 조회"""
        return self._config_cache.get("hierarchical", {}).get("search_settings", {}).get("search_limits", {
            "max_traversal_depth": 5,
            "max_expansion_results": 50,
            "top_k_multiplier": 3
        })
    
    def get_reference_strengths(self) -> Dict[str, float]:
        """참조 강도 설정 조회"""
        return self._config_cache.get("hierarchical", {}).get("reference_strengths", {
            "referring_to_target": 0.8,
            "referenced_by_target": 0.9,
            "structural_relation": 0.6
        })
    
    # ==================== 새로 추가된 설정 조회 메서드들 ====================
    
    def get_intent_keywords(self) -> Dict[str, Any]:
        """의도 감지 키워드 설정 조회"""
        return self._config_cache.get("intent_keywords", {}).get("intent_patterns", {})
    
    def get_intent_priorities(self) -> Dict[str, int]:
        """의도별 우선순위 조회"""
        intent_patterns = self.get_intent_keywords()
        priorities = {}
        for intent, config in intent_patterns.items():
            priorities[intent] = config.get("priority", 99)
        return priorities
    
    def get_dynamic_scoring_config(self) -> Dict[str, Any]:
        """동적 점수 설정 조회"""
        return self._config_cache.get("dynamic_scoring", {})
    
    def get_query_length_adjustments(self) -> Dict[str, Any]:
        """쿼리 길이별 조정 설정 조회"""
        return self.get_dynamic_scoring_config().get("query_length_adjustments", {})
    
    def get_exact_matching_bonuses(self) -> Dict[str, Any]:
        """정확 매칭 보너스 설정 조회"""
        return self.get_dynamic_scoring_config().get("exact_matching_bonuses", {})
    
    def get_field_matching_bonuses(self) -> Dict[str, Any]:
        """필드 매칭 보너스 설정 조회"""
        return self.get_dynamic_scoring_config().get("field_matching_bonuses", {})
    
    def get_weight_presets(self) -> Dict[str, Any]:
        """가중치 프리셋 조회"""
        return self._config_cache.get("weight_presets", {}).get("intent_based_weights", {})
    
    def get_intent_weights(self, intent: str) -> Dict[str, float]:
        """특정 의도에 대한 가중치 조회"""
        presets = self.get_weight_presets()
        return presets.get(intent, presets.get("DEFAULT", {
            "dense_weight": 0.6,
            "bm25_weight": 0.4,
            "article_hint_bonus": 0.1,
            "lawname_hint_bonus": 0.1,
            "sibling_bonus": 0.05
        }))
    
    def get_scoring_patterns(self) -> Dict[str, Any]:
        """패턴별 점수 설정 조회"""
        return self._config_cache.get("patterns", {}).get("scoring_patterns", {})
    
    # ==================== 날짜 추출 설정 ====================
    
    def get_date_extraction_patterns(self) -> List[Dict[str, Any]]:
        """날짜 추출 패턴 조회"""
        return self._config_cache.get("date_extraction", {}).get("date_patterns", [])
    
    def get_date_extraction_strategy(self) -> Dict[str, Any]:
        """날짜 추출 전략 조회"""
        return self._config_cache.get("date_extraction", {}).get("extraction_strategy", {})
    
    def get_date_priority_rules(self) -> List[Dict[str, Any]]:
        """날짜 우선순위 규칙 조회"""
        return self._config_cache.get("date_extraction", {}).get("priority_rules", [])
    
    # ==================== 동적 설정 업데이트 ====================
    
    def update_search_strategy_weight(self, strategy: str, weight: float):
        """검색 전략 가중치 동적 업데이트"""
        try:
            if "hierarchical" not in self._config_cache:
                self._config_cache["hierarchical"] = {}
            
            if "search_settings" not in self._config_cache["hierarchical"]:
                self._config_cache["hierarchical"]["search_settings"] = {}
            
            if "strategies" not in self._config_cache["hierarchical"]["search_settings"]:
                self._config_cache["hierarchical"]["search_settings"]["strategies"] = {}
            
            self._config_cache["hierarchical"]["search_settings"]["strategies"][strategy] = weight
            self.logger.info(f"검색 전략 가중치 업데이트: {strategy} = {weight}")
            
        except Exception as e:
            self.logger.error(f"검색 전략 가중치 업데이트 실패: {e}")
    
    def update_confidence_threshold(self, threshold_name: str, value: float):
        """신뢰도 임계값 동적 업데이트"""
        try:
            if "hierarchical" not in self._config_cache:
                self._config_cache["hierarchical"] = {}
            
            if "search_settings" not in self._config_cache["hierarchical"]:
                self._config_cache["hierarchical"]["search_settings"] = {}
            
            if "confidence_thresholds" not in self._config_cache["hierarchical"]["search_settings"]:
                self._config_cache["hierarchical"]["search_settings"]["confidence_thresholds"] = {}
            
            self._config_cache["hierarchical"]["search_settings"]["confidence_thresholds"][threshold_name] = value
            self.logger.info(f"신뢰도 임계값 업데이트: {threshold_name} = {value}")
            
        except Exception as e:
            self.logger.error(f"신뢰도 임계값 업데이트 실패: {e}")
    
    def add_legal_keyword(self, keyword: str, synonyms: list, related_terms: list = None, domain: str = None):
        """법령 키워드 동적 추가"""
        try:
            if "thesaurus" not in self._config_cache:
                self._config_cache["thesaurus"] = {}
            
            if "legal_keywords" not in self._config_cache["thesaurus"]:
                self._config_cache["thesaurus"]["legal_keywords"] = {}
            
            self._config_cache["thesaurus"]["legal_keywords"][keyword] = {
                "synonyms": synonyms,
                "related_terms": related_terms or [],
                "domain": domain or "일반"
            }
            
            self.logger.info(f"법령 키워드 추가: {keyword}")
            
        except Exception as e:
            self.logger.error(f"법령 키워드 추가 실패: {e}")
    
    def save_config_to_file(self, config_type: str):
        """설정을 파일에 저장"""
        try:
            if config_type not in self.config_files:
                self.logger.error(f"알 수 없는 설정 타입: {config_type}")
                return False
            
            config_path = self.config_dir / self.config_files[config_type]
            config_data = self._config_cache.get(config_type, {})
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"설정 파일 저장 완료: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"설정 파일 저장 실패: {e}")
            return False
    
    # ==================== 기본값 설정 ====================
    
    def _get_default_hierarchical_config(self) -> Dict[str, Any]:
        """기본 위계형 설정"""
        return {
            "search_settings": {
                "strategies": {
                    "direct_matching": 0.4,
                    "hierarchy_reasoning": 0.3,
                    "semantic_expansion": 0.2,
                    "structural_traversal": 0.1
                },
                "confidence_thresholds": {
                    "semantic_similarity": 0.7,
                    "article_matching": 0.9,
                    "keyword_matching": 0.7
                },
                "search_limits": {
                    "max_traversal_depth": 5,
                    "max_expansion_results": 50,
                    "top_k_multiplier": 3
                }
            },
            "legal_hierarchy_weights": {
                "article": 1.3,
                "paragraph": 1.2,
                "chapter": 1.0,
                "section": 1.1
            }
        }
    
    def _get_default_thesaurus_config(self) -> Dict[str, Any]:
        """기본 시소러스 설정"""
        return {
            "legal_keywords": {
                "개인정보": {
                    "synonyms": ["개인식별정보", "개인데이터"],
                    "related_terms": ["정보주체", "개인정보처리자"],
                    "domain": "개인정보보호"
                }
            }
        }
    
    def _get_default_patterns_config(self) -> Dict[str, Any]:
        """기본 패턴 설정"""
        return {
            "text_patterns": {
                "query_patterns": {
                    "article_ref": "제(\\d+)조",
                    "paragraph_ref": "제?(\\d+)항"
                }
            }
        }
    
    # ==================== 유틸리티 메서드 ====================
    
    def get_config_summary(self) -> Dict[str, Any]:
        """설정 요약 정보 조회"""
        try:
            summary = {
                "loaded_configs": list(self._config_cache.keys()),
                "search_strategies_count": len(self.get_search_strategies()),
                "thesaurus_keywords_count": len(self.get_legal_thesaurus()),
                "relation_patterns_count": len(self.get_legal_relation_patterns()),
                "config_dir": str(self.config_dir)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"설정 요약 조회 실패: {e}")
            return {}
    
    def validate_config(self) -> Dict[str, Any]:
        """설정 유효성 검증"""
        try:
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # 검색 전략 가중치 합계 확인
            strategies = self.get_search_strategies()
            total_weight = sum(strategies.values())
            
            if abs(total_weight - 1.0) > 0.01:
                validation_results["warnings"].append(
                    f"검색 전략 가중치 합계가 1.0이 아님: {total_weight:.3f}"
                )
            
            # 신뢰도 임계값 범위 확인
            thresholds = self.get_confidence_thresholds()
            for name, value in thresholds.items():
                if not (0.0 <= value <= 1.0):
                    validation_results["errors"].append(
                        f"신뢰도 임계값 범위 오류: {name} = {value} (0.0-1.0 범위 벗어남)"
                    )
                    validation_results["valid"] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"설정 검증 중 오류: {e}")
            return {"valid": False, "errors": [str(e)], "warnings": []}


# 전역 설정 로더 인스턴스
_config_loader = None

def get_config_loader() -> HierarchicalConfigLoader:
    """전역 설정 로더 인스턴스 조회"""
    global _config_loader
    if _config_loader is None:
        _config_loader = HierarchicalConfigLoader()
    return _config_loader

def reload_config():
    """설정 다시 로드"""
    global _config_loader
    _config_loader = None
    return get_config_loader()
