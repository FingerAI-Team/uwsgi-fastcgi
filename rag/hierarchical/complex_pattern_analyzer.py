"""
Phase 3: 복합 패턴 분석기

복합 패턴을 분석하여 계층 구조, 연속성, 연관성을 파악합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .data_structures import HeaderInfo, PatternAnalysisResult

class PatternComplexity(Enum):
    """패턴 복잡도 열거형"""
    SIMPLE = "simple"           # 단일 패턴
    NESTED = "nested"           # 중첩 패턴
    CONTINUOUS = "continuous"   # 연속 패턴
    MIXED = "mixed"            # 혼합 패턴
    HIERARCHICAL = "hierarchical"  # 계층적 패턴

class PatternRelation(Enum):
    """패턴 간 관계 열거형"""
    PARENT = "parent"           # 부모-자식
    SIBLING = "sibling"         # 형제 관계
    CONTINUATION = "continuation"  # 연속 관계
    REFERENCE = "reference"     # 참조 관계
    INDEPENDENT = "independent" # 독립적

@dataclass
class PatternHierarchy:
    """패턴 계층 정보"""
    pattern: HeaderInfo
    level: int                  # 계층 레벨 (0: 최상위)
    parent: Optional['PatternHierarchy'] = None
    children: List['PatternHierarchy'] = None
    siblings: List['PatternHierarchy'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.siblings is None:
            self.siblings = []

@dataclass
class ComplexPatternAnalysis:
    """복합 패턴 분석 결과"""
    patterns: List[PatternHierarchy]
    complexity: PatternComplexity
    hierarchy_depth: int
    continuous_ranges: List[Dict[str, Any]]
    pattern_relations: Dict[str, List[PatternRelation]]
    analysis_notes: str

class ComplexPatternAnalyzer:
    """복합 패턴 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔧 복합 패턴 분석기 초기화")
        
        # 계층 레벨 정의
        self.hierarchy_levels = {
            "chapter": 1,      # 장
            "section": 2,      # 절
            "division": 3,     # 관
            "article": 4,      # 조
            "paragraph": 5,    # 항
            "subparagraph": 6, # 호
            "item": 7          # 목
        }
        
        # 연속 패턴 정의
        self.continuous_patterns = [
            r"제(\d+)조부터\s+제(\d+)조까지",
            r"제(\d+)장부터\s+제(\d+)장까지",
            r"제(\d+)절부터\s+제(\d+)절까지",
            r"(\d+)항부터\s+(\d+)항까지"
        ]
        
        self.logger.info("✅ 복합 패턴 분석기 초기화 완료")
    
    def analyze_complex_patterns(self, analysis_result: PatternAnalysisResult) -> ComplexPatternAnalysis:
        """
        복합 패턴 분석
        
        Args:
            analysis_result: 기본 패턴 분석 결과
            
        Returns:
            복합 패턴 분석 결과
        """
        try:
            self.logger.info(f"🔍 복합 패턴 분석 시작: {len(analysis_result.patterns)}개 패턴")
            
            # 1. 계층 구조 분석
            hierarchy = self._build_hierarchy(analysis_result.patterns)
            
            # 2. 복잡도 판별
            complexity = self._determine_complexity(hierarchy)
            
            # 3. 연속 패턴 분석
            continuous_ranges = self._analyze_continuous_patterns(analysis_result)
            
            # 4. 패턴 간 관계 분석
            pattern_relations = self._analyze_pattern_relations(hierarchy)
            
            # 5. 분석 결과 생성
            result = ComplexPatternAnalysis(
                patterns=hierarchy,
                complexity=complexity,
                hierarchy_depth=self._calculate_hierarchy_depth(hierarchy),
                continuous_ranges=continuous_ranges,
                pattern_relations=pattern_relations,
                analysis_notes=f"복합 패턴 분석 완료: {complexity.value}"
            )
            
            self.logger.info(f"✅ 복합 패턴 분석 완료: {complexity.value}, 계층 깊이: {result.hierarchy_depth}")
            return result
            
        except Exception as e:
            self.logger.error(f"복합 패턴 분석 중 오류: {e}")
            # 기본 분석 결과 반환
            return ComplexPatternAnalysis(
                patterns=[],
                complexity=PatternComplexity.SIMPLE,
                hierarchy_depth=0,
                continuous_ranges=[],
                pattern_relations={},
                analysis_notes=f"분석 실패: {e}"
            )
    
    def _build_hierarchy(self, patterns: List[HeaderInfo]) -> List[PatternHierarchy]:
        """계층 구조 구축"""
        try:
            # 패턴을 계층 레벨 순으로 정렬
            sorted_patterns = sorted(patterns, key=lambda p: self._get_pattern_level(p.type))
            
            hierarchy_list = []
            current_parents = {}  # 각 레벨별 현재 부모
            
            for pattern in sorted_patterns:
                level = self._get_pattern_level(pattern.type)
                hierarchy_node = PatternHierarchy(pattern=pattern, level=level)
                
                # 부모 찾기
                parent_level = level - 1
                if parent_level in current_parents:
                    parent = current_parents[parent_level]
                    hierarchy_node.parent = parent
                    parent.children.append(hierarchy_node)
                
                # 형제 찾기
                if level in current_parents:
                    sibling = current_parents[level]
                    hierarchy_node.siblings.append(sibling)
                    sibling.siblings.append(hierarchy_node)
                
                # 현재 레벨의 부모 업데이트
                current_parents[level] = hierarchy_node
                hierarchy_list.append(hierarchy_node)
                
                # 하위 레벨 부모들 초기화
                for l in range(level + 1, max(self.hierarchy_levels.values()) + 1):
                    if l in current_parents:
                        del current_parents[l]
            
            return hierarchy_list
            
        except Exception as e:
            self.logger.error(f"계층 구조 구축 중 오류: {e}")
            return []
    
    def _get_pattern_level(self, pattern_type: str) -> int:
        """패턴 타입의 계층 레벨 반환"""
        return self.hierarchy_levels.get(pattern_type, 0)
    
    def _determine_complexity(self, hierarchy: List[PatternHierarchy]) -> PatternComplexity:
        """패턴 복잡도 판별"""
        try:
            if not hierarchy:
                return PatternComplexity.SIMPLE
            
            # 계층 깊이 확인
            max_depth = max(node.level for node in hierarchy)
            
            # 중첩 패턴 확인
            has_nested = any(len(node.children) > 0 for node in hierarchy)
            
            # 연속 패턴 확인
            has_continuous = any(len(node.siblings) > 0 for node in hierarchy)
            
            # 복잡도 판별
            if max_depth <= 1 and not has_nested and not has_continuous:
                return PatternComplexity.SIMPLE
            elif has_nested and not has_continuous:
                return PatternComplexity.NESTED
            elif has_continuous and not has_nested:
                return PatternComplexity.CONTINUOUS
            elif has_nested and has_continuous:
                return PatternComplexity.MIXED
            elif max_depth >= 3:
                return PatternComplexity.HIERARCHICAL
            else:
                return PatternComplexity.MIXED
                
        except Exception as e:
            self.logger.error(f"복잡도 판별 중 오류: {e}")
            return PatternComplexity.SIMPLE
    
    def _analyze_continuous_patterns(self, analysis_result: PatternAnalysisResult) -> List[Dict[str, Any]]:
        """연속 패턴 분석"""
        try:
            continuous_ranges = []
            
            # 라인별로 연속 패턴 검사
            for line_num, patterns in analysis_result.line_analysis.items():
                line_text = patterns[0].line_text if patterns else ""
                
                for pattern in patterns:
                    # 연속 패턴 매칭
                    for continuous_pattern in self.continuous_patterns:
                        import re
                        matches = re.finditer(continuous_pattern, line_text)
                        
                        for match in matches:
                            range_info = {
                                "pattern_type": pattern.type,
                                "start_value": match.group(1),
                                "end_value": match.group(2),
                                "line_number": line_num,
                                "line_text": line_text,
                                "full_match": match.group(0)
                            }
                            continuous_ranges.append(range_info)
            
            self.logger.debug(f"연속 패턴 {len(continuous_ranges)}개 발견")
            return continuous_ranges
            
        except Exception as e:
            self.logger.error(f"연속 패턴 분석 중 오류: {e}")
            return []
    
    def _analyze_pattern_relations(self, hierarchy: List[PatternHierarchy]) -> Dict[str, List[PatternRelation]]:
        """패턴 간 관계 분석"""
        try:
            pattern_relations = {}
            
            for node in hierarchy:
                pattern_id = f"{node.pattern.type}_{node.pattern.text}"
                relations = []
                
                # 부모 관계
                if node.parent:
                    relations.append(PatternRelation.PARENT)
                
                # 자식 관계
                if node.children:
                    relations.append(PatternRelation.PARENT)
                
                # 형제 관계
                if node.siblings:
                    relations.append(PatternRelation.SIBLING)
                
                # 연속 관계 (같은 타입의 연속된 패턴)
                if self._has_continuation_relation(node):
                    relations.append(PatternRelation.CONTINUATION)
                
                # 참조 관계 (다른 패턴을 참조하는 경우)
                if self._has_reference_relation(node):
                    relations.append(PatternRelation.REFERENCE)
                
                # 독립적 관계 (아무 관계도 없는 경우)
                if not relations:
                    relations.append(PatternRelation.INDEPENDENT)
                
                pattern_relations[pattern_id] = relations
            
            return pattern_relations
            
        except Exception as e:
            self.logger.error(f"패턴 관계 분석 중 오류: {e}")
            return {}
    
    def _has_continuation_relation(self, node: PatternHierarchy) -> bool:
        """연속 관계가 있는지 확인"""
        try:
            if not node.siblings:
                return False
            
            # 같은 타입의 연속된 패턴 확인
            for sibling in node.siblings:
                if (sibling.pattern.type == node.pattern.type and
                    self._is_consecutive_pattern(node.pattern, sibling.pattern)):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"연속 관계 확인 중 오류: {e}")
            return False
    
    def _is_consecutive_pattern(self, pattern1: HeaderInfo, pattern2: HeaderInfo) -> bool:
        """두 패턴이 연속적인지 확인"""
        try:
            # 숫자 패턴의 경우 연속성 확인
            if pattern1.groups and pattern2.groups:
                try:
                    num1 = int(pattern1.groups[0])
                    num2 = int(pattern2.groups[0])
                    return abs(num2 - num1) == 1
                except (ValueError, IndexError):
                    pass
            
            # 한글 패턴의 경우 연속성 확인
            if pattern1.groups and pattern2.groups:
                try:
                    char1 = pattern1.groups[0]
                    char2 = pattern2.groups[0]
                    if len(char1) == 1 and len(char2) == 1:
                        # 가나다 순서로 연속적인지 확인
                        return ord(char2) - ord(char1) == 1
                except (IndexError, TypeError):
                    pass
            
            return False
            
        except Exception as e:
            self.logger.error(f"연속 패턴 확인 중 오류: {e}")
            return False
    
    def _has_reference_relation(self, node: PatternHierarchy) -> bool:
        """참조 관계가 있는지 확인"""
        try:
            # 패턴 텍스트에 다른 패턴 참조가 포함되어 있는지 확인
            text = node.pattern.text.lower()
            reference_indicators = ["참조", "참고", "관련", "연결", "의", "에서", "까지"]
            
            return any(indicator in text for indicator in reference_indicators)
            
        except Exception as e:
            self.logger.error(f"참조 관계 확인 중 오류: {e}")
            return False
    
    def _calculate_hierarchy_depth(self, hierarchy: List[PatternHierarchy]) -> int:
        """계층 깊이 계산"""
        try:
            if not hierarchy:
                return 0
            
            return max(node.level for node in hierarchy)
            
        except Exception as e:
            self.logger.error(f"계층 깊이 계산 중 오류: {e}")
            return 0
    
    def get_pattern_summary(self, analysis: ComplexPatternAnalysis) -> Dict[str, Any]:
        """패턴 분석 요약 반환"""
        try:
            return {
                "total_patterns": len(analysis.patterns),
                "complexity": analysis.complexity.value,
                "hierarchy_depth": analysis.hierarchy_depth,
                "continuous_ranges": len(analysis.continuous_ranges),
                "pattern_types": self._count_pattern_types(analysis.patterns),
                "relation_summary": self._summarize_relations(analysis.pattern_relations)
            }
            
        except Exception as e:
            self.logger.error(f"패턴 요약 생성 중 오류: {e}")
            return {}
    
    def _count_pattern_types(self, patterns: List[PatternHierarchy]) -> Dict[str, int]:
        """패턴 타입별 개수 계산"""
        try:
            type_count = {}
            for node in patterns:
                pattern_type = node.pattern.type
                type_count[pattern_type] = type_count.get(pattern_type, 0) + 1
            return type_count
            
        except Exception as e:
            self.logger.error(f"패턴 타입 개수 계산 중 오류: {e}")
            return {}
    
    def _summarize_relations(self, pattern_relations: Dict[str, List[PatternRelation]]) -> Dict[str, int]:
        """관계 유형별 개수 요약"""
        try:
            relation_count = {}
            for relations in pattern_relations.values():
                for relation in relations:
                    relation_type = relation.value
                    relation_count[relation_type] = relation_count.get(relation_type, 0) + 1
            return relation_count
            
        except Exception as e:
            self.logger.error(f"관계 요약 생성 중 오류: {e}")
            return {}
