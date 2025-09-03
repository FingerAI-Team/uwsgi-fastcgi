"""
패턴 분류기

발견된 패턴들을 분석하여 헤더 vs 버퍼, 단일 vs 복합으로 분류
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from .data_structures import HeaderInfo, PatternAnalysisResult

class PatternClassifier:
    """패턴 분류기"""
    
    def __init__(self):
        self.logger = logging.getLogger('hierarchical')
        # 로거 레벨을 DEBUG로 설정
        self.logger.setLevel(logging.DEBUG)
        
        # 참조/인용 패턴 정의
        self.reference_patterns = [
            # 따옴표로 감싸진 헤더
            (r'["""](제\d+[장절관조]|\(\d+\)|①|(\d+)\.|([가-힣])\.)["""]', "quoted_header"),
            
            # 조사 뒤에 오는 헤더
            (r'(제\d+[장절관조]|\(\d+\)|①|(\d+)\.|([가-힣])\.)\s*[에의에서]', "postposition_header")
            
            # 연속 참조 (중간 공백 없음)
            (r'제\d+조제\d+항', "continuous_reference"),
            (r'제\d+장제\d+절', "continuous_reference"),
            
            # 범위 참조
            (r'제\d+조부터\s+제\d+조까지', "range_reference"),
            (r'제\d+장부터\s+제\d+장까지', "range_reference"),
        ]
        
        # 컴파일된 참조 패턴들
        self.compiled_reference_patterns = []
        for pattern, description in self.reference_patterns:
            try:
                compiled = re.compile(pattern, re.UNICODE)
                self.compiled_reference_patterns.append((compiled, description))
            except Exception as e:
                self.logger.error(f"참조 패턴 컴파일 실패: {description} - {pattern}, 오류: {e}")
        
        self.logger.info(f"✅ 패턴 분류기 초기화 완료: {len(self.compiled_reference_patterns)}개 참조 패턴")
    
    def classify_patterns(self, analysis_result: PatternAnalysisResult) -> Dict[str, Any]:
        """
        패턴들을 분류
        
        Args:
            analysis_result: 패턴 분석 결과
            
        Returns:
            분류 결과
        """
        classification_result = {
            "headers": [],      # 헤더로 분류된 패턴들
            "references": [],   # 참조/인용으로 분류된 패턴들
            "single_patterns": [],  # 단일 패턴들
            "complex_patterns": [], # 복합 패턴들
            "summary": {}
        }
        
        try:
            for pattern in analysis_result.patterns:
                self.logger.debug(f"🔍 패턴 분류 시작: '{pattern.text}' (타입: {pattern.type})")
                
                # 1단계: 참조/인용 vs 헤더 판별
                if self._is_reference_pattern(pattern):
                    classification_result["references"].append(pattern)
                    self.logger.debug(f"📌 참조/인용으로 분류: '{pattern.text}'")
                else:
                    classification_result["headers"].append(pattern)
                    self.logger.debug(f"📌 헤더로 분류: '{pattern.text}'")
            
            # 2단계: 단일 vs 복합 패턴 판별
            for line_num in analysis_result.line_analysis:
                line_patterns = analysis_result.get_line_patterns(line_num)
                
                if len(line_patterns) == 1:
                    # 단일 패턴
                    classification_result["single_patterns"].extend(line_patterns)
                elif len(line_patterns) > 1:
                    # 복합 패턴
                    classification_result["complex_patterns"].extend(line_patterns)
            
            # 3단계: 요약 정보 생성
            classification_result["summary"] = {
                "total_patterns": len(analysis_result.patterns),
                "header_count": len(classification_result["headers"]),
                "reference_count": len(classification_result["references"]),
                "single_pattern_count": len(classification_result["single_patterns"]),
                "complex_pattern_count": len(classification_result["complex_patterns"])
            }
            
            self.logger.info(f"패턴 분류 완료: {classification_result['summary']}")
            return classification_result
            
        except Exception as e:
            self.logger.error(f"패턴 분류 중 오류: {e}")
            return classification_result
    
    def _is_reference_pattern(self, pattern: HeaderInfo) -> bool:
        """
        패턴이 참조/인용 패턴인지 판별
        
        Args:
            pattern: 판별할 패턴
            
        Returns:
            참조/인용 패턴이면 True
        """
        try:
            # 헤더 텍스트만 검색 (수정된 부분)
            header_text = pattern.text
            
            self.logger.debug(f"🔍 참조 패턴 판별 시작: '{header_text}' (타입: {pattern.type})")
            
            # 참조 패턴들 체크
            for compiled_pattern, description in self.compiled_reference_patterns:
                if compiled_pattern.search(header_text):
                    self.logger.debug(f"✅ 참조 패턴 매칭: {description} - '{header_text}'")
                    self.logger.debug(f"   📍 매칭된 정규식: {compiled_pattern.pattern}")
                    return True
            
            # 추가 문맥 분석
            if self._has_postposition_after(pattern):
                self.logger.debug(f"✅ 조사 뒤에 오는 패턴: '{header_text}'")
                return True
            
            if self._is_quoted_text(pattern):
                self.logger.debug(f"✅ 따옴표로 감싸진 텍스트: '{header_text}'")
                return True
            
            self.logger.debug(f"❌ 참조 패턴 아님: '{header_text}' → 헤더로 분류")
            return False
            
        except Exception as e:
            self.logger.error(f"참조 패턴 판별 중 오류: {e}")
            return False
    
    def _has_postposition_after(self, pattern: HeaderInfo) -> bool:
        """패턴 뒤에 조사가 있는지 확인"""
        try:
            line_text = pattern.line_text
            after_pattern = line_text[pattern.end:].strip()
            
            # 조사 패턴 체크
            postposition_pattern = r'^[에의에서]'
            if re.search(postposition_pattern, after_pattern):
                self.logger.debug(f"조사 뒤에 오는 패턴: {pattern.text} + {after_pattern[:5]}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"조사 확인 중 오류: {e}")
            return False
    
    def _is_quoted_text(self, pattern: HeaderInfo) -> bool:
        """패턴이 따옴표로 감싸진 텍스트인지 확인"""
        try:
            line_text = pattern.line_text
            
            # 따옴표 패턴 체크
            quote_patterns = [
                r'["""].*["""]',  # 일반 따옴표
                r'[''].*['']',    # 작은따옴표
            ]
            
            for quote_pattern in quote_patterns:
                if re.search(quote_pattern, line_text):
                    self.logger.debug(f"따옴표로 감싸진 텍스트: {pattern.text}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"따옴표 확인 중 오류: {e}")
            return False
    
    def analyze_complex_patterns(self, complex_patterns: List[HeaderInfo]) -> List[Dict[str, Any]]:
        """
        복합 패턴 분석
        
        Args:
            complex_patterns: 복합 패턴들
            
        Returns:
            복합 패턴 분석 결과
        """
        analysis_results = []
        
        try:
            # 라인별로 그룹화
            line_groups = {}
            for pattern in complex_patterns:
                line_num = pattern.line_number
                if line_num not in line_groups:
                    line_groups[line_num] = []
                line_groups[line_num].append(pattern)
            
            # 각 라인별로 분석
            for line_num, patterns in line_groups.items():
                line_analysis = self._analyze_line_patterns(line_num, patterns)
                analysis_results.append(line_analysis)
            
            self.logger.info(f"복합 패턴 분석 완료: {len(analysis_results)}개 라인 분석")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"복합 패턴 분석 중 오류: {e}")
            return analysis_results
    
    def _analyze_line_patterns(self, line_num: int, patterns: List[HeaderInfo]) -> Dict[str, Any]:
        """한 라인의 패턴들을 분석"""
        try:
            # 위치 순으로 정렬
            sorted_patterns = sorted(patterns, key=lambda x: x["start"])
            
            analysis = {
                "line_number": line_num,
                "patterns": sorted_patterns,
                "pattern_count": len(sorted_patterns),
                "is_continuous": self._is_continuous_patterns(sorted_patterns),
                "hierarchy_levels": [p.type for p in sorted_patterns],
                "analysis_type": self._determine_analysis_type(sorted_patterns)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"라인 패턴 분석 중 오류: {e}")
            return {"line_number": line_num, "error": str(e)}
    
    def _is_continuous_patterns(self, patterns: List[HeaderInfo]) -> bool:
        """패턴들이 연속적인지 확인"""
        try:
            if len(patterns) < 2:
                return False
            
            # 위치가 연속적인지 확인
            for i in range(len(patterns) - 1):
                if patterns[i].end != patterns[i + 1].start:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"연속 패턴 확인 중 오류: {e}")
            return False
    
    def _determine_analysis_type(self, patterns: List[HeaderInfo]) -> str:
        """복합 패턴의 분석 타입 결정"""
        try:
            if self._is_continuous_patterns(patterns):
                return "continuous_headers"  # 연속 헤더
            else:
                return "mixed_patterns"      # 혼재 패턴
                
        except Exception as e:
            self.logger.error(f"분석 타입 결정 중 오류: {e}")
            return "unknown"
