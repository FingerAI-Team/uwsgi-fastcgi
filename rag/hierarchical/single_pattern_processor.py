"""
Phase 2: 단일 패턴 처리기

7가지 케이스별로 전문화된 패턴 처리 로직을 구현합니다.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .data_structures import HeaderInfo, ChunkInfo, BufferInfo, ChunkingResult

class PatternCase(Enum):
    """패턴 처리 케이스 열거형"""
    HEADER_ONLY = "header_only"           # 헤더만 있는 경우
    HEADER_WITH_CONTENT = "header_with_content"  # 헤더 + 내용
    REFERENCE_ONLY = "reference_only"     # 참조/인용만 있는 경우
    REFERENCE_WITH_CONTENT = "reference_with_content"  # 참조/인용 + 내용
    SINGLE_PATTERN = "single_pattern"     # 단일 패턴
    COMPLEX_PATTERN = "complex_pattern"   # 복합 패턴
    MIXED_PATTERN = "mixed_pattern"       # 혼합 패턴

@dataclass
class ProcessingContext:
    """패턴 처리 컨텍스트"""
    current_line: str
    line_number: int
    previous_patterns: List[HeaderInfo]
    next_patterns: List[HeaderInfo]
    buffer_state: BufferInfo
    metadata: Dict[str, Any]

class SinglePatternProcessor:
    """
    단일 패턴 처리기
    
    7가지 케이스별로 전문화된 처리 로직을 제공합니다.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔧 단일 패턴 처리기 초기화")
        
        # 케이스별 처리기 매핑
        self.case_processors = {
            PatternCase.HEADER_ONLY: self._process_header_only,
            PatternCase.HEADER_WITH_CONTENT: self._process_header_with_content,
            PatternCase.REFERENCE_ONLY: self._process_reference_only,
            PatternCase.REFERENCE_WITH_CONTENT: self._process_reference_with_content,
            PatternCase.SINGLE_PATTERN: self._process_single_pattern,
            PatternCase.COMPLEX_PATTERN: self._process_complex_pattern,
            PatternCase.MIXED_PATTERN: self._process_mixed_pattern,
        }
    
    def process_pattern(self, pattern: HeaderInfo, context: ProcessingContext) -> ChunkingResult:
        """
        패턴을 처리하여 청킹 결과를 반환
        
        Args:
            pattern: 처리할 패턴
            context: 처리 컨텍스트
            
        Returns:
            청킹 결과
        """
        try:
            # 패턴 케이스 판별
            case = self._determine_pattern_case(pattern, context)
            self.logger.debug(f"패턴 케이스 판별: {case.value}")
            
            # 해당 케이스 처리기 호출
            if case in self.case_processors:
                processor = self.case_processors[case]
                result = processor(pattern, context)
                self.logger.debug(f"케이스 {case.value} 처리 완료")
                return result
            else:
                self.logger.warning(f"알 수 없는 패턴 케이스: {case}")
                return self._process_default(pattern, context)
                
        except Exception as e:
            self.logger.error(f"패턴 처리 중 오류: {e}")
            return self._process_default(pattern, context)
    
    def _determine_pattern_case(self, pattern: HeaderInfo, context: ProcessingContext) -> PatternCase:
        """
        패턴의 케이스를 판별
        
        Args:
            pattern: 분석할 패턴
            context: 처리 컨텍스트
            
        Returns:
            패턴 케이스
        """
        try:
            # 1. 헤더 전용 케이스
            if self._is_header_only(pattern, context):
                return PatternCase.HEADER_ONLY
            
            # 2. 헤더 + 내용 케이스
            if self._is_header_with_content(pattern, context):
                return PatternCase.HEADER_WITH_CONTENT
            
            # 3. 참조/인용 전용 케이스
            if self._is_reference_only(pattern, context):
                return PatternCase.REFERENCE_ONLY
            
            # 4. 참조/인용 + 내용 케이스
            if self._is_reference_with_content(pattern, context):
                return PatternCase.REFERENCE_WITH_CONTENT
            
            # 5. 단일 패턴 케이스
            if self._is_single_pattern(pattern, context):
                return PatternCase.SINGLE_PATTERN
            
            # 6. 복합 패턴 케이스
            if self._is_complex_pattern(pattern, context):
                return PatternCase.COMPLEX_PATTERN
            
            # 7. 혼합 패턴 케이스 (기본값)
            return PatternCase.MIXED_PATTERN
            
        except Exception as e:
            self.logger.error(f"패턴 케이스 판별 중 오류: {e}")
            return PatternCase.MIXED_PATTERN
    
    def _is_header_only(self, pattern: HeaderInfo, context: ProcessingContext) -> bool:
        """헤더 전용 케이스 판별"""
        # 헤더 타입이고 다음 라인에 내용이 없는 경우
        return (pattern.type in ["chapter", "section", "division", "article"] and
                not self._has_content_in_next_line(context))
    
    def _is_header_with_content(self, pattern: HeaderInfo, context: ProcessingContext) -> bool:
        """헤더 + 내용 케이스 판별"""
        # 헤더 타입이고 다음 라인에 내용이 있는 경우
        return (pattern.type in ["chapter", "section", "division", "article"] and
                self._has_content_in_next_line(context))
    
    def _is_reference_only(self, pattern: HeaderInfo, context: ProcessingContext) -> bool:
        """참조/인용 전용 케이스 판별"""
        # 참조/인용 타입이고 다음 라인에 내용이 없는 경우
        return (pattern.type in ["reference", "citation"] and
                not self._has_content_in_next_line(context))
    
    def _is_reference_with_content(self, pattern: HeaderInfo, context: ProcessingContext) -> bool:
        """참조/인용 + 내용 케이스 판별"""
        # 참조/인용 타입이고 다음 라인에 내용이 있는 경우
        return (pattern.type in ["reference", "citation"] and
                self._has_content_in_next_line(context))
    
    def _is_single_pattern(self, pattern: HeaderInfo, context: ProcessingContext) -> bool:
        """단일 패턴 케이스 판별"""
        # 단일 패턴 타입 (항, 호, 목 등)
        return pattern.type in ["paragraph", "subparagraph", "item"]
    
    def _is_complex_pattern(self, pattern: HeaderInfo, context: ProcessingContext) -> bool:
        """복합 패턴 케이스 판별"""
        # 복합 패턴 (여러 패턴이 결합된 경우)
        return len(pattern.groups) > 2 or self._has_nested_patterns(pattern, context)
    
    def _has_content_in_next_line(self, context: ProcessingContext) -> bool:
        """다음 라인에 내용이 있는지 확인"""
        # 다음 라인이 있고, 패턴이 아닌 일반 텍스트인 경우
        if context.line_number + 1 < len(context.next_patterns):
            next_line = context.next_patterns[context.line_number + 1]
            return not self._is_pattern_line(next_line)
        return False
    
    def _is_pattern_line(self, line: str) -> bool:
        """라인이 패턴 라인인지 확인"""
        # 간단한 패턴 체크 (실제로는 더 정교한 로직 필요)
        pattern_indicators = ["제", "조", "장", "절", "관", "①", "②", "③", "1.", "(1)", "가.", "(가)"]
        return any(indicator in line for indicator in pattern_indicators)
    
    def _has_nested_patterns(self, pattern: HeaderInfo, context: ProcessingContext) -> bool:
        """중첩된 패턴이 있는지 확인"""
        # 패턴 내에 다른 패턴이 포함된 경우
        return any(self._contains_nested_pattern(pattern.text, nested_type) 
                  for nested_type in ["paragraph", "subparagraph", "item"])
    
    def _contains_nested_pattern(self, text: str, pattern_type: str) -> bool:
        """텍스트에 중첩된 패턴이 포함되어 있는지 확인"""
        if pattern_type == "paragraph":
            return bool(re.search(r'[①-⑳]|\(\d+\)|\d+\.', text))
        elif pattern_type == "subparagraph":
            return bool(re.search(r'\d+[\.\)]|\(\d+\)|[가-힣][\.\)]|\([가-힣]\)', text))
        elif pattern_type == "item":
            return bool(re.search(r'[가-힣]\.|\([가-힣]\)', text))
        return False
    
    # ==================== 케이스별 처리기 ====================
    
    def _process_header_only(self, pattern: HeaderInfo, context: ProcessingContext) -> ChunkingResult:
        """헤더 전용 케이스 처리"""
        self.logger.debug(f"헤더 전용 처리: {pattern.type}")
        
        # 헤더만으로 청크 생성
        chunk = ChunkInfo(
            text=pattern.text,
            metadata=self._create_header_metadata(pattern),
            pattern_type=pattern.type,
            start_line=context.line_number,
            end_line=context.line_number
        )
        
        return ChunkingResult(
            chunks=[chunk],
            buffer_state=context.buffer_state,
            processing_notes=f"헤더 전용 처리: {pattern.type}"
        )
    
    def _process_header_with_content(self, pattern: HeaderInfo, context: ProcessingContext) -> ChunkingResult:
        """헤더 + 내용 케이스 처리"""
        self.logger.debug(f"헤더 + 내용 처리: {pattern.type}")
        
        # 헤더와 내용을 포함한 청크 생성
        content_text = self._extract_content_text(context)
        full_text = f"{pattern.text}\n{content_text}"
        
        chunk = ChunkInfo(
            text=full_text,
            metadata=self._create_header_metadata(pattern),
            pattern_type=pattern.type,
            start_line=context.line_number,
            end_line=self._find_content_end_line(context)
        )
        
        return ChunkingResult(
            chunks=[chunk],
            buffer_state=context.buffer_state,
            processing_notes=f"헤더 + 내용 처리: {pattern.type}"
        )
    
    def _process_reference_only(self, pattern: HeaderInfo, context: ProcessingContext) -> ChunkingResult:
        """참조/인용 전용 케이스 처리"""
        self.logger.debug(f"참조/인용 전용 처리: {pattern.type}")
        
        # 참조/인용만으로 청크 생성
        chunk = ChunkInfo(
            text=pattern.text,
            metadata=self._create_reference_metadata(pattern),
            pattern_type=pattern.type,
            start_line=context.line_number,
            end_line=context.line_number
        )
        
        return ChunkingResult(
            chunks=[chunk],
            buffer_state=context.buffer_state,
            processing_notes=f"참조/인용 전용 처리: {pattern.type}"
        )
    
    def _process_reference_with_content(self, pattern: HeaderInfo, context: ProcessingContext) -> ChunkingResult:
        """참조/인용 + 내용 케이스 처리"""
        self.logger.debug(f"참조/인용 + 내용 처리: {pattern.type}")
        
        # 참조/인용과 내용을 포함한 청크 생성
        content_text = self._extract_content_text(context)
        full_text = f"{pattern.text}\n{content_text}"
        
        chunk = ChunkInfo(
            text=full_text,
            metadata=self._create_reference_metadata(pattern),
            pattern_type=pattern.type,
            start_line=context.line_number,
            end_line=self._find_content_end_line(context)
        )
        
        return ChunkingResult(
            chunks=[chunk],
            buffer_state=context.buffer_state,
            processing_notes=f"참조/인용 + 내용 처리: {pattern.type}"
        )
    
    def _process_single_pattern(self, pattern: HeaderInfo, context: ProcessingContext) -> ChunkingResult:
        """단일 패턴 케이스 처리"""
        self.logger.debug(f"단일 패턴 처리: {pattern.type}")
        
        # 단일 패턴 처리 (항, 호, 목 등)
        chunk = ChunkInfo(
            text=pattern.text,
            metadata=self._create_single_pattern_metadata(pattern),
            pattern_type=pattern.type,
            start_line=context.line_number,
            end_line=context.line_number
        )
        
        return ChunkingResult(
            chunks=[chunk],
            buffer_state=context.buffer_state,
            processing_notes=f"단일 패턴 처리: {pattern.type}"
        )
    
    def _process_complex_pattern(self, pattern: HeaderInfo, context: ProcessingContext) -> ChunkingResult:
        """복합 패턴 케이스 처리"""
        self.logger.debug(f"복합 패턴 처리: {pattern.type}")
        
        # 복합 패턴을 여러 청크로 분해
        sub_chunks = self._decompose_complex_pattern(pattern, context)
        
        return ChunkingResult(
            chunks=sub_chunks,
            buffer_state=context.buffer_state,
            processing_notes=f"복합 패턴 분해 처리: {pattern.type}"
        )
    
    def _process_mixed_pattern(self, pattern: HeaderInfo, context: ProcessingContext) -> ChunkingResult:
        """혼합 패턴 케이스 처리 (기본 처리)"""
        self.logger.debug(f"혼합 패턴 처리: {pattern.type}")
        
        # 기본 청크 생성
        chunk = ChunkInfo(
            text=pattern.text,
            metadata=self._create_default_metadata(pattern),
            pattern_type=pattern.type,
            start_line=context.line_number,
            end_line=context.line_number
        )
        
        return ChunkingResult(
            chunks=[chunk],
            buffer_state=context.buffer_state,
            processing_notes=f"혼합 패턴 기본 처리: {pattern.type}"
        )
    
    def _process_default(self, pattern: HeaderInfo, context: ProcessingContext) -> ChunkingResult:
        """기본 처리 (오류 발생 시)"""
        self.logger.warning(f"기본 처리 사용: {pattern.type}")
        
        chunk = ChunkInfo(
            text=pattern.text,
            metadata=self._create_default_metadata(pattern),
            pattern_type=pattern.type,
            start_line=context.line_number,
            end_line=context.line_number
        )
        
        return ChunkingResult(
            chunks=[chunk],
            buffer_state=context.buffer_state,
            processing_notes=f"기본 처리 (오류 발생): {pattern.type}"
        )
    
    # ==================== 헬퍼 메서드 ====================
    
    def _extract_content_text(self, context: ProcessingContext) -> str:
        """컨텍스트에서 내용 텍스트 추출"""
        # 다음 라인부터 패턴이 나올 때까지의 텍스트 수집
        content_lines = []
        current_line = context.line_number + 1
        
        while current_line < len(context.next_patterns):
            line = context.next_patterns[current_line]
            if self._is_pattern_line(line):
                break
            content_lines.append(line)
            current_line += 1
        
        return "\n".join(content_lines)
    
    def _find_content_end_line(self, context: ProcessingContext) -> int:
        """내용이 끝나는 라인 번호 찾기"""
        current_line = context.line_number + 1
        
        while current_line < len(context.next_patterns):
            line = context.next_patterns[current_line]
            if self._is_pattern_line(line):
                break
            current_line += 1
        
        return current_line - 1
    
    def _decompose_complex_pattern(self, pattern: HeaderInfo, context: ProcessingContext) -> List[ChunkInfo]:
        """복합 패턴을 여러 청크로 분해"""
        chunks = []
        
        # 패턴 그룹별로 청크 생성
        for i, group in enumerate(pattern.groups):
            if group:  # 빈 그룹은 건너뛰기
                chunk = ChunkInfo(
                    text=f"{pattern.text} - 그룹{i+1}: {group}",
                    metadata=self._create_complex_pattern_metadata(pattern, i),
                    pattern_type=f"{pattern.type}_group_{i+1}",
                    start_line=context.line_number,
                    end_line=context.line_number
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_header_metadata(self, pattern: HeaderInfo) -> Dict[str, Any]:
        """헤더 메타데이터 생성"""
        metadata = {
            "pattern_type": pattern.type,
            "is_header": True,
            "header_text": pattern.text,
            "description": pattern.description
        }
        
        # 패턴 타입별 특수 메타데이터
        if pattern.type == "chapter":
            metadata.update({
                "chapter_number": pattern.groups[0] if pattern.groups else "",
                "chapter_title": pattern.groups[1] if len(pattern.groups) > 1 else ""
            })
        elif pattern.type == "section":
            metadata.update({
                "section_number": pattern.groups[0] if pattern.groups else "",
                "section_title": pattern.groups[1] if len(pattern.groups) > 1 else ""
            })
        elif pattern.type == "article":
            metadata.update({
                "article_number": pattern.groups[0] if pattern.groups else "",
                "article_sub": pattern.groups[1] if len(pattern.groups) > 1 else "",
                "article_title": pattern.groups[2] if len(pattern.groups) > 2 else ""
            })
        
        # 상태 플래그 통합 (기존 시스템과 완벽 호환)
        if hasattr(pattern, 'status_flags') and pattern.status_flags:
            metadata.update(pattern.status_flags)
        
        return metadata
    
    def _create_reference_metadata(self, pattern: HeaderInfo) -> Dict[str, Any]:
        """참조/인용 메타데이터 생성"""
        return {
            "pattern_type": pattern.type,
            "is_reference": True,
            "reference_text": pattern.text,
            "description": pattern.description,
            "reference_type": "internal" if "조" in pattern.text else "external"
        }
    
    def _create_single_pattern_metadata(self, pattern: HeaderInfo) -> Dict[str, Any]:
        """단일 패턴 메타데이터 생성"""
        return {
            "pattern_type": pattern.type,
            "is_single_pattern": True,
            "pattern_text": pattern.text,
            "description": pattern.description,
            "level": self._get_pattern_level(pattern.type)
        }
    
    def _create_complex_pattern_metadata(self, pattern: HeaderInfo, group_index: int) -> Dict[str, Any]:
        """복합 패턴 메타데이터 생성"""
        return {
            "pattern_type": pattern.type,
            "is_complex_pattern": True,
            "pattern_text": pattern.text,
            "description": pattern.description,
            "group_index": group_index,
            "group_value": pattern.groups[group_index] if group_index < len(pattern.groups) else ""
        }
    
    def _create_default_metadata(self, pattern: HeaderInfo) -> Dict[str, Any]:
        """기본 메타데이터 생성"""
        return {
            "pattern_type": pattern.type,
            "pattern_text": pattern.text,
            "description": pattern.description,
            "processing_method": "default"
        }
    
    def _get_pattern_level(self, pattern_type: str) -> str:
        """패턴 레벨 반환"""
        level_mapping = {
            "paragraph": "high",
            "subparagraph": "medium", 
            "item": "low"
        }
        return level_mapping.get(pattern_type, "unknown")
