"""
데이터 구조 정의

청킹 시스템에 필요한 기본 데이터 구조들
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

@dataclass
class HeaderInfo:
    """헤더 정보를 저장하는 데이터 클래스"""
    
    # 기본 정보
    type: str  # chapter, section, division, article, paragraph, subparagraph, item
    description: str  # 장, 절, 관, 조, 항, 호, 목
    text: str  # 원본 텍스트
    start: int  # 시작 위치
    end: int  # 끝 위치
    
    # 추가 정보
    line_number: int = 0  # 라인 번호
    line_text: str = ""  # 전체 라인 텍스트
    groups: tuple = field(default_factory=tuple)  # 정규식 그룹
    
    # 특별한 패턴 정보
    sub_number: Optional[str] = None  # 의조 번호
    is_sub_article: bool = False  # 의조 여부
    circle_number: Optional[str] = None  # 원형 숫자
    
    # 상태 플래그 (기존 시스템과 완벽 호환)
    status_flags: Dict[str, Any] = field(default_factory=lambda: {
        "is_omission": False,
        "is_deletion": False,
        "is_amendment": False,
        "is_appendix": False,
        "is_attachment": False,
        "appendix_type": "main"
    })
    
    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ChunkInfo:
    """청킹 결과를 저장하는 데이터 클래스"""
    
    # 기본 정보
    chunk_id: str  # 청크 고유 ID
    header: HeaderInfo  # 헤더 정보
    content: str  # 청크 내용
    content_type: str  # content, header_only
    
    # 위계 정보
    hierarchy_level: str  # 위계 레벨
    parent_chunk_id: Optional[str] = None  # 부모 청크 ID
    child_chunk_ids: List[str] = field(default_factory=list)  # 자식 청크 ID들
    
    # 메타데이터
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BufferInfo:
    """버퍼 정보를 저장하는 데이터 클래스"""
    
    # 기본 정보
    buffer_id: str  # 버퍼 고유 ID
    content: List[str] = field(default_factory=list)  # 버퍼 내용
    current_header: Optional[HeaderInfo] = None  # 현재 헤더
    
    # 상태 정보
    is_active: bool = True  # 활성 상태
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_content(self, text: str):
        """버퍼에 내용 추가"""
        self.content.append(text)
        self.last_updated = datetime.now()
    
    def get_full_content(self) -> str:
        """전체 버퍼 내용을 문자열로 반환"""
        return "\n".join(self.content)
    
    def clear(self):
        """버퍼 내용 비우기"""
        self.content.clear()
        self.last_updated = datetime.now()

class ChunkingResult:
    """청킹 결과를 관리하는 클래스"""
    
    def __init__(self):
        self.chunks: List[ChunkInfo] = []
        self.buffers: List[BufferInfo] = []
        self.logger = logging.getLogger(__name__)
        
        # 통계 정보
        self.total_chunks = 0
        self.total_buffers = 0
        self.processing_time = 0.0
        
        # 에러 정보
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
    
    def add_chunk(self, chunk: ChunkInfo):
        """청크 추가"""
        self.chunks.append(chunk)
        self.total_chunks += 1
        self.logger.debug(f"청크 추가: {chunk.chunk_id} - {chunk.header.type}")
    
    def add_buffer(self, buffer: BufferInfo):
        """버퍼 추가"""
        self.buffers.append(buffer)
        self.total_buffers += 1
        self.logger.debug(f"버퍼 추가: {buffer.buffer_id}")
    
    def add_error(self, error_info: Dict[str, Any]):
        """에러 정보 추가"""
        self.errors.append(error_info)
        self.logger.error(f"에러 발생: {error_info}")
    
    def add_warning(self, warning_info: Dict[str, Any]):
        """경고 정보 추가"""
        self.warnings.append(warning_info)
        self.logger.warning(f"경고 발생: {warning_info}")
    
    def get_summary(self) -> Dict[str, Any]:
        """결과 요약 반환"""
        return {
            "total_chunks": self.total_chunks,
            "total_buffers": self.total_buffers,
            "processing_time": self.processing_time,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "chunk_types": self._get_chunk_type_summary(),
            "buffer_types": self._get_buffer_type_summary()
        }
    
    def _get_chunk_type_summary(self) -> Dict[str, int]:
        """청크 타입별 개수 요약"""
        summary = {}
        for chunk in self.chunks:
            chunk_type = chunk.header.type
            summary[chunk_type] = summary.get(chunk_type, 0) + 1
        return summary
    
    def _get_buffer_type_summary(self) -> Dict[str, int]:
        """버퍼 타입별 개수 요약"""
        summary = {}
        for buffer in self.buffers:
            buffer_type = "content" if buffer.current_header else "header_only"
            summary[buffer_type] = summary.get(buffer_type, 0) + 1
        return summary

class PatternAnalysisResult:
    """패턴 분석 결과를 저장하는 클래스"""
    
    def __init__(self):
        self.patterns: List[HeaderInfo] = []
        self.line_analysis: Dict[int, List[HeaderInfo]] = {}
        self.pattern_summary: Dict[str, int] = {}
        
        # 분석 메타데이터
        self.total_lines = 0
        self.total_patterns = 0
        self.analysis_time = 0.0
    
    def add_pattern(self, pattern: HeaderInfo):
        """패턴 추가"""
        self.patterns.append(pattern)
        self.total_patterns += 1
        
        # 라인별 분석 결과에 추가
        line_num = pattern.line_number
        if line_num not in self.line_analysis:
            self.line_analysis[line_num] = []
        self.line_analysis[line_num].append(pattern)
    
    def get_line_patterns(self, line_number: int) -> List[HeaderInfo]:
        """특정 라인의 패턴들 반환"""
        return self.line_analysis.get(line_number, [])
    
    def get_patterns_by_type(self, pattern_type: str) -> List[HeaderInfo]:
        """특정 타입의 패턴들 반환"""
        return [p for p in self.patterns if p.type == pattern_type]
    
    def update_summary(self):
        """패턴 요약 업데이트"""
        self.pattern_summary = {}
        for pattern in self.patterns:
            pattern_type = pattern.type
            self.pattern_summary[pattern_type] = self.pattern_summary.get(pattern_type, 0) + 1
