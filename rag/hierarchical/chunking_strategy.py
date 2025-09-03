"""
Phase 2: 패턴별 청킹 전략

패턴 타입별로 최적화된 청킹 방법을 제공합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .data_structures import HeaderInfo, ChunkInfo, BufferInfo, ChunkingResult

@dataclass
class ChunkingContext:
    """청킹 컨텍스트"""
    pattern: HeaderInfo
    surrounding_text: List[str]
    buffer_state: BufferInfo
    metadata: Dict[str, Any]
    chunking_rules: Dict[str, Any]

class ChunkingStrategy(ABC):
    """청킹 전략 추상 클래스"""
    
    @abstractmethod
    def should_chunk(self, context: ChunkingContext) -> bool:
        """청킹 여부 결정"""
        pass
    
    @abstractmethod
    def create_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """청크 생성"""
        pass
    
    @abstractmethod
    def get_chunk_size(self, context: ChunkingContext) -> int:
        """청크 크기 반환"""
        pass

class HeaderChunkingStrategy(ChunkingStrategy):
    """헤더 청킹 전략"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def should_chunk(self, context: ChunkingContext) -> bool:
        """헤더는 항상 청킹"""
        return True
    
    def create_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """헤더 청크 생성"""
        pattern = context.pattern
        
        # 헤더 타입별 특수 처리
        if pattern.type == "chapter":
            return self._create_chapter_chunk(context)
        elif pattern.type == "section":
            return self._create_section_chunk(context)
        elif pattern.type == "division":
            return self._create_division_chunk(context)
        elif pattern.type == "article":
            return self._create_article_chunk(context)
        else:
            return self._create_generic_header_chunk(context)
    
    def get_chunk_size(self, context: ChunkingContext) -> int:
        """헤더 청크 크기"""
        return 1  # 헤더는 1라인
    
    def _create_chapter_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """장 청크 생성"""
        pattern = context.pattern
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": "chapter",
                "is_header": True,
                "chapter_number": pattern.groups[0] if pattern.groups else "",
                "chapter_title": pattern.groups[1] if len(pattern.groups) > 1 else "",
                "level": "chapter",
                "chunking_strategy": "header"
            },
            pattern_type="chapter",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )
    
    def _create_section_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """절 청크 생성"""
        pattern = context.pattern
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": "section",
                "is_header": True,
                "section_number": pattern.groups[0] if pattern.groups else "",
                "section_title": pattern.groups[1] if len(pattern.groups) > 1 else "",
                "level": "section",
                "chunking_strategy": "header"
            },
            pattern_type="section",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )
    
    def _create_division_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """관 청크 생성"""
        pattern = context.pattern
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": "division",
                "is_header": True,
                "division_number": pattern.groups[0] if pattern.groups else "",
                "division_title": pattern.groups[1] if len(pattern.groups) > 1 else "",
                "level": "division",
                "chunking_strategy": "header"
            },
            pattern_type="division",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )
    
    def _create_article_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """조 청크 생성"""
        pattern = context.pattern
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": "article",
                "is_header": True,
                "article_number": pattern.groups[0] if pattern.groups else "",
                "article_sub": pattern.groups[1] if len(pattern.groups) > 1 else "",
                "article_title": pattern.groups[2] if len(pattern.groups) > 2 else "",
                "level": "article",
                "chunking_strategy": "header"
            },
            pattern_type="article",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )
    
    def _create_generic_header_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """일반 헤더 청크 생성"""
        pattern = context.pattern
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": pattern.type,
                "is_header": True,
                "header_text": pattern.text,
                "description": pattern.description,
                "level": "generic",
                "chunking_strategy": "header"
            },
            pattern_type=pattern.type,
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )

class ContentChunkingStrategy(ChunkingStrategy):
    """내용 청킹 전략"""
    
    def __init__(self, max_chunk_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_chunk_size = max_chunk_size
    
    def should_chunk(self, context: ChunkingContext) -> bool:
        """내용 청킹 여부 결정"""
        # 내용이 있고 최대 크기를 초과하는 경우
        return (len(context.surrounding_text) > 0 and 
                self._calculate_content_size(context) > self.max_chunk_size)
    
    def create_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """내용 청크 생성"""
        # 내용을 최적 크기로 분할
        chunks = self._split_content_optimally(context)
        
        # 첫 번째 청크 반환 (나머지는 별도 처리)
        return chunks[0] if chunks else self._create_empty_chunk(context)
    
    def get_chunk_size(self, context: ChunkingContext) -> int:
        """내용 청크 크기"""
        return min(self._calculate_content_size(context), self.max_chunk_size)
    
    def _calculate_content_size(self, context: ChunkingContext) -> int:
        """내용 크기 계산"""
        return sum(len(line) for line in context.surrounding_text)
    
    def _split_content_optimally(self, context: ChunkingContext) -> List[ChunkInfo]:
        """내용을 최적으로 분할"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in context.surrounding_text:
            line_size = len(line)
            
            # 현재 청크에 추가할 수 있는지 확인
            if current_size + line_size <= self.max_chunk_size:
                current_chunk.append(line)
                current_size += line_size
            else:
                # 현재 청크 완성
                if current_chunk:
                    chunks.append(self._create_content_chunk(current_chunk, context))
                
                # 새 청크 시작
                current_chunk = [line]
                current_size = line_size
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(self._create_content_chunk(current_chunk, context))
        
        return chunks
    
    def _create_content_chunk(self, content_lines: List[str], context: ChunkingContext) -> ChunkInfo:
        """내용 청크 생성"""
        return ChunkInfo(
            text="\n".join(content_lines),
            metadata={
                "pattern_type": "content",
                "is_content": True,
                "content_size": len(content_lines),
                "chunking_strategy": "content",
                "max_chunk_size": self.max_chunk_size
            },
            pattern_type="content",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )
    
    def _create_empty_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """빈 청크 생성"""
        return ChunkInfo(
            text="",
            metadata={
                "pattern_type": "empty",
                "is_empty": True,
                "chunking_strategy": "content"
            },
            pattern_type="empty",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )

class ReferenceChunkingStrategy(ChunkingStrategy):
    """참조/인용 청킹 전략"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def should_chunk(self, context: ChunkingContext) -> bool:
        """참조/인용은 항상 청킹"""
        return True
    
    def create_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """참조/인용 청크 생성"""
        pattern = context.pattern
        
        # 참조 타입 판별
        reference_type = self._determine_reference_type(pattern)
        
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": "reference",
                "is_reference": True,
                "reference_text": pattern.text,
                "reference_type": reference_type,
                "description": pattern.description,
                "chunking_strategy": "reference"
            },
            pattern_type="reference",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )
    
    def get_chunk_size(self, context: ChunkingContext) -> int:
        """참조/인용 청크 크기"""
        return 1  # 참조/인용은 1라인
    
    def _determine_reference_type(self, pattern: HeaderInfo) -> str:
        """참조 타입 판별"""
        text = pattern.text.lower()
        
        if "조" in text:
            return "internal_article"
        elif "법" in text:
            return "law_reference"
        elif "규정" in text:
            return "regulation_reference"
        elif "지침" in text:
            return "guideline_reference"
        else:
            return "general_reference"

class SinglePatternChunkingStrategy(ChunkingStrategy):
    """단일 패턴 청킹 전략"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def should_chunk(self, context: ChunkingContext) -> bool:
        """단일 패턴은 항상 청킹"""
        return True
    
    def create_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """단일 패턴 청크 생성"""
        pattern = context.pattern
        
        # 패턴 타입별 특수 처리
        if pattern.type == "paragraph":
            return self._create_paragraph_chunk(context)
        elif pattern.type == "subparagraph":
            return self._create_subparagraph_chunk(context)
        elif pattern.type == "item":
            return self._create_item_chunk(context)
        else:
            return self._create_generic_single_chunk(context)
    
    def get_chunk_size(self, context: ChunkingContext) -> int:
        """단일 패턴 청크 크기"""
        return 1  # 단일 패턴은 1라인
    
    def _create_paragraph_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """항 청크 생성"""
        pattern = context.pattern
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": "paragraph",
                "is_single_pattern": True,
                "paragraph_text": pattern.text,
                "description": pattern.description,
                "level": "high",
                "chunking_strategy": "single_pattern"
            },
            pattern_type="paragraph",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )
    
    def _create_subparagraph_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """호 청크 생성"""
        pattern = context.pattern
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": "subparagraph",
                "is_single_pattern": True,
                "subparagraph_text": pattern.text,
                "description": pattern.description,
                "level": "medium",
                "chunking_strategy": "single_pattern"
            },
            pattern_type="subparagraph",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )
    
    def _create_item_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """목 청크 생성"""
        pattern = context.pattern
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": "item",
                "is_single_pattern": True,
                "item_text": pattern.text,
                "description": pattern.description,
                "level": "low",
                "chunking_strategy": "single_pattern"
            },
            pattern_type="item",
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )
    
    def _create_generic_single_chunk(self, context: ChunkingContext) -> ChunkInfo:
        """일반 단일 패턴 청크 생성"""
        pattern = context.pattern
        return ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": pattern.type,
                "is_single_pattern": True,
                "pattern_text": pattern.text,
                "description": pattern.description,
                "level": "unknown",
                "chunking_strategy": "single_pattern"
            },
            pattern_type=pattern.type,
            start_line=context.metadata.get("line_number", 0),
            end_line=context.metadata.get("line_number", 0)
        )

class ChunkingStrategyFactory:
    """청킹 전략 팩토리"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 패턴 타입별 전략 매핑
        self.strategies = {
            "chapter": HeaderChunkingStrategy(),
            "section": HeaderChunkingStrategy(),
            "division": HeaderChunkingStrategy(),
            "article": HeaderChunkingStrategy(),
            "content": ContentChunkingStrategy(),
            "reference": ReferenceChunkingStrategy(),
            "citation": ReferenceChunkingStrategy(),
            "paragraph": SinglePatternChunkingStrategy(),
            "subparagraph": SinglePatternChunkingStrategy(),
            "item": SinglePatternChunkingStrategy(),
        }
    
    def get_strategy(self, pattern_type: str) -> ChunkingStrategy:
        """패턴 타입에 맞는 전략 반환"""
        strategy = self.strategies.get(pattern_type)
        
        if strategy is None:
            self.logger.warning(f"패턴 타입 '{pattern_type}'에 대한 전략이 없습니다. 기본 전략 사용")
            return ContentChunkingStrategy()  # 기본 전략
        
        return strategy
    
    def get_all_strategies(self) -> Dict[str, ChunkingStrategy]:
        """모든 전략 반환"""
        return self.strategies.copy()
    
    def register_strategy(self, pattern_type: str, strategy: ChunkingStrategy):
        """새로운 전략 등록"""
        self.strategies[pattern_type] = strategy
        self.logger.info(f"새로운 전략 등록: {pattern_type} -> {strategy.__class__.__name__}")
    
    def unregister_strategy(self, pattern_type: str):
        """전략 제거"""
        if pattern_type in self.strategies:
            del self.strategies[pattern_type]
            self.logger.info(f"전략 제거: {pattern_type}")
        else:
            self.logger.warning(f"제거할 전략이 없습니다: {pattern_type}")
