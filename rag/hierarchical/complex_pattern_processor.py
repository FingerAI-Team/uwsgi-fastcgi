"""
Phase 3: 복합 패턴 처리기

복합 패턴을 처리하여 최적화된 청킹을 수행합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .data_structures import HeaderInfo, ChunkInfo, BufferInfo, ChunkingResult
from .complex_pattern_analyzer import ComplexPatternAnalyzer, ComplexPatternAnalysis, PatternComplexity, PatternRelation

class ChunkingStrategy(Enum):
    """청킹 전략 열거형"""
    HIERARCHICAL = "hierarchical"     # 계층별 청킹
    CONTINUOUS = "continuous"         # 연속성 기반 청킹
    RELATION_BASED = "relation_based" # 관계 기반 청킹
    ADAPTIVE = "adaptive"            # 적응적 청킹
    OPTIMIZED = "optimized"          # 최적화된 청킹

@dataclass
class ChunkingContext:
    """청킹 컨텍스트"""
    complexity: PatternComplexity
    hierarchy_depth: int
    continuous_ranges: List[Dict[str, Any]]
    pattern_relations: Dict[str, List[PatternRelation]]
    target_chunk_size: int = 1000
    max_chunk_size: int = 2000
    min_chunk_size: int = 200

class ComplexPatternProcessor:
    """복합 패턴 처리기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔧 복합 패턴 처리기 초기화")
        
        # 복합 패턴 분석기 초기화
        self.analyzer = ComplexPatternAnalyzer()
        
        # 전략별 처리기 매핑
        self.strategy_processors = {
            PatternComplexity.SIMPLE: self._process_simple_patterns,
            PatternComplexity.NESTED: self._process_nested_patterns,
            PatternComplexity.CONTINUOUS: self._process_continuous_patterns,
            PatternComplexity.MIXED: self._process_mixed_patterns,
            PatternComplexity.HIERARCHICAL: self._process_hierarchical_patterns
        }
        
        self.logger.info("✅ 복합 패턴 처리기 초기화 완료")
    
    def process_complex_patterns(self, analysis_result: Any, 
                               chunking_context: ChunkingContext) -> ChunkingResult:
        """
        복합 패턴 처리 및 청킹
        
        Args:
            analysis_result: 기본 패턴 분석 결과
            chunking_context: 청킹 컨텍스트
            
        Returns:
            청킹 결과
        """
        try:
            self.logger.info(f"🔍 복합 패턴 처리 시작: {chunking_context.complexity.value}")
            
            # 1. 복합 패턴 분석
            complex_analysis = self.analyzer.analyze_complex_patterns(analysis_result)
            
            # 2. 복잡도에 따른 처리기 선택
            processor = self.strategy_processors.get(complex_analysis.complexity)
            if processor:
                result = processor(complex_analysis, chunking_context)
                self.logger.debug(f"복합 패턴 처리 완료: {complex_analysis.complexity.value}")
                return result
            else:
                self.logger.warning(f"알 수 없는 복잡도: {complex_analysis.complexity}")
                return self._process_default(complex_analysis, chunking_context)
                
        except Exception as e:
            self.logger.error(f"복합 패턴 처리 중 오류: {e}")
            return self._create_error_result(f"복합 패턴 처리 실패: {e}")
    
    def _process_simple_patterns(self, analysis: ComplexPatternAnalysis, 
                                context: ChunkingContext) -> ChunkingResult:
        """단순 패턴 처리"""
        self.logger.debug("단순 패턴 처리")
        
        chunks = []
        for hierarchy_node in analysis.patterns:
            chunk = self._create_simple_chunk(hierarchy_node.pattern, context)
            chunks.append(chunk)
        
        return ChunkingResult(
            chunks=chunks,
            processing_notes="단순 패턴 처리 완료"
        )
    
    def _process_nested_patterns(self, analysis: ComplexPatternAnalysis, 
                                context: ChunkingContext) -> ChunkingResult:
        """중첩 패턴 처리"""
        self.logger.debug("중첩 패턴 처리")
        
        chunks = []
        
        # 최상위 패턴부터 처리
        top_level_patterns = [node for node in analysis.patterns if node.level == 1]
        
        for top_node in top_level_patterns:
            # 부모 패턴과 자식들을 하나의 청크로 그룹화
            chunk = self._create_nested_chunk(top_node, context)
            chunks.append(chunk)
        
        return ChunkingResult(
            chunks=chunks,
            processing_notes="중첩 패턴 처리 완료"
        )
    
    def _process_continuous_patterns(self, analysis: ComplexPatternAnalysis, 
                                   context: ChunkingContext) -> ChunkingResult:
        """연속 패턴 처리"""
        self.logger.debug("연속 패턴 처리")
        
        chunks = []
        
        # 연속 범위별로 청크 생성
        for range_info in analysis.continuous_ranges:
            chunk = self._create_continuous_chunk(range_info, analysis.patterns, context)
            chunks.append(chunk)
        
        # 독립적인 패턴들도 청크로 생성
        independent_patterns = [node for node in analysis.patterns 
                              if PatternRelation.INDEPENDENT in 
                              analysis.pattern_relations.get(f"{node.pattern.type}_{node.pattern.text}", [])]
        
        for node in independent_patterns:
            chunk = self._create_simple_chunk(node.pattern, context)
            chunks.append(chunk)
        
        return ChunkingResult(
            chunks=chunks,
            processing_notes="연속 패턴 처리 완료"
        )
    
    def _process_mixed_patterns(self, analysis: ComplexPatternAnalysis, 
                               context: ChunkingContext) -> ChunkingResult:
        """혼합 패턴 처리"""
        self.logger.debug("혼합 패턴 처리")
        
        chunks = []
        
        # 계층별로 그룹화하여 처리
        level_groups = {}
        for node in analysis.patterns:
            level = node.level
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # 각 레벨별로 처리
        for level, nodes in level_groups.items():
            if len(nodes) == 1:
                # 단일 패턴
                chunk = self._create_simple_chunk(nodes[0].pattern, context)
                chunks.append(chunk)
            else:
                # 복수 패턴 - 관계 기반으로 그룹화
                grouped_chunks = self._group_related_patterns(nodes, analysis.pattern_relations, context)
                chunks.extend(grouped_chunks)
        
        return ChunkingResult(
            chunks=chunks,
            processing_notes="혼합 패턴 처리 완료"
        )
    
    def _process_hierarchical_patterns(self, analysis: ComplexPatternAnalysis, 
                                     context: ChunkingContext) -> ChunkingResult:
        """계층적 패턴 처리"""
        self.logger.debug("계층적 패턴 처리")
        
        chunks = []
        
        # 최상위 레벨부터 순차적으로 처리
        for level in range(1, context.hierarchy_depth + 1):
            level_nodes = [node for node in analysis.patterns if node.level == level]
            
            if not level_nodes:
                continue
            
            # 같은 레벨의 패턴들을 관계에 따라 그룹화
            if len(level_nodes) == 1:
                chunk = self._create_hierarchical_chunk(level_nodes[0], context)
                chunks.append(chunk)
            else:
                # 관계 기반 그룹화
                grouped_chunks = self._group_hierarchical_patterns(level_nodes, analysis, context)
                chunks.extend(grouped_chunks)
        
        return ChunkingResult(
            chunks=chunks,
            processing_notes="계층적 패턴 처리 완료"
        )
    
    def _process_default(self, analysis: ComplexPatternAnalysis, 
                        context: ChunkingContext) -> ChunkingResult:
        """기본 처리 (오류 발생 시)"""
        self.logger.warning("기본 처리 사용")
        
        chunks = []
        for hierarchy_node in analysis.patterns:
            chunk = self._create_simple_chunk(hierarchy_node.pattern, context)
            chunks.append(chunk)
        
        return ChunkingResult(
            chunks=chunks,
            processing_notes="기본 처리 (오류 발생)"
        )
    
    def _create_simple_chunk(self, pattern: HeaderInfo, context: ChunkingContext) -> ChunkInfo:
        """단순 청크 생성"""
        return ChunkInfo(
            chunk_id=f"simple_{pattern.type}_{id(pattern)}",
            header=pattern,
            content=pattern.text,
            content_type="header_only",
            hierarchy_level=pattern.type,
            metadata={
                "pattern_type": pattern.type,
                "is_simple": True,
                "chunking_strategy": "simple",
                "processing_phase": "Phase3"
            }
        )
    
    def _create_nested_chunk(self, parent_node: Any, context: ChunkingContext) -> ChunkInfo:
        """중첩 청크 생성"""
        # 부모와 자식들의 텍스트를 결합
        content_parts = [parent_node.pattern.text]
        
        for child in parent_node.children:
            content_parts.append(child.pattern.text)
        
        combined_content = "\n".join(content_parts)
        
        return ChunkInfo(
            chunk_id=f"nested_{parent_node.pattern.type}_{id(parent_node.pattern)}",
            header=parent_node.pattern,
            content=combined_content,
            content_type="nested",
            hierarchy_level=parent_node.pattern.type,
            metadata={
                "pattern_type": parent_node.pattern.type,
                "is_nested": True,
                "children_count": len(parent_node.children),
                "chunking_strategy": "nested",
                "processing_phase": "Phase3"
            }
        )
    
    def _create_continuous_chunk(self, range_info: Dict[str, Any], 
                               patterns: List[Any], context: ChunkingContext) -> ChunkInfo:
        """연속 청크 생성"""
        # 범위에 해당하는 패턴들을 찾아서 결합
        start_value = range_info["start_value"]
        end_value = range_info["end_value"]
        pattern_type = range_info["pattern_type"]
        
        content_parts = [range_info["full_match"]]
        
        # 범위 내의 패턴들을 찾아서 추가
        for node in patterns:
            if (node.pattern.type == pattern_type and 
                node.pattern.groups and 
                start_value <= node.pattern.groups[0] <= end_value):
                content_parts.append(node.pattern.text)
        
        combined_content = "\n".join(content_parts)
        
        return ChunkInfo(
            chunk_id=f"continuous_{pattern_type}_{start_value}_{end_value}",
            header=patterns[0].pattern if patterns else None,
            content=combined_content,
            content_type="continuous",
            hierarchy_level=pattern_type,
            metadata={
                "pattern_type": pattern_type,
                "is_continuous": True,
                "start_value": start_value,
                "end_value": end_value,
                "chunking_strategy": "continuous",
                "processing_phase": "Phase3"
            }
        )
    
    def _create_hierarchical_chunk(self, node: Any, context: ChunkingContext) -> ChunkInfo:
        """계층적 청크 생성"""
        return ChunkInfo(
            chunk_id=f"hierarchical_{node.pattern.type}_{id(node.pattern)}",
            header=node.pattern,
            content=node.pattern.text,
            content_type="hierarchical",
            hierarchy_level=node.pattern.type,
            metadata={
                "pattern_type": node.pattern.type,
                "is_hierarchical": True,
                "hierarchy_level": node.level,
                "chunking_strategy": "hierarchical",
                "processing_phase": "Phase3"
            }
        )
    
    def _group_related_patterns(self, nodes: List[Any], 
                               pattern_relations: Dict[str, List[PatternRelation]], 
                               context: ChunkingContext) -> List[ChunkInfo]:
        """관련 패턴들을 그룹화하여 청크 생성"""
        chunks = []
        
        # 관계별로 그룹화
        related_groups = []
        processed_nodes = set()
        
        for node in nodes:
            if node in processed_nodes:
                continue
            
            # 관련된 패턴들을 찾아서 그룹화
            group = [node]
            processed_nodes.add(node)
            
            node_id = f"{node.pattern.type}_{node.pattern.text}"
            relations = pattern_relations.get(node_id, [])
            
            # 형제 관계나 연속 관계가 있는 패턴들을 그룹에 추가
            for other_node in nodes:
                if other_node in processed_nodes:
                    continue
                
                other_id = f"{other_node.pattern.type}_{other_node.pattern.text}"
                other_relations = pattern_relations.get(other_id, [])
                
                if (PatternRelation.SIBLING in relations or 
                    PatternRelation.CONTINUATION in relations):
                    group.append(other_node)
                    processed_nodes.add(other_node)
            
            related_groups.append(group)
        
        # 각 그룹을 청크로 생성
        for group in related_groups:
            if len(group) == 1:
                chunk = self._create_simple_chunk(group[0].pattern, context)
            else:
                chunk = self._create_grouped_chunk(group, context)
            chunks.append(chunk)
        
        return chunks
    
    def _group_hierarchical_patterns(self, nodes: List[Any], 
                                   analysis: ComplexPatternAnalysis, 
                                   context: ChunkingContext) -> List[ChunkInfo]:
        """계층적 패턴들을 그룹화하여 청크 생성"""
        chunks = []
        
        # 부모-자식 관계를 고려하여 그룹화
        for node in nodes:
            if node.children:
                # 자식이 있는 경우 중첩 청크 생성
                chunk = self._create_nested_chunk(node, context)
                chunks.append(chunk)
            else:
                # 자식이 없는 경우 단순 청크 생성
                chunk = self._create_simple_chunk(node.pattern, context)
                chunks.append(chunk)
        
        return chunks
    
    def _create_grouped_chunk(self, group: List[Any], context: ChunkingContext) -> ChunkInfo:
        """그룹화된 청크 생성"""
        # 그룹의 첫 번째 패턴을 헤더로 사용
        header = group[0].pattern
        
        # 모든 패턴의 텍스트를 결합
        content_parts = [node.pattern.text for node in group]
        combined_content = "\n".join(content_parts)
        
        return ChunkInfo(
            chunk_id=f"grouped_{header.type}_{id(header)}",
            header=header,
            content=combined_content,
            content_type="grouped",
            hierarchy_level=header.type,
            metadata={
                "pattern_type": header.type,
                "is_grouped": True,
                "group_size": len(group),
                "chunking_strategy": "relation_based",
                "processing_phase": "Phase3"
            }
        )
    
    def _create_error_result(self, error_message: str) -> ChunkingResult:
        """오류 결과 생성"""
        return ChunkingResult(
            chunks=[],
            processing_notes=f"오류: {error_message}"
        )
    
    def get_processing_summary(self, result: ChunkingResult) -> Dict[str, Any]:
        """처리 결과 요약 반환"""
        try:
            chunk_types = {}
            strategies = {}
            
            for chunk in result.chunks:
                # 청크 타입별 개수
                chunk_type = chunk.content_type
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                # 전략별 개수
                strategy = chunk.metadata.get("chunking_strategy", "unknown")
                strategies[strategy] = strategies.get(strategy, 0) + 1
            
            return {
                "total_chunks": len(result.chunks),
                "chunk_types": chunk_types,
                "strategies": strategies,
                "processing_notes": result.processing_notes
            }
            
        except Exception as e:
            self.logger.error(f"처리 요약 생성 중 오류: {e}")
            return {}
