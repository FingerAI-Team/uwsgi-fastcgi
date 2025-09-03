"""
Phase 2: 통합 처리 엔진

단일 패턴 처리기와 청킹 전략을 통합하여 전체 패턴 처리 플로우를 관리합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .single_pattern_processor import SinglePatternProcessor, ProcessingContext
from .chunking_strategy import ChunkingStrategyFactory, ChunkingContext
from .data_structures import (
    HeaderInfo, ChunkInfo, BufferInfo, ChunkingResult, 
    PatternAnalysisResult
)

@dataclass
class ProcessingPipeline:
    """처리 파이프라인 설정"""
    enable_single_pattern_processing: bool = True
    enable_chunking_strategy: bool = True
    enable_metadata_enrichment: bool = True
    enable_quality_control: bool = True
    max_chunk_size: int = 1000
    min_chunk_size: int = 50

class PatternProcessingEngine:
    """
    통합 패턴 처리 엔진
    
    Phase 2의 모든 구성 요소를 통합하여 패턴 처리를 수행합니다.
    """
    
    def __init__(self, pipeline_config: Optional[ProcessingPipeline] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 통합 패턴 처리 엔진 초기화")
        
        # 파이프라인 설정
        self.pipeline_config = pipeline_config or ProcessingPipeline()
        
        # 구성 요소 초기화
        self.single_pattern_processor = SinglePatternProcessor()
        self.chunking_strategy_factory = ChunkingStrategyFactory()
        
        # 처리 통계
        self.processing_stats = {
            "total_patterns": 0,
            "processed_patterns": 0,
            "chunks_created": 0,
            "errors": 0,
            "processing_time": 0.0
        }
        
        self.logger.info("✅ 통합 패턴 처리 엔진 초기화 완료")
    
    def process_patterns(self, analysis_result: PatternAnalysisResult) -> List[ChunkingResult]:
        """
        패턴 분석 결과를 처리하여 청킹 결과 목록을 반환
        
        Args:
            analysis_result: 패턴 분석 결과
            
        Returns:
            청킹 결과 목록
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"🔧 패턴 처리 시작: {analysis_result.total_patterns}개 패턴")
            
            # 통계 초기화
            self.processing_stats["total_patterns"] = analysis_result.total_patterns
            self.processing_stats["processed_patterns"] = 0
            self.processing_stats["chunks_created"] = 0
            self.processing_stats["errors"] = 0
            
            # 패턴별 처리
            chunking_results = []
            
            for i, pattern in enumerate(analysis_result.patterns):
                try:
                    self.logger.debug(f"패턴 {i+1}/{analysis_result.total_patterns} 처리: {pattern.type}")
                    
                    # 단일 패턴 처리
                    if self.pipeline_config.enable_single_pattern_processing:
                        result = self._process_single_pattern(pattern, analysis_result, i)
                    else:
                        result = self._create_default_result(pattern, i)
                    
                    # 청킹 전략 적용
                    if self.pipeline_config.enable_chunking_strategy:
                        result = self._apply_chunking_strategy(result, pattern, i)
                    
                    # 메타데이터 보강
                    if self.pipeline_config.enable_metadata_enrichment:
                        result = self._enrich_metadata(result, pattern, i)
                    
                    # 품질 관리
                    if self.pipeline_config.enable_quality_control:
                        result = self._apply_quality_control(result, pattern, i)
                    
                    chunking_results.append(result)
                    
                    # 통계 업데이트
                    self.processing_stats["processed_patterns"] += 1
                    self.processing_stats["chunks_created"] += len(result.chunks)
                    
                except Exception as e:
                    self.logger.error(f"패턴 {i+1} 처리 중 오류: {e}")
                    self.processing_stats["errors"] += 1
                    
                    # 오류 발생 시 기본 결과 생성
                    error_result = self._create_error_result(pattern, i, str(e))
                    chunking_results.append(error_result)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            self.processing_stats["processing_time"] = processing_time
            
            self.logger.info(f"✅ 패턴 처리 완료: {len(chunking_results)}개 결과, {processing_time:.2f}초")
            
            return chunking_results
            
        except Exception as e:
            self.logger.error(f"패턴 처리 중 치명적 오류: {e}")
            raise
    
    def _process_single_pattern(self, pattern: HeaderInfo, analysis_result: PatternAnalysisResult, index: int) -> ChunkingResult:
        """단일 패턴 처리"""
        try:
            # 처리 컨텍스트 생성
            context = self._create_processing_context(pattern, analysis_result, index)
            
            # 단일 패턴 처리기로 처리
            result = self.single_pattern_processor.process_pattern(pattern, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"단일 패턴 처리 중 오류: {e}")
            return self._create_error_result(pattern, index, str(e))
    
    def _create_processing_context(self, pattern: HeaderInfo, analysis_result: PatternAnalysisResult, index: int) -> ProcessingContext:
        """처리 컨텍스트 생성"""
        # 이전/다음 패턴 정보 수집
        previous_patterns = analysis_result.patterns[:index]
        next_patterns = analysis_result.patterns[index+1:] if index + 1 < len(analysis_result.patterns) else []
        
        # 버퍼 상태 생성
        buffer_state = BufferInfo(
            current_content="",
            metadata={},
            is_active=True
        )
        
        # 메타데이터 생성
        metadata = {
            "line_number": index,
            "total_patterns": analysis_result.total_patterns,
            "pattern_index": index,
            "has_previous": len(previous_patterns) > 0,
            "has_next": len(next_patterns) > 0
        }
        
        return ProcessingContext(
            current_line=pattern.line_text,
            line_number=index,
            previous_patterns=previous_patterns,
            next_patterns=next_patterns,
            buffer_state=buffer_state,
            metadata=metadata
        )
    
    def _apply_chunking_strategy(self, result: ChunkingResult, pattern: HeaderInfo, index: int) -> ChunkingResult:
        """청킹 전략 적용"""
        try:
            # 패턴 타입에 맞는 전략 선택
            strategy = self.chunking_strategy_factory.get_strategy(pattern.type)
            
            # 청킹 컨텍스트 생성
            chunking_context = ChunkingContext(
                pattern=pattern,
                surrounding_text=self._extract_surrounding_text(pattern, index),
                buffer_state=result.buffer_state,
                metadata={"line_number": index},
                chunking_rules={"max_size": self.pipeline_config.max_chunk_size}
            )
            
            # 청킹 전략 적용
            if strategy.should_chunk(chunking_context):
                optimized_chunks = []
                
                for chunk in result.chunks:
                    # 청킹 전략으로 최적화된 청크 생성
                    optimized_chunk = strategy.create_chunk(chunking_context)
                    optimized_chunks.append(optimized_chunk)
                
                # 결과 업데이트
                result.chunks = optimized_chunks
                result.processing_notes += f" | 청킹 전략 적용: {strategy.__class__.__name__}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"청킹 전략 적용 중 오류: {e}")
            return result  # 오류 발생 시 원본 결과 반환
    
    def _extract_surrounding_text(self, pattern: HeaderInfo, index: int) -> List[str]:
        """주변 텍스트 추출"""
        # 간단한 구현 (실제로는 더 정교한 로직 필요)
        return [pattern.line_text] if pattern.line_text else []
    
    def _enrich_metadata(self, result: ChunkingResult, pattern: HeaderInfo, index: int) -> ChunkingResult:
        """메타데이터 보강"""
        try:
            for chunk in result.chunks:
                # 기본 메타데이터 추가
                chunk.metadata.update({
                    "processing_engine": "PatternProcessingEngine",
                    "processing_phase": "Phase2",
                    "pattern_index": index,
                    "processing_timestamp": self._get_timestamp(),
                    "chunk_id": f"chunk_{index}_{id(chunk)}"
                })
                
                # 패턴별 특수 메타데이터
                if hasattr(chunk, 'pattern_type'):
                    chunk.metadata["original_pattern_type"] = chunk.pattern_type
                
                # 크기 정보 추가
                chunk.metadata["text_length"] = len(chunk.text)
                chunk.metadata["word_count"] = len(chunk.text.split())
            
            result.processing_notes += " | 메타데이터 보강 완료"
            return result
            
        except Exception as e:
            self.logger.error(f"메타데이터 보강 중 오류: {e}")
            return result
    
    def _apply_quality_control(self, result: ChunkingResult, pattern: HeaderInfo, index: int) -> ChunkingResult:
        """품질 관리 적용"""
        try:
            quality_issues = []
            
            for chunk in result.chunks:
                # 최소 크기 검사
                if len(chunk.text) < self.pipeline_config.min_chunk_size:
                    quality_issues.append(f"청크 {chunk.metadata.get('chunk_id', 'unknown')}: 최소 크기 미달")
                
                # 최대 크기 검사
                if len(chunk.text) > self.pipeline_config.max_chunk_size:
                    quality_issues.append(f"청크 {chunk.metadata.get('chunk_id', 'unknown')}: 최대 크기 초과")
                
                # 빈 텍스트 검사
                if not chunk.text.strip():
                    quality_issues.append(f"청크 {chunk.metadata.get('chunk_id', 'unknown')}: 빈 텍스트")
            
            # 품질 이슈가 있으면 로그에 기록
            if quality_issues:
                self.logger.warning(f"품질 이슈 발견: {len(quality_issues)}개")
                for issue in quality_issues:
                    self.logger.warning(f"  - {issue}")
                
                result.processing_notes += f" | 품질 이슈: {len(quality_issues)}개"
            else:
                result.processing_notes += " | 품질 검사 통과"
            
            return result
            
        except Exception as e:
            self.logger.error(f"품질 관리 적용 중 오류: {e}")
            return result
    
    def _create_default_result(self, pattern: HeaderInfo, index: int) -> ChunkingResult:
        """기본 결과 생성"""
        chunk = ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": pattern.type,
                "description": pattern.description,
                "processing_method": "default",
                "pattern_index": index
            },
            pattern_type=pattern.type,
            start_line=index,
            end_line=index
        )
        
        return ChunkingResult(
            chunks=[chunk],
            buffer_state=BufferInfo(),
            processing_notes="기본 처리 (단일 패턴 처리 비활성화)"
        )
    
    def _create_error_result(self, pattern: HeaderInfo, index: int, error_message: str) -> ChunkingResult:
        """오류 결과 생성"""
        chunk = ChunkInfo(
            text=pattern.text,
            metadata={
                "pattern_type": pattern.type,
                "description": pattern.description,
                "processing_method": "error",
                "pattern_index": index,
                "error_message": error_message,
                "error_occurred": True
            },
            pattern_type=pattern.type,
            start_line=index,
            end_line=index
        )
        
        return ChunkingResult(
            chunks=[chunk],
            buffer_state=BufferInfo(),
            processing_notes=f"오류 발생: {error_message}"
        )
    
    def _get_timestamp(self) -> str:
        """타임스탬프 생성"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.processing_stats = {
            "total_patterns": 0,
            "processed_patterns": 0,
            "chunks_created": 0,
            "errors": 0,
            "processing_time": 0.0
        }
        self.logger.info("📊 처리 통계 초기화 완료")
    
    def get_pipeline_config(self) -> ProcessingPipeline:
        """파이프라인 설정 반환"""
        return self.pipeline_config
    
    def update_pipeline_config(self, new_config: ProcessingPipeline):
        """파이프라인 설정 업데이트"""
        self.pipeline_config = new_config
        self.logger.info("⚙️ 파이프라인 설정 업데이트 완료")
    
    def validate_pipeline_config(self) -> List[str]:
        """파이프라인 설정 유효성 검사"""
        validation_errors = []
        
        if self.pipeline_config.max_chunk_size <= 0:
            validation_errors.append("최대 청크 크기는 0보다 커야 합니다")
        
        if self.pipeline_config.min_chunk_size < 0:
            validation_errors.append("최소 청크 크기는 0 이상이어야 합니다")
        
        if self.pipeline_config.max_chunk_size < self.pipeline_config.min_chunk_size:
            validation_errors.append("최대 청크 크기는 최소 청크 크기보다 커야 합니다")
        
        return validation_errors
    
    def get_available_strategies(self) -> Dict[str, str]:
        """사용 가능한 전략 목록 반환"""
        strategies = self.chunking_strategy_factory.get_all_strategies()
        return {pattern_type: strategy.__class__.__name__ 
                for pattern_type, strategy in strategies.items()}
    
    def register_custom_strategy(self, pattern_type: str, strategy):
        """사용자 정의 전략 등록"""
        try:
            self.chunking_strategy_factory.register_strategy(pattern_type, strategy)
            self.logger.info(f"사용자 정의 전략 등록 완료: {pattern_type}")
            return True
        except Exception as e:
            self.logger.error(f"사용자 정의 전략 등록 실패: {e}")
            return False
