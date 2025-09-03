"""
Phase 3: 복합 패턴 처리 통합 엔진

Phase 3의 모든 컴포넌트를 통합하여 복합 패턴 처리를 수행합니다.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .pattern_scanner import PatternScanner
from .pattern_classifier import PatternClassifier
from .complex_pattern_analyzer import ComplexPatternAnalyzer, ComplexPatternAnalysis, PatternComplexity
from .complex_pattern_processor import ComplexPatternProcessor, ChunkingContext
from .data_structures import PatternAnalysisResult, HeaderInfo, ChunkingResult

@dataclass
class Phase3Config:
    """Phase 3 설정"""
    enable_complex_analysis: bool = True
    enable_adaptive_chunking: bool = True
    enable_quality_control: bool = True
    enable_performance_monitoring: bool = True
    
    # 청킹 설정
    target_chunk_size: int = 1000
    max_chunk_size: int = 2000
    min_chunk_size: int = 200
    
    # 성능 설정
    enable_parallel_processing: bool = False
    max_workers: int = 4

@dataclass
class ProcessingMetrics:
    """처리 성능 메트릭"""
    total_processing_time: float = 0.0
    pattern_analysis_time: float = 0.0
    complex_analysis_time: float = 0.0
    chunking_time: float = 0.0
    quality_control_time: float = 0.0
    
    # 패턴 통계
    total_patterns: int = 0
    complex_patterns: int = 0
    total_chunks: int = 0
    
    # 품질 지표
    average_chunk_size: float = 0.0
    chunk_size_variance: float = 0.0
    pattern_coverage: float = 0.0

class Phase3ProcessingEngine:
    """Phase 3 통합 처리 엔진"""
    
    def __init__(self, config: Phase3Config = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Phase 3 통합 처리 엔진 초기화")
        
        # 설정 초기화
        self.config = config or Phase3Config()
        
        # 컴포넌트 초기화
        self.pattern_scanner = PatternScanner()
        self.pattern_classifier = PatternClassifier()
        self.complex_analyzer = ComplexPatternAnalyzer()
        self.complex_processor = ComplexPatternProcessor()
        
        # 성능 모니터링
        self.metrics = ProcessingMetrics()
        
        self.logger.info("✅ Phase 3 통합 처리 엔진 초기화 완료")
    
    def process_text(self, text: str) -> ChunkingResult:
        """
        텍스트를 복합 패턴 처리하여 청킹
        
        Args:
            text: 처리할 텍스트
            
        Returns:
            청킹 결과
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 Phase 3 복합 패턴 처리 시작: {len(text)}자")
            
            # 1. 기본 패턴 스캔 및 분류
            pattern_analysis_time = time.time()
            basic_result = self._perform_basic_pattern_analysis(text)
            self.metrics.pattern_analysis_time = time.time() - pattern_analysis_time
            
            # 2. 복합 패턴 분석
            complex_analysis_time = time.time()
            complex_analysis = self._perform_complex_pattern_analysis(basic_result)
            self.metrics.complex_analysis_time = time.time() - complex_analysis_time
            
            # 3. 복합 패턴 처리 및 청킹
            chunking_time = time.time()
            chunking_result = self._perform_complex_chunking(basic_result, complex_analysis)
            self.metrics.chunking_time = time.time() - chunking_time
            
            # 4. 품질 관리
            quality_time = time.time()
            if self.config.enable_quality_control:
                chunking_result = self._apply_quality_control(chunking_result)
            self.metrics.quality_control_time = time.time() - quality_time
            
            # 5. 성능 메트릭 업데이트
            self._update_metrics(basic_result, complex_analysis, chunking_result)
            
            # 6. 최종 결과 반환
            total_time = time.time() - start_time
            self.metrics.total_processing_time = total_time
            
            self.logger.info(f"✅ Phase 3 복합 패턴 처리 완료: {len(chunking_result.chunks)}개 청크, {total_time:.2f}초")
            
            return chunking_result
            
        except Exception as e:
            self.logger.error(f"Phase 3 처리 중 오류: {e}")
            return self._create_error_result(f"Phase 3 처리 실패: {e}")
    
    def _perform_basic_pattern_analysis(self, text: str) -> PatternAnalysisResult:
        """기본 패턴 분석 수행"""
        try:
            # 텍스트를 라인별로 분리
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # 패턴 스캔
            all_patterns = self.pattern_scanner.scan_multiple_lines(lines)
            
            # PatternAnalysisResult로 변환
            analysis_result = PatternAnalysisResult()
            for pattern in all_patterns:
                # HeaderInfo 객체로 변환
                header_info = HeaderInfo(
                    type=pattern['type'],
                    description=pattern['description'],
                    text=pattern['text'],
                    start=pattern['start'],
                    end=pattern['end'],
                    line_number=pattern['line_number'],
                    line_text=pattern['line_text'],
                    groups=pattern['groups']
                )
                
                # 상태 플래그 추가
                if 'status_flags' in pattern:
                    header_info.status_flags = pattern['status_flags']
                
                analysis_result.add_pattern(header_info)
            
            self.logger.debug(f"기본 패턴 분석 완료: {len(analysis_result.patterns)}개 패턴")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"기본 패턴 분석 중 오류: {e}")
            raise
    
    def _perform_complex_pattern_analysis(self, basic_result: PatternAnalysisResult) -> ComplexPatternAnalysis:
        """복합 패턴 분석 수행"""
        try:
            if not self.config.enable_complex_analysis:
                self.logger.debug("복합 패턴 분석 비활성화")
                return self._create_default_complex_analysis()
            
            complex_analysis = self.complex_analyzer.analyze_complex_patterns(basic_result)
            self.logger.debug(f"복합 패턴 분석 완료: {complex_analysis.complexity.value}")
            return complex_analysis
            
        except Exception as e:
            self.logger.error(f"복합 패턴 분석 중 오류: {e}")
            return self._create_default_complex_analysis()
    
    def _perform_complex_chunking(self, basic_result: PatternAnalysisResult, 
                                 complex_analysis: ComplexPatternAnalysis) -> ChunkingResult:
        """복합 패턴 청킹 수행"""
        try:
            # 청킹 컨텍스트 생성
            chunking_context = ChunkingContext(
                complexity=complex_analysis.complexity,
                hierarchy_depth=complex_analysis.hierarchy_depth,
                continuous_ranges=complex_analysis.continuous_ranges,
                pattern_relations=complex_analysis.pattern_relations,
                target_chunk_size=self.config.target_chunk_size,
                max_chunk_size=self.config.max_chunk_size,
                min_chunk_size=self.config.min_chunk_size
            )
            
            # 복합 패턴 처리
            chunking_result = self.complex_processor.process_complex_patterns(
                basic_result, chunking_context
            )
            
            self.logger.debug(f"복합 패턴 청킹 완료: {len(chunking_result.chunks)}개 청크")
            return chunking_result
            
        except Exception as e:
            self.logger.error(f"복합 패턴 청킹 중 오류: {e}")
            return self._create_error_result(f"복합 패턴 청킹 실패: {e}")
    
    def _apply_quality_control(self, chunking_result: ChunkingResult) -> ChunkingResult:
        """품질 관리 적용"""
        try:
            if not chunking_result.chunks:
                return chunking_result
            
            # 청크 크기 검증
            valid_chunks = []
            for chunk in chunking_result.chunks:
                if self._is_chunk_valid(chunk):
                    valid_chunks.append(chunk)
                else:
                    self.logger.warning(f"유효하지 않은 청크 제거: {chunk.chunk_id}")
            
            # 청킹 결과 업데이트
            chunking_result.chunks = valid_chunks
            chunking_result.processing_notes += " | 품질 관리 적용"
            
            self.logger.debug(f"품질 관리 완료: {len(valid_chunks)}개 유효 청크")
            return chunking_result
            
        except Exception as e:
            self.logger.error(f"품질 관리 적용 중 오류: {e}")
            return chunking_result
    
    def _is_chunk_valid(self, chunk: Any) -> bool:
        """청크 유효성 검증"""
        try:
            # 기본 검증
            if not chunk or not chunk.content:
                return False
            
            # 크기 검증
            content_length = len(chunk.content)
            if content_length < self.config.min_chunk_size:
                return False
            if content_length > self.config.max_chunk_size:
                return False
            
            # 메타데이터 검증
            if not hasattr(chunk, 'metadata') or not chunk.metadata:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"청크 유효성 검증 중 오류: {e}")
            return False
    
    def _update_metrics(self, basic_result: PatternAnalysisResult, 
                       complex_analysis: ComplexPatternAnalysis, 
                       chunking_result: ChunkingResult):
        """성능 메트릭 업데이트"""
        try:
            # 패턴 통계
            self.metrics.total_patterns = len(basic_result.patterns)
            self.metrics.complex_patterns = len(complex_analysis.patterns)
            self.metrics.total_chunks = len(chunking_result.chunks)
            
            # 청크 크기 통계
            if chunking_result.chunks:
                chunk_sizes = [len(chunk.content) for chunk in chunking_result.chunks]
                self.metrics.average_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
                
                # 분산 계산
                if len(chunk_sizes) > 1:
                    mean = self.metrics.average_chunk_size
                    variance = sum((size - mean) ** 2 for size in chunk_sizes) / (len(chunk_sizes) - 1)
                    self.metrics.chunk_size_variance = variance
            
            # 패턴 커버리지
            if self.metrics.total_patterns > 0:
                self.metrics.pattern_coverage = self.metrics.total_chunks / self.metrics.total_patterns
            
        except Exception as e:
            self.logger.error(f"메트릭 업데이트 중 오류: {e}")
    
    def _create_default_complex_analysis(self) -> ComplexPatternAnalysis:
        """기본 복합 패턴 분석 결과 생성"""
        from .complex_pattern_analyzer import PatternComplexity
        
        return ComplexPatternAnalysis(
            patterns=[],
            complexity=PatternComplexity.SIMPLE,
            hierarchy_depth=0,
            continuous_ranges=[],
            pattern_relations={},
            analysis_notes="기본 분석 (복합 분석 비활성화)"
        )
    
    def _create_error_result(self, error_message: str) -> ChunkingResult:
        """오류 결과 생성"""
        return ChunkingResult(
            chunks=[],
            processing_notes=f"오류: {error_message}"
        )
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """처리 결과 요약 반환"""
        try:
            return {
                "phase": "Phase3",
                "metrics": {
                    "total_processing_time": self.metrics.total_processing_time,
                    "pattern_analysis_time": self.metrics.pattern_analysis_time,
                    "complex_analysis_time": self.metrics.complex_analysis_time,
                    "chunking_time": self.metrics.chunking_time,
                    "quality_control_time": self.metrics.quality_control_time
                },
                "statistics": {
                    "total_patterns": self.metrics.total_patterns,
                    "complex_patterns": self.metrics.complex_patterns,
                    "total_chunks": self.metrics.total_chunks,
                    "average_chunk_size": self.metrics.average_chunk_size,
                    "chunk_size_variance": self.metrics.chunk_size_variance,
                    "pattern_coverage": self.metrics.pattern_coverage
                },
                "configuration": {
                    "enable_complex_analysis": self.config.enable_complex_analysis,
                    "enable_adaptive_chunking": self.config.enable_adaptive_chunking,
                    "enable_quality_control": self.config.enable_quality_control,
                    "target_chunk_size": self.config.target_chunk_size,
                    "max_chunk_size": self.config.max_chunk_size,
                    "min_chunk_size": self.config.min_chunk_size
                }
            }
            
        except Exception as e:
            self.logger.error(f"처리 요약 생성 중 오류: {e}")
            return {"error": str(e)}
    
    def reset_metrics(self):
        """성능 메트릭 초기화"""
        self.metrics = ProcessingMetrics()
        self.logger.info("성능 메트릭 초기화 완료")
    
    def optimize_configuration(self, target_performance: Dict[str, float] = None):
        """설정 최적화"""
        try:
            if not target_performance:
                return
            
            # 청크 크기 최적화
            if "target_chunk_size" in target_performance:
                self.config.target_chunk_size = int(target_performance["target_chunk_size"])
            
            if "max_chunk_size" in target_performance:
                self.config.max_chunk_size = int(target_performance["max_chunk_size"])
            
            if "min_chunk_size" in target_performance:
                self.config.min_chunk_size = int(target_performance["min_chunk_size"])
            
            self.logger.info(f"설정 최적화 완료: {target_performance}")
            
        except Exception as e:
            self.logger.error(f"설정 최적화 중 오류: {e}")
    
    def export_processing_report(self) -> str:
        """처리 보고서 내보내기"""
        try:
            summary = self.get_processing_summary()
            
            report = f"""
=== Phase 3 복합 패턴 처리 보고서 ===

📊 성능 메트릭:
- 총 처리 시간: {summary['metrics']['total_processing_time']:.2f}초
- 패턴 분석 시간: {summary['metrics']['pattern_analysis_time']:.2f}초
- 복합 분석 시간: {summary['metrics']['complex_analysis_time']:.2f}초
- 청킹 시간: {summary['metrics']['chunking_time']:.2f}초
- 품질 관리 시간: {summary['metrics']['quality_control_time']:.2f}초

📈 통계:
- 총 패턴 수: {summary['statistics']['total_patterns']}
- 복합 패턴 수: {summary['statistics']['complex_patterns']}
- 총 청크 수: {summary['statistics']['total_chunks']}
- 평균 청크 크기: {summary['statistics']['average_chunk_size']:.1f}자
- 청크 크기 분산: {summary['statistics']['chunk_size_variance']:.1f}
- 패턴 커버리지: {summary['statistics']['pattern_coverage']:.2f}

⚙️ 설정:
- 복합 분석 활성화: {summary['configuration']['enable_complex_analysis']}
- 적응적 청킹 활성화: {summary['configuration']['enable_adaptive_chunking']}
- 품질 관리 활성화: {summary['configuration']['enable_quality_control']}
- 목표 청크 크기: {summary['configuration']['target_chunk_size']}자
- 최대 청크 크기: {summary['configuration']['max_chunk_size']}자
- 최소 청크 크기: {summary['configuration']['min_chunk_size']}자

생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"처리 보고서 내보내기 중 오류: {e}")
            return f"보고서 생성 실패: {e}"
