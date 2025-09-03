"""
Phase 3: 복합 패턴 처리 시스템 통합 테스트

Phase 3의 모든 컴포넌트가 올바르게 작동하는지 검증합니다.
"""

import sys
import os
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_complex_pattern_analyzer():
    """복합 패턴 분석기 테스트"""
    print("🔍 복합 패턴 분석기 테스트 시작...")
    
    try:
        from complex_pattern_analyzer import ComplexPatternAnalyzer, PatternComplexity
        from data_structures import PatternAnalysisResult, HeaderInfo
        
        analyzer = ComplexPatternAnalyzer()
        
        # 테스트용 패턴 분석 결과 생성
        analysis_result = PatternAnalysisResult()
        
        # 계층적 패턴들 추가
        patterns = [
            HeaderInfo(type="chapter", description="장", text="제1장", start=0, end=4, 
                      line_number=0, line_text="제1장 총칙", groups=("1", "총칙")),
            HeaderInfo(type="section", description="절", text="제1절", start=0, end=4, 
                      line_number=1, line_text="제1절 목적", groups=("1", "목적")),
            HeaderInfo(type="article", description="조", text="제1조", start=0, end=4, 
                      line_number=2, line_text="제1조(목적) 이 법은 도서관 자료를 수집한다.", groups=("1", "", "목적")),
            HeaderInfo(type="paragraph", description="항", text="①", start=0, end=1, 
                      line_number=3, line_text="① 도서관 자료의 수집", groups=("①",)),
            HeaderInfo(type="subparagraph", description="호", text="1.", start=0, end=2, 
                      line_number=4, line_text="1. 도서", groups=("1",)),
            HeaderInfo(type="item", description="목", text="가.", start=0, end=2, 
                      line_number=5, line_text="가. 도서", groups=("가",))
        ]
        
        for pattern in patterns:
            analysis_result.add_pattern(pattern)
        
        # 복합 패턴 분석 수행
        complex_analysis = analyzer.analyze_complex_patterns(analysis_result)
        
        print(f"✅ 복합 패턴 분석 완료:")
        print(f"  - 복잡도: {complex_analysis.complexity.value}")
        print(f"  - 계층 깊이: {complex_analysis.hierarchy_depth}")
        print(f"  - 연속 범위: {len(complex_analysis.continuous_ranges)}개")
        print(f"  - 패턴 관계: {len(complex_analysis.pattern_relations)}개")
        
        # 분석 요약 확인
        summary = analyzer.get_pattern_summary(complex_analysis)
        print(f"  - 총 패턴: {summary['total_patterns']}개")
        print(f"  - 패턴 타입: {summary['pattern_types']}")
        
        # 복잡도가 계층적이어야 함
        if complex_analysis.complexity == PatternComplexity.HIERARCHICAL:
            print("✅ 계층적 패턴 복잡도 정확히 감지됨")
            return True
        else:
            print(f"❌ 예상된 복잡도: HIERARCHICAL, 실제: {complex_analysis.complexity.value}")
            return False
            
    except Exception as e:
        print(f"❌ 복합 패턴 분석기 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complex_pattern_processor():
    """복합 패턴 처리기 테스트"""
    print("\n🔧 복합 패턴 처리기 테스트 시작...")
    
    try:
        from complex_pattern_processor import ComplexPatternProcessor, ChunkingContext
        from complex_pattern_analyzer import PatternComplexity, PatternRelation
        from data_structures import PatternAnalysisResult, HeaderInfo
        
        processor = ComplexPatternProcessor()
        
        # 테스트용 패턴 분석 결과 생성
        analysis_result = PatternAnalysisResult()
        
        # 중첩 패턴들 추가
        patterns = [
            HeaderInfo(type="chapter", description="장", text="제1장", start=0, end=4, 
                      line_number=0, line_text="제1장 총칙", groups=("1", "총칙")),
            HeaderInfo(type="article", description="조", text="제1조", start=0, end=4, 
                      line_number=1, line_text="제1조(목적) 이 법은 도서관 자료를 수집한다.", groups=("1", "", "목적")),
            HeaderInfo(type="paragraph", description="항", text="①", start=0, end=1, 
                      line_number=2, line_text="① 도서관 자료의 수집", groups=("①",))
        ]
        
        for pattern in patterns:
            analysis_result.add_pattern(pattern)
        
        # 청킹 컨텍스트 생성
        context = ChunkingContext(
            complexity=PatternComplexity.NESTED,
            hierarchy_depth=3,
            continuous_ranges=[],
            pattern_relations={},
            target_chunk_size=1000,
            max_chunk_size=2000,
            min_chunk_size=200
        )
        
        # 복합 패턴 처리 수행
        chunking_result = processor.process_complex_patterns(analysis_result, context)
        
        print(f"✅ 복합 패턴 처리 완료:")
        print(f"  - 생성된 청크: {len(chunking_result.chunks)}개")
        print(f"  - 처리 노트: {chunking_result.processing_notes}")
        
        # 청크 내용 확인
        for i, chunk in enumerate(chunking_result.chunks):
            print(f"  - 청크 {i+1}: {chunk.content_type}, 크기: {len(chunk.content)}자")
            print(f"    메타데이터: {chunk.metadata}")
        
        # 처리 요약 확인
        summary = processor.get_processing_summary(chunking_result)
        print(f"  - 총 청크: {summary['total_chunks']}개")
        print(f"  - 청크 타입: {summary['chunk_types']}")
        print(f"  - 전략: {summary['strategies']}")
        
        if len(chunking_result.chunks) > 0:
            print("✅ 복합 패턴 처리 성공")
            return True
        else:
            print("❌ 청크 생성 실패")
            return False
            
    except Exception as e:
        print(f"❌ 복합 패턴 처리기 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase3_processing_engine():
    """Phase 3 통합 처리 엔진 테스트"""
    print("\n🚀 Phase 3 통합 처리 엔진 테스트 시작...")
    
    try:
        from phase3_processing_engine import Phase3ProcessingEngine, Phase3Config
        
        # Phase 3 설정
        config = Phase3Config(
            enable_complex_analysis=True,
            enable_adaptive_chunking=True,
            enable_quality_control=True,
            enable_performance_monitoring=True,
            target_chunk_size=1000,
            max_chunk_size=2000,
            min_chunk_size=200
        )
        
        # Phase 3 엔진 초기화
        engine = Phase3ProcessingEngine(config)
        
        # 테스트용 텍스트 (복합 패턴 포함)
        test_text = """제1장 총칙
제1절 목적
제1조(목적) 이 법은 도서관 자료를 수집한다.
① 도서관 자료의 수집
1. 도서
가. 도서
나. 잡지
2. 전자자료
제2조(정의) 이 법에서 "도서관"이라 함은 도서관법에 따른 도서관을 말한다.
제3조부터 제5조까지는 생략한다.
부칙 제1조
별지 제1호"""
        
        print(f"📝 테스트 텍스트:")
        print(test_text)
        print()
        
        # Phase 3 처리 수행
        chunking_result = engine.process_text(test_text)
        
        print(f"✅ Phase 3 처리 완료:")
        print(f"  - 생성된 청크: {len(chunking_result.chunks)}개")
        print(f"  - 처리 노트: {chunking_result.processing_notes}")
        
        # 청크 내용 확인
        for i, chunk in enumerate(chunking_result.chunks):
            print(f"  - 청크 {i+1}: {chunk.content_type}")
            print(f"    헤더: {chunk.header.text if chunk.header else 'N/A'}")
            print(f"    크기: {len(chunk.content)}자")
            print(f"    메타데이터: {chunk.metadata}")
            print()
        
        # 처리 요약 확인
        summary = engine.get_processing_summary()
        print(f"📊 처리 요약:")
        print(f"  - Phase: {summary['phase']}")
        print(f"  - 총 처리 시간: {summary['metrics']['total_processing_time']:.2f}초")
        print(f"  - 총 패턴: {summary['statistics']['total_patterns']}개")
        print(f"  - 총 청크: {summary['statistics']['total_chunks']}개")
        print(f"  - 평균 청크 크기: {summary['statistics']['average_chunk_size']:.1f}자")
        
        # 성능 보고서 내보내기
        report = engine.export_processing_report()
        print(f"\n📋 성능 보고서:")
        print(report)
        
        if len(chunking_result.chunks) > 0:
            print("✅ Phase 3 통합 처리 성공")
            return True
        else:
            print("❌ Phase 3 처리 실패")
            return False
            
    except Exception as e:
        print(f"❌ Phase 3 통합 처리 엔진 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complex_pattern_scenarios():
    """복합 패턴 시나리오 테스트"""
    print("\n🎯 복합 패턴 시나리오 테스트 시작...")
    
    try:
        from phase3_processing_engine import Phase3ProcessingEngine, Phase3Config
        
        engine = Phase3ProcessingEngine()
        
        # 시나리오 1: 중첩 패턴
        print("📋 시나리오 1: 중첩 패턴")
        nested_text = """제1장 총칙
제1조(목적) 이 법은 도서관 자료를 수집한다.
① 도서관 자료의 수집
1. 도서
가. 도서
나. 잡지
2. 전자자료"""
        
        result1 = engine.process_text(nested_text)
        print(f"  - 중첩 패턴 결과: {len(result1.chunks)}개 청크")
        
        # 시나리오 2: 연속 패턴
        print("📋 시나리오 2: 연속 패턴")
        continuous_text = """제1조(목적) 이 법은 도서관 자료를 수집한다.
제2조(정의) 이 법에서 "도서관"이라 함은 도서관법에 따른 도서관을 말한다.
제3조(적용범위) 이 법은 모든 도서관에 적용한다.
제4조부터 제6조까지는 생략한다."""
        
        result2 = engine.process_text(continuous_text)
        print(f"  - 연속 패턴 결과: {len(result2.chunks)}개 청크")
        
        # 시나리오 3: 혼합 패턴
        print("📋 시나리오 3: 혼합 패턴")
        mixed_text = """제1장 총칙
제1절 목적
제1조(목적) 이 법은 도서관 자료를 수집한다.
제2조(정의) 이 법에서 "도서관"이라 함은 도서관법에 따른 도서관을 말한다.
① 도서관 자료의 수집
1. 도서
가. 도서
나. 잡지
제3조부터 제5조까지는 생략한다.
부칙 제1조"""
        
        result3 = engine.process_text(mixed_text)
        print(f"  - 혼합 패턴 결과: {len(result3.chunks)}개 청크")
        
        # 모든 시나리오 성공 확인
        success = (len(result1.chunks) > 0 and 
                  len(result2.chunks) > 0 and 
                  len(result3.chunks) > 0)
        
        if success:
            print("✅ 모든 복합 패턴 시나리오 성공")
            return True
        else:
            print("❌ 일부 시나리오 실패")
            return False
            
    except Exception as e:
        print(f"❌ 복합 패턴 시나리오 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 Phase 3 복합 패턴 처리 시스템 통합 테스트 시작!")
    print("=" * 70)
    
    test_results = []
    
    try:
        # 1. 복합 패턴 분석기 테스트
        result1 = test_complex_pattern_analyzer()
        test_results.append(("복합 패턴 분석기", result1))
        
        # 2. 복합 패턴 처리기 테스트
        result2 = test_complex_pattern_processor()
        test_results.append(("복합 패턴 처리기", result2))
        
        # 3. Phase 3 통합 처리 엔진 테스트
        result3 = test_phase3_processing_engine()
        test_results.append(("Phase 3 통합 처리 엔진", result3))
        
        # 4. 복합 패턴 시나리오 테스트
        result4 = test_complex_pattern_scenarios()
        test_results.append(("복합 패턴 시나리오", result4))
        
        # 결과 요약
        print("\n" + "=" * 70)
        print("📊 Phase 3 통합 테스트 결과 요약:")
        
        success_count = 0
        for test_name, result in test_results:
            status = "✅ 성공" if result else "❌ 실패"
            print(f"  {test_name}: {status}")
            if result:
                success_count += 1
        
        print(f"\n🎯 전체 테스트: {success_count}/{len(test_results)} 성공")
        
        if success_count == len(test_results):
            print("🎉 Phase 3 시스템 통합 테스트 완료!")
            print("🚀 복합 패턴 처리 시스템이 정상적으로 작동합니다!")
            print("🔧 모든 컴포넌트가 올바르게 통합되었습니다!")
        else:
            print("⚠️ 일부 테스트가 실패했습니다. Phase 3 시스템을 확인해주세요.")
        
    except Exception as e:
        print(f"\n❌ Phase 3 통합 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
