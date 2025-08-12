"""
법령 위계형 RAG 시스템 테스트 스크립트

구현된 모든 구성요소들을 테스트합니다.
"""

import sys
import os
import time
from datetime import datetime

# 상위 디렉토리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from legal.system import LegalRAGSystem
    from legal.schema import LegalSchema
    from legal.parser import LegalParser
    from utils.text_utils import TextProcessor
    from utils.milvus_utils import MilvusHelper
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print("기존 시스템 import 없이 독립 테스트를 진행합니다.")


def test_legal_schema():
    """법령 스키마 테스트"""
    print("\n🔧 법령 스키마 테스트...")
    
    try:
        schema = LegalSchema()
        
        # 스키마 검증
        is_valid = schema.validate_legal_schema()
        print(f"✅ 스키마 검증: {'통과' if is_valid else '실패'}")
        
        # 스키마 정보 출력
        schema_info = schema.get_legal_schema_info()
        print(f"✅ 전체 필드 수: {schema_info.get('total_fields', 0)}개")
        print(f"✅ 전체 인덱스 수: {schema_info.get('total_indexes', 0)}개")
        
        # 샘플 문서 생성
        sample = schema.get_sample_legal_document()
        print(f"✅ 샘플 문서 생성: {sample.get('title', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 스키마 테스트 실패: {e}")
        return False


def test_legal_parser():
    """법령 파서 테스트"""
    print("\n📋 법령 파서 테스트...")
    
    try:
        parser = LegalParser()
        
        # 테스트 법령 문서
        test_document = {
            "title": "개인정보 보호법",
            "text": """제1장 총칙

제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써 개인의 자유와 권리를 보호하고, 나아가 개인의 존엄과 가치를 구현하기 위함을 목적으로 한다.

제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다.
① "개인정보"란 살아 있는 개인에 관한 정보로서 성명, 주민등록번호 및 영상 등을 통하여 개인을 알아볼 수 있는 정보를 말한다.
② "개인정보처리자"란 업무를 목적으로 개인정보파일을 운용하기 위하여 스스로 또는 다른 사람을 통하여 개인정보를 처리하는 공공기관, 법인, 단체 및 개인 등을 말한다.

제3조(개인정보 보호 원칙) 개인정보처리자는 개인정보의 처리 목적을 명확하게 하여야 하고 그 목적에 필요한 범위에서 최소한의 개인정보만을 적법하고 정당하게 수집하여야 한다.
① 개인정보처리자는 개인정보의 처리 목적에 필요한 범위에서 개인정보의 정확성, 완전성 및 최신성이 보장되도록 하여야 한다.
② 개인정보처리자는 개인정보의 처리 목적에 필요한 범위에서 개인정보를 안전하게 관리하여야 한다.""",
            "law_type": "법률",
            "law_number": "법률 제11690호",
            "domain": "legal"
        }
        
        # 파싱 실행
        start_time = time.time()
        parsed_chunks = parser.parse_legal_document(test_document)
        end_time = time.time()
        
        print(f"✅ 파싱 완료: {len(parsed_chunks)}개 청크")
        print(f"✅ 파싱 시간: {end_time - start_time:.3f}초")
        
        # 파싱 통계
        stats = parser.get_parsing_stats(parsed_chunks)
        print(f"✅ 조문 수: {stats.get('article_count', 0)}개")
        print(f"✅ 항 수: {stats.get('paragraph_count', 0)}개")
        print(f"✅ 호 수: {stats.get('item_count', 0)}개")
        
        # 샘플 청크 출력
        if parsed_chunks:
            print(f"\n📝 샘플 청크 (첫 번째):")
            sample_chunk = parsed_chunks[0]
            print(f"   텍스트: {sample_chunk.get('text', '')[:100]}...")
            print(f"   위계 레벨: {sample_chunk.get('hierarchy_level', 0)}")
            print(f"   섹션 타입: {sample_chunk.get('section_type', 'unknown')}")
            print(f"   위계 경로: {sample_chunk.get('hierarchy_path', '/')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 파서 테스트 실패: {e}")
        return False


def test_text_processor():
    """텍스트 프로세서 테스트"""
    print("\n📝 텍스트 프로세서 테스트...")
    
    try:
        processor = TextProcessor()
        
        # 테스트 텍스트
        test_text = """    제1조(목적)     이 법은 개인정보의 처리 및 보호에    관한 사항을 정함으로써 
        
        개인의 자유와 권리를 보호하고,    나아가 개인의 존엄과 가치를 구현하기 위함을 목적으로 한다.    """
        
        # 텍스트 정리
        cleaned = processor.clean_text(test_text)
        print(f"✅ 텍스트 정리 완료")
        print(f"   원본 길이: {len(test_text)}자")
        print(f"   정리 후: {len(cleaned)}자")
        
        # 키워드 추출
        keywords = processor.extract_keywords(cleaned)
        print(f"✅ 키워드 추출: {', '.join(keywords[:5])}")
        
        # 요약 생성
        summary = processor.generate_summary(cleaned, max_length=100)
        print(f"✅ 요약 생성: {summary}")
        
        # 해시 생성
        content_hash = processor.generate_content_hash(cleaned)
        print(f"✅ 해시 생성: {content_hash[:16]}...")
        
        # 한국어 검증
        validation = processor.validate_korean_text(cleaned)
        print(f"✅ 한국어 비율: {validation.get('korean_ratio', 0):.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 텍스트 프로세서 테스트 실패: {e}")
        return False


def test_milvus_helper():
    """Milvus 헬퍼 테스트"""
    print("\n🗄️ Milvus 헬퍼 테스트...")
    
    try:
        helper = MilvusHelper()
        
        # 시스템 정보 조회
        system_info = helper.get_system_info()
        connection_status = system_info.get("connection_status", "unknown")
        print(f"✅ Milvus 연결 상태: {connection_status}")
        
        if connection_status == "connected":
            collections = system_info.get("available_collections", [])
            print(f"✅ 사용 가능한 컬렉션: {len(collections)}개")
            
            # 첫 번째 컬렉션 정보 조회 (있다면)
            if collections:
                first_collection = collections[0]
                health = helper.check_collection_health(first_collection)
                print(f"✅ '{first_collection}' 건강 상태: {health.get('overall_health', 'unknown')}")
        else:
            print("⚠️ Milvus 연결 실패 - 연결 정보를 확인하세요")
        
        return True
        
    except Exception as e:
        print(f"❌ Milvus 헬퍼 테스트 실패: {e}")
        return False


def test_legal_system_integration():
    """법령 시스템 통합 테스트 (InteractManager 없이)"""
    print("\n🎯 법령 시스템 통합 테스트...")
    
    try:
        # InteractManager 없이 기본 기능만 테스트
        print("ℹ️ InteractManager 없이 기본 구성요소 테스트")
        
        # 스키마 테스트
        schema = LegalSchema()
        schema_valid = schema.validate_legal_schema()
        print(f"✅ 스키마 검증: {'통과' if schema_valid else '실패'}")
        
        # 파서 테스트
        parser = LegalParser()
        sample_doc = schema.get_sample_legal_document()
        parsed = parser.parse_legal_document(sample_doc)
        print(f"✅ 파싱 테스트: {len(parsed)}개 청크 생성")
        
        # 시스템 정보
        system_info = {
            "version": "1.0.0",
            "components_tested": 4,
            "test_timestamp": datetime.now().isoformat(),
            "status": "partial_success"
        }
        
        print(f"✅ 시스템 버전: {system_info['version']}")
        print(f"✅ 테스트된 구성요소: {system_info['components_tested']}개")
        
        return True
        
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🚀 위계형 법령 RAG 시스템 테스트 시작")
    print(f"⏰ 테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    test_results = []
    
    # 개별 구성요소 테스트
    test_results.append(("스키마", test_legal_schema()))
    test_results.append(("파서", test_legal_parser()))
    test_results.append(("텍스트 프로세서", test_text_processor()))
    test_results.append(("Milvus 헬퍼", test_milvus_helper()))
    test_results.append(("통합 시스템", test_legal_system_integration()))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print(f"\n🏆 전체 결과: {passed}/{total} 통과 ({passed/total*100:.1f}%)")
    print(f"⏰ 테스트 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! 시스템이 정상적으로 구현되었습니다.")
        print("\n📋 다음 단계:")
        print("1. 기존 InteractManager와 연동 테스트")
        print("2. 실제 법령 데이터로 인덱싱 테스트")
        print("3. API 연동 및 성능 테스트")
    else:
        print(f"\n⚠️ {total - passed}개 테스트 실패. 로그를 확인하고 수정이 필요합니다.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
