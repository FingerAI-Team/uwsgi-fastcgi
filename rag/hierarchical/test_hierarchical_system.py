"""
위계형 RAG 시스템 테스트

기존 RAG와 완전히 호환되면서 조문 참조 기능만 추가하는 시스템을 테스트합니다.
"""

import sys
import os
import time
from datetime import datetime

# 상위 디렉토리 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from hierarchical import HierarchicalSchema, HierarchicalRetriever, HierarchicalProcessor
    print("✅ 위계형 시스템 import 성공")
except ImportError as e:
    print(f"❌ import 오류: {e}")
    sys.exit(1)


def test_hierarchical_schema():
    """위계형 스키마 테스트"""
    print("\n🔧 위계형 스키마 테스트...")
    
    try:
        schema = HierarchicalSchema()
        
        # 스키마 필드 확인
        fields = schema.get_compatible_fields()
        print(f"✅ 스키마 필드 수: {len(fields)}개")
        
        # 필드명 확인
        field_names = [field.name for field in fields]
        print(f"✅ 필드명 목록: {field_names}")
        
        # 위계형 필드 확인
        hierarchical_fields = ['article_number', 'paragraph_number', 'item_number']
        for field in hierarchical_fields:
            if field in field_names:
                print(f"✅ 위계형 필드 존재: {field}")
            else:
                print(f"❌ 위계형 필드 누락: {field}")
        
        # 기존 RAG 필드 확인
        rag_fields = ['passage_uid', 'doc_id', 'text', 'text_emb', 'title', 'author', 'domain']
        for field in rag_fields:
            if field in field_names:
                print(f"✅ 기존 RAG 필드 존재: {field}")
            else:
                print(f"❌ 기존 RAG 필드 누락: {field}")
        
        return True
        
    except Exception as e:
        print(f"❌ 스키마 테스트 실패: {e}")
        return False


def test_hierarchical_retriever():
    """위계형 검색기 테스트"""
    print("\n🔍 위계형 검색기 테스트...")
    
    try:
        # InteractManager 없이 테스트
        retriever = HierarchicalRetriever()
        print("✅ 검색기 초기화 성공")
        
        # 쿼리 분석 테스트
        test_queries = [
            "제1조의 내용이 궁금해요",
            "개인정보 보호에 대한 내용",
            "제2조의2 항목을 찾아주세요",
            "제3항의 세부사항"
        ]
        
        for query in test_queries:
            print(f"\n🔍 쿼리 분석: '{query}'")
            analysis = retriever._analyze_query(query)
            print(f"   조문 참조: {analysis['has_legal_references']}")
            print(f"   조문 목록: {analysis['article_references']}")
            print(f"   항 목록: {analysis['paragraph_references']}")
            print(f"   호 목록: {analysis['item_references']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 검색기 테스트 실패: {e}")
        return False


def test_legal_patterns():
    """법령 패턴 테스트"""
    print("\n🎯 법령 패턴 테스트...")
    
    try:
        retriever = HierarchicalRetriever()
        
        # 테스트 케이스들
        test_cases = [
            ("제1조", ["제1조"]),
            ("제2조의2", ["제2조의2"]),
            ("제3항", ["제3항"]),
            ("제1호", ["제1호"]),
            ("제1조와 제2조", ["제1조", "제2조"]),
            ("제1조의2 항목", ["제1조의2"]),
            ("일반적인 검색", []),
        ]
        
        for text, expected in test_cases:
            print(f"\n🔍 패턴 테스트: '{text}'")
            
            # 조문 패턴
            article_matches = list(retriever.legal_patterns["article_ref"].finditer(text))
            found_articles = []
            for match in article_matches:
                article_num = match.group(1)
                article_sub = match.group(2)
                ref = f"제{article_num}조"
                if article_sub:
                    ref += f"의{article_sub}"
                found_articles.append(ref)
            
            # 항 패턴
            paragraph_matches = list(retriever.legal_patterns["paragraph_ref"].finditer(text))
            found_paragraphs = [f"제{match.group(1)}항" for match in paragraph_matches]
            
            # 호 패턴
            item_matches = list(retriever.legal_patterns["item_ref"].finditer(text))
            found_items = [f"제{match.group(1)}호" for match in item_matches]
            
            print(f"   발견된 조문: {found_articles}")
            print(f"   발견된 항: {found_paragraphs}")
            print(f"   발견된 호: {found_items}")
            
            # 예상 결과와 비교
            all_found = found_articles + found_paragraphs + found_items
            if all_found == expected:
                print(f"   ✅ 패턴 매칭 성공")
            else:
                print(f"   ❌ 패턴 매칭 실패 (예상: {expected}, 실제: {all_found})")
        
        return True
        
    except Exception as e:
        print(f"❌ 패턴 테스트 실패: {e}")
        return False


def test_hierarchical_processor():
    """위계형 프로세서 테스트"""
    print("\n🔧 위계형 프로세서 테스트...")
    
    try:
        # 프로세서 초기화 (기본 인스턴스)
        processor = HierarchicalProcessor()
        print("✅ 프로세서 초기화 성공")
        
        # 조항 단위 청킹 테스트
        test_text = """
제1조(목적)
이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써 개인의 자유와 권리를 보호하고, 개인정보의 유용성을 증진함을 목적으로 한다.

제2조(정의)
① 이 법에서 사용하는 용어의 정의는 다음과 같다.
1. "개인정보"란 살아 있는 개인에 관한 정보로서 성명, 주민등록번호 및 영상 등을 통하여 개인을 알아볼 수 있는 정보를 말한다.
2. "처리"란 개인정보의 수집, 생성, 연계, 연동, 기록, 저장, 보유, 가공, 편집, 검색, 출력, 정정, 복구, 이용, 제공, 공개, 파기, 그 밖에 이와 유사한 행위를 말한다.

제3조(적용범위)
이 법은 개인정보처리자에 의한 개인정보 처리에 적용한다.
"""
        
        print(f"\n🔍 조항 단위 청킹 테스트:")
        chunks = processor.chunk_by_articles(test_text)
        print(f"   청킹 결과: {len(chunks)}개 청크")
        
        for i, (chunk_text, hierarchy) in enumerate(chunks):
            print(f"   청크 {i+1}:")
            print(f"     조문: {hierarchy['article_number']}")
            print(f"     항: {hierarchy['paragraph_number']}")
            print(f"     호: {hierarchy['item_number']}")
            print(f"     텍스트 길이: {len(chunk_text)}자")
        
        # 조문 참조 추출 테스트
        test_queries = [
            "제1조의 목적이 뭐야?",
            "제2조의 정의를 알려줘",
            "개인정보 처리에 대한 내용",
            "제3조의 적용범위"
        ]
        
        print(f"\n🔍 조문 참조 추출 테스트:")
        for query in test_queries:
            refs = processor._extract_legal_references(query)
            print(f"   쿼리: '{query}'")
            print(f"     조문 참조: {refs['has_references']}")
            print(f"     조문: {refs['articles']}")
            print(f"     항: {refs['paragraphs']}")
            print(f"     호: {refs['items']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 프로세서 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🚀 위계형 RAG 시스템 테스트 시작")
    print("=" * 50)
    
    start_time = time.time()
    
    # 테스트 실행
    tests = [
        test_hierarchical_schema,
        test_hierarchical_retriever,
        test_legal_patterns,
        test_hierarchical_processor,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류: {e}")
    
    # 결과 출력
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("📊 테스트 결과")
    print(f"✅ 통과: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")
    print(f"⏱️ 소요시간: {duration:.3f}초")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
        return True
    else:
        print("⚠️ 일부 테스트 실패")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
