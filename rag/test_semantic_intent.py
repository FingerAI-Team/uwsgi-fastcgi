#!/usr/bin/env python3
"""
의미적 의도 기반 검색 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hierarchical.config.config_loader import get_config_loader
from hierarchical.legal.retriever import LegalRetriever
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_intent_detection():
    """의미적 의도 감지 테스트"""
    print("🎯 의미적 의도 감지 테스트")
    print("=" * 50)
    
    config_loader = get_config_loader()
    
    test_queries = [
        "개인정보 처리법의 목적은 뭐야?",
        "개인정보 처리법의 정의는 뭐야?",
        "개인정보 처리법의 벌칙은 어떻게 되나요?",
        "개인정보 처리법의 절차는 어떻게 되나요?",
        "개인정보 처리법의 예외는 뭐가 있나요?",
        "개인정보 처리법의 적용 범위는 어디까지인가요?",
        "개인정보 처리법에서 권리는 어떻게 보호되나요?",
        "제1조는 뭐라고 되어있나요?",
        "개인정보 처리법 제2조의2는 뭐인가요?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 쿼리: {query}")
        intent_analysis = config_loader.detect_semantic_intent(query)
        
        if intent_analysis["has_semantic_intent"]:
            primary_intent = intent_analysis["primary_intent"]
            print(f"  ✅ 의도 감지: {primary_intent['intent']}")
            print(f"  📊 신뢰도: {primary_intent['confidence']:.2f}")
            print(f"  🎯 매칭 키워드: {primary_intent['matched_keywords']}")
            
            # 모든 감지된 의도들 출력
            if len(intent_analysis["detected_intents"]) > 1:
                print("  📋 모든 감지된 의도들:")
                for i, intent in enumerate(intent_analysis["detected_intents"]):
                    print(f"    {i+1}. {intent['intent']} (신뢰도: {intent['confidence']:.2f}, 키워드: {intent['matched_keywords']})")
        else:
            print("  ❌ 의도 감지 안됨")
            
            # 디버깅을 위해 각 의도별 점수 계산
            print("  🔍 디버깅 정보:")
            intents = config_loader.get_semantic_intents()
            query_lower = query.lower()
            for intent_name, intent_config in intents.items():
                confidence = 0.0
                matched_keywords = []
                keywords = intent_config.get("keywords", [])
                
                for keyword in keywords:
                    if keyword in query_lower:
                        if keyword in ["목적", "정의", "벌칙", "절차", "예외", "적용", "범위", "권리"]:
                            confidence += 0.6
                        else:
                            confidence += 0.3
                        matched_keywords.append(keyword)
                
                if confidence > 0:
                    print(f"    - {intent_name}: {confidence:.2f} (임계값: {intent_config.get('confidence_threshold', 0.3):.2f}) - 키워드: {matched_keywords}")

def test_query_analysis():
    """쿼리 분석 테스트"""
    print("\n\n🔍 쿼리 분석 테스트")
    print("=" * 50)
    
    retriever = LegalRetriever()
    
    test_queries = [
        "개인정보 처리법의 목적은 뭐야?",
        "개인정보 처리법의 정의는 뭐야?",
        "제1조는 뭐라고 되어있나요?",
        "개인정보 처리법 제2조의2는 뭐인가요?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 쿼리: {query}")
        analysis = retriever._analyze_legal_query(query)
        
        print(f"  📝 원본 쿼리: {analysis['original_query']}")
        print(f"  🔧 처리된 쿼리: {analysis['processed_query']}")
        print(f"  📋 조문 참조: {analysis['has_legal_references']}")
        print(f"  🎯 의미적 의도: {analysis['has_semantic_intent']}")
        
        if analysis['has_legal_references']:
            print(f"  📖 조문 참조: {analysis['article_references']}")
            print(f"  📄 항 참조: {analysis['paragraph_references']}")
            print(f"  📋 호 참조: {analysis['item_references']}")
            print(f"  ⚖️ 법령 참조: {analysis['law_references']}")
        
        if analysis['has_semantic_intent']:
            print(f"  🎯 감지된 의도: {analysis['semantic_intent']}")
            print(f"  📊 의도 신뢰도: {analysis['semantic_intent_confidence']:.2f}")

def test_config_loading():
    """설정 로딩 테스트"""
    print("\n\n⚙️ 설정 로딩 테스트")
    print("=" * 50)
    
    config_loader = get_config_loader()
    
    # 의미적 의도 설정
    semantic_intents = config_loader.get_semantic_intents()
    print(f"📚 의미적 의도 설정: {len(semantic_intents)}개")
    for intent_name, intent_config in semantic_intents.items():
        print(f"  - {intent_name}: {intent_config.get('description', '설명 없음')}")
    
    # 의도 감지 설정
    detection_config = config_loader.get_intent_detection_config()
    print(f"\n🎯 의도 감지 설정:")
    for key, value in detection_config.items():
        print(f"  - {key}: {value}")
    
    # 검색 전략 설정
    search_strategies = config_loader.get_search_strategies_config()
    print(f"\n🔍 검색 전략 설정: {len(search_strategies)}개")
    for strategy_name, strategy_config in search_strategies.items():
        print(f"  - {strategy_name}: vector_weight={strategy_config.get('vector_weight', 'N/A')}")
    
    # 결과 병합 설정
    merging_config = config_loader.get_result_merging_config()
    print(f"\n🔗 결과 병합 설정:")
    for key, value in merging_config.items():
        print(f"  - {key}: {value}")

if __name__ == "__main__":
    print("🚀 의미적 의도 기반 검색 시스템 테스트")
    print("=" * 60)
    
    try:
        # 1. 설정 로딩 테스트
        test_config_loading()
        
        # 2. 의도 감지 테스트
        test_intent_detection()
        
        # 3. 쿼리 분석 테스트
        test_query_analysis()
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
