#!/usr/bin/env python3
"""
상세 로그 테스트 스크립트
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

def test_detailed_logging():
    """상세 로그 테스트"""
    print("🔍 상세 로그 테스트")
    print("=" * 50)
    
    # 설정 로더 테스트
    print("📋 설정 로더 테스트")
    config_loader = get_config_loader()
    
    # 의도 감지 테스트
    print("\n🎯 의도 감지 테스트")
    test_queries = [
        "개인정보 처리법의 정의는 뭐야?",
        "개인정보 처리법의 목적은 뭐야?",
        "개인정보 처리법의 벌칙은 어떻게 되나요?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 쿼리: {query}")
        intent_analysis = config_loader.detect_semantic_intent(query)
        
        if intent_analysis["has_semantic_intent"]:
            primary_intent = intent_analysis["primary_intent"]
            print(f"  ✅ 의도 감지: {primary_intent['intent']}")
            print(f"  📊 신뢰도: {primary_intent['confidence']:.3f}")
            print(f"  🎯 매칭 키워드: {primary_intent['matched_keywords']}")
        else:
            print("  ❌ 의도 감지 안됨")
    
    # 검색 테스트
    print("\n🔍 검색 테스트")
    retriever = LegalRetriever()
    
    test_query = "개인정보 처리법의 정의는 뭐야?"
    print(f"🔍 검색 쿼리: {test_query}")
    
    try:
        results = retriever.search_legal_documents("legal_documents", test_query, {"top_k": 5})
        print(f"📊 검색 결과: {len(results)}개")
        
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}위: {result.get('title', 'N/A')[:50]}...")
            print(f"      점수: {result.get('score', 0.0):.4f}")
            if "intent_info" in result:
                print(f"      의도: {result['intent_info']['detected_intent']}")
                print(f"      신뢰도: {result['intent_info']['confidence']:.3f}")
        
    except Exception as e:
        print(f"❌ 검색 오류: {e}")
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    test_detailed_logging()
