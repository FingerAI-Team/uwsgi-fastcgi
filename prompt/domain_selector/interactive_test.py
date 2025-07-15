#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
도메인 셀렉터 인터랙티브 테스트 스크립트
사용자가 직접 질의를 입력하여 도메인 셀렉터를 테스트할 수 있습니다.
"""

from domain_selector import DomainSelector
from domain_service import DomainService


def print_domain_results(results):
    """도메인 결과를 예쁘게 출력합니다."""
    if not results:
        print("  ❌ 발견된 도메인 없음")
        return
    
    print("  ✅ 발견된 도메인:")
    for i, result in enumerate(results, 1):
        print(f"    {i}. {result['domain']}")
        print(f"       직접 참조: {'🔍' if result['is_direct_reference'] else '💬'}")
        print(f"       신뢰도: {result['confidence']}")
        print(f"       발견된 키워드: {', '.join(result['keywords_found'])}")
        print(f"       우선순위: {result['priority']}")
        print()


def print_service_results(result):
    """서비스 결과를 예쁘게 출력합니다."""
    print("  📊 서비스 분석 결과:")
    print(f"    직접 참조 여부: {'🔍' if result['has_direct_reference'] else '💬'}")
    print(f"    도메인 필터링 필요: {'✅' if result['should_filter_by_domain'] else '❌'}")
    print(f"    도메인 후보: {', '.join(result['domain_candidates']) if result['domain_candidates'] else '없음'}")
    print()


def print_domain_context(context):
    """도메인 컨텍스트를 예쁘게 출력합니다."""
    print("  🎯 도메인 컨텍스트:")
    print(f"    질의 타입: {context['query_type']}")
    print(f"    검색 전략: {context['search_strategy']}")
    print(f"    대상 도메인: {', '.join(context['target_domains']) if context['target_domains'] else '없음'}")
    print()


def interactive_test():
    """인터랙티브 테스트를 실행합니다."""
    print("=" * 60)
    print("🔍 도메인 셀렉터 인터랙티브 테스트")
    print("=" * 60)
    print()
    print("사용 가능한 도메인:")
    print("  - 법률 (법률, 법, 조항, 규정, 법령, 법전, 민법, 형법, 상법, 헌법)")
    print("  - 의료 (의료, 병원, 진료, 치료, 약, 증상, 질병, 의사, 환자, 진단)")
    print("  - 금융 (금융, 은행, 투자, 주식, 보험, 대출, 예금, 신용카드, 펀드, 증권)")
    print("  - 교육 (교육, 학교, 학습, 수업, 교과서, 시험, 학생, 교사, 대학, 강의)")
    print("  - 기술 (기술, 프로그래밍, 소프트웨어, 하드웨어, AI, 머신러닝, 데이터베이스, 네트워크, 클라우드, 보안)")
    print("  - 경제 (경제, 경영, 기업, 시장, 무역, 수출, 수입, GDP, 인플레이션, 환율)")
    print()
    print("시그니처 패턴 예시:")
    print("  - 직접 참조: '법률상에서', '의료에 의하면', '금융에 따르면'")
    print("  - 검색 요청: '법률에 대해 검색해줘', '의료에 관한 정보를 찾아줘'")
    print()
    print("종료하려면 'quit', 'exit', '종료' 중 하나를 입력하세요.")
    print("-" * 60)
    
    # 도메인 셀렉터와 서비스 초기화
    selector = DomainSelector()
    service = DomainService()
    
    while True:
        try:
            # 사용자 입력 받기
            query = input("\n💬 질의를 입력하세요: ").strip()
            
            # 종료 조건 확인
            if query.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n👋 테스트를 종료합니다.")
                break
            
            if not query:
                print("❌ 질의를 입력해주세요.")
                continue
            
            print(f"\n🔍 분석 중: '{query}'")
            print("-" * 40)
            
            # 1. 기본 도메인 셀렉터 테스트
            print("1️⃣ 기본 도메인 셀렉터 결과:")
            domain_results = selector.select_domains(query)
            print_domain_results(domain_results)
            
            # 2. 도메인 서비스 테스트
            print("2️⃣ 도메인 서비스 결과:")
            service_result = service.process_query(query)
            print_service_results(service_result)
            
            # 3. 도메인 컨텍스트 테스트
            print("3️⃣ 도메인 컨텍스트:")
            context = service.get_domain_context(query)
            print_domain_context(context)
            
            # 4. 추가 정보
            if service_result['domain_candidates']:
                print("4️⃣ 도메인별 필터링된 쿼리:")
                for domain in service_result['domain_candidates']:
                    filtered_query = service.get_domain_filter_query(query, domain)
                    print(f"   {domain}: '{filtered_query}'")
                print()
            
        except KeyboardInterrupt:
            print("\n\n👋 테스트를 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")


if __name__ == "__main__":
    interactive_test() 