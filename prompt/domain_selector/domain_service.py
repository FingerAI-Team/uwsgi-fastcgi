from typing import List, Dict, Any, Optional
from domain_selector import DomainSelector


class DomainService:
    """
    도메인 셀렉터를 RAG 시스템과 통합하기 위한 서비스 클래스
    """
    
    def __init__(self, config_path: str = None):
        """
        DomainService 초기화
        
        Args:
            config_path: 도메인 설정 파일 경로
        """
        self.selector = DomainSelector(config_path)
        # 기본 도메인 목록 로드
        self.default_domains = self.selector.config.get("default_domains", [])
    
    def get_default_domains(self) -> List[str]:
        """
        기본 도메인 목록을 반환합니다.
        
        Returns:
            기본 도메인 목록
        """
        return self.default_domains.copy()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        질의를 처리하여 도메인 정보를 반환합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            도메인 처리 결과:
            - domains: 발견된 도메인 정보 리스트
            - has_direct_reference: 직접 참조가 있는지 여부
            - domain_candidates: 도메인 후보 리스트
            - should_filter_by_domain: 도메인별 필터링이 필요한지 여부
            - used_default_domains: 기본 도메인을 사용했는지 여부
        """
        # 도메인 선택 (시그니처 패턴이 있는 경우만)
        domain_results = self.selector.select_domains(query)
        
        # 직접 참조가 있는지 확인
        has_direct_reference = any(result["is_direct_reference"] for result in domain_results)
        
        # 도메인 후보 리스트
        domain_candidates = [result["domain"] for result in domain_results]
        
        # 도메인 후보가 없으면 기본 도메인 사용
        if not domain_candidates and self.default_domains:
            domain_candidates = self.default_domains
        
        # 도메인별 필터링이 필요한지 판단
        # 시그니처 패턴이 있는 경우만 필터링 필요
        should_filter_by_domain = has_direct_reference
        
        return {
            "domains": domain_results,
            "has_direct_reference": has_direct_reference,
            "domain_candidates": domain_candidates,
            "should_filter_by_domain": should_filter_by_domain,
            "original_query": query
        }
    
    def get_search_domains(self, query: str) -> List[str]:
        """
        검색에 사용할 도메인 리스트를 반환합니다.
        도메인 후보가 없으면 기본 도메인 목록을 반환합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            검색할 도메인 리스트
        """
        result = self.process_query(query)
        return result["domain_candidates"]
    
    def is_domain_specific_query(self, query: str) -> bool:
        """
        질의가 특정 도메인에 대한 것인지 확인합니다.
        시그니처 패턴이 있는 경우만 True를 반환합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            도메인 특화 질의 여부
        """
        result = self.process_query(query)
        return result["should_filter_by_domain"]
    
    def get_domain_filter_query(self, query: str, domain: str) -> str:
        """
        특정 도메인에 대한 필터링된 검색 쿼리를 생성합니다.
        
        Args:
            query: 원본 질의
            domain: 필터링할 도메인
            
        Returns:
            도메인 필터가 적용된 검색 쿼리
        """
        # 간단한 도메인 필터링을 위해 도메인 키워드를 쿼리에 추가
        domain_keywords = self.selector.domains.get(domain, {}).get("keywords", [])
        if domain_keywords:
            # 가장 대표적인 키워드를 사용
            primary_keyword = domain_keywords[0]
            return f"{query} {primary_keyword}"
        
        return query
    
    def get_domain_context(self, query: str) -> Dict[str, Any]:
        """
        도메인 컨텍스트 정보를 반환합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            도메인 컨텍스트 정보
        """
        result = self.process_query(query)
        
        context = {
            "query_type": "general",
            "target_domains": [],
            "search_strategy": "general"
        }
        
        if result["has_direct_reference"]:
            context["query_type"] = "domain_specific"
            context["target_domains"] = [
                domain["domain"] for domain in result["domains"] 
                if domain["is_direct_reference"]
            ]
            context["search_strategy"] = "domain_filtered"
        elif result["domain_candidates"]:
            context["query_type"] = "domain_related"
            context["search_strategy"] = "domain_enhanced"
            context["target_domains"] = result["domain_candidates"]
        
        return context 