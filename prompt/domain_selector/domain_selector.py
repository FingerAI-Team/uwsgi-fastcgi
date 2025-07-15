import json
import re
from typing import List, Dict, Tuple, Set
from pathlib import Path


class DomainSelector:
    """
    질의에 대한 도메인을 선택하는 룰 기반 모듈
    형태소 분석을 통해 도메인 키워드를 찾고, 시그니처 패턴을 분석하여
    직접 참조인지 일반 언급인지 판단합니다.
    """
    
    def __init__(self, config_path: str = None):
        """
        DomainSelector 초기화
        
        Args:
            config_path: 도메인 설정 파일 경로
        """
        if config_path is None:
            config_path = Path(__file__).parent / "domain_config.json"
        
        self.config = self._load_config(config_path)
        self.domains = self.config.get("domains", {})
        self.signature_patterns = self.config.get("signature_patterns", {})
        
        # 도메인 키워드 매핑 생성
        self.domain_keyword_map = self._build_domain_keyword_map()
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일을 로드합니다."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"도메인 설정 파일을 찾을 수 없습니다: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"도메인 설정 파일이 올바른 JSON 형식이 아닙니다: {config_path}")
    
    def _build_domain_keyword_map(self) -> Dict[str, str]:
        """도메인 키워드와 도메인명의 매핑을 생성합니다."""
        keyword_map = {}
        for domain_name, domain_info in self.domains.items():
            keywords = domain_info.get("keywords", [])
            for keyword in keywords:
                keyword_map[keyword] = domain_name
        return keyword_map
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        질의에서 키워드를 추출합니다.
        간단한 형태소 분석을 수행합니다.
        """
        # 공백을 제거한 텍스트에서 키워드 추출
        query_no_space = query.replace(" ", "")
        
        # 한글, 영문, 숫자로 구성된 단어들을 추출 (공백 제거 후)
        words_no_space = re.findall(r'[가-힣a-zA-Z0-9]+', query_no_space)
        
        # 원본 텍스트에서도 키워드 추출 (공백 포함)
        words_with_space = re.findall(r'[가-힣a-zA-Z0-9]+', query)
        
        # 2글자 이상의 단어만 필터링
        keywords_no_space = [word for word in words_no_space if len(word) >= 2]
        keywords_with_space = [word for word in words_with_space if len(word) >= 2]
        
        # 두 결과를 합치고 중복 제거
        all_keywords = list(set(keywords_no_space + keywords_with_space))
        
        return all_keywords
    
    def _find_domain_keywords(self, keywords: List[str]) -> Set[str]:
        """
        키워드 목록에서 도메인 키워드를 찾습니다.
        
        Args:
            keywords: 추출된 키워드 목록
            
        Returns:
            발견된 도메인명들의 집합
        """
        found_domains = set()
        
        for keyword in keywords:
            if keyword in self.domain_keyword_map:
                found_domains.add(self.domain_keyword_map[keyword])
        
        return found_domains
    
    def _find_domain_keywords_in_query(self, query: str) -> Set[str]:
        """
        질의에서 직접 도메인 키워드를 찾습니다.
        공백을 무시하고 매칭합니다.
        
        Args:
            query: 원본 질의
            
        Returns:
            발견된 도메인명들의 집합
        """
        found_domains = set()
        query_lower = query.lower()
        
        for domain_name, domain_info in self.domains.items():
            keywords = domain_info.get("keywords", [])
            for keyword in keywords:
                # 공백을 제거한 질의에서 키워드 검색
                if keyword in query_lower.replace(" ", ""):
                    found_domains.add(domain_name)
                    break
        
        return found_domains
    
    def _analyze_signature_patterns(self, query: str, domain_keywords: Set[str]) -> Dict[str, bool]:
        """
        시그니처 패턴을 분석하여 직접 참조 여부를 판단합니다.
        
        Args:
            query: 원본 질의
            domain_keywords: 발견된 도메인 키워드들
            
        Returns:
            도메인별 직접 참조 여부 (True: 직접 참조, False: 일반 언급)
        """
        direct_reference = {}
        
        for domain in domain_keywords:
            is_direct = False
            
            # 모든 시그니처 패턴 확인
            for pattern in self.signature_patterns:
                # 패턴에서 ~ 부분을 도메인 키워드로 대체하여 검사
                for keyword in self.domains[domain]["keywords"]:
                    test_pattern = pattern.replace("~", keyword)
                    if test_pattern in query:
                        is_direct = True
                        break
                if is_direct:
                    break
            
            direct_reference[domain] = is_direct
        
        return direct_reference
    
    def select_domains(self, query: str) -> List[Dict[str, any]]:
        """
        질의에 대해 관련된 도메인들을 선택합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            도메인 정보 리스트. 각 도메인은 다음 정보를 포함:
            - domain: 도메인명
            - is_direct_reference: 직접 참조 여부 (True/False)
            - confidence: 신뢰도 (0 또는 1)
            - keywords_found: 발견된 키워드들
        """
        # 1. 질의에서 정확히 매칭되는 키워드 찾기 (전체 문장에서 검사)
        found_keywords_in_query = []
        
        # 모든 도메인의 모든 키워드를 수집하고 길이순으로 정렬
        all_keywords = []
        for domain_name, domain_info in self.domains.items():
            keywords = domain_info.get("keywords", [])
            for keyword in keywords:
                all_keywords.append((keyword, domain_name))
        
        # 키워드를 길이순으로 정렬 (긴 키워드 우선)
        all_keywords.sort(key=lambda x: len(x[0]), reverse=True)
        
        # 질의에서 매칭되는 키워드 찾기 (긴 키워드부터 검사)
        # 같은 키워드라도 다른 도메인에 속한다면 모두 찾기 위해 위치 겹침 검사 제거
        
        for keyword, domain_name in all_keywords:
            if keyword in query:
                found_keywords_in_query.append((keyword, domain_name))
        
        # 2. 발견된 키워드에 대해서만 시그니처 패턴 검사
        direct_reference_domains = set()
        
        for keyword, domain_name in found_keywords_in_query:
            for pattern in self.signature_patterns:
                # 붙어있는 패턴과 띄어있는 패턴 모두 검사
                test_pattern_attached = pattern.replace("~", keyword)  # "국회법상에서"
                test_pattern_spaced = pattern.replace("~", keyword + " ")  # "국회법 상에서"
                
                # 둘 중 하나라도 매칭되면 해당 도메인 선택
                if test_pattern_attached in query or test_pattern_spaced in query:
                    direct_reference_domains.add(domain_name)
                    break
        
        # 3. 결과 구성
        results = []
        for domain in direct_reference_domains:
            # 발견된 키워드들 찾기 (패턴 매칭에 사용된 키워드만)
            found_keywords = []
            for keyword, domain_name in found_keywords_in_query:
                if domain_name == domain:
                    for pattern in self.signature_patterns:
                        # 붙어있는 패턴과 띄어있는 패턴 모두 검사
                        test_pattern_attached = pattern.replace("~", keyword)
                        test_pattern_spaced = pattern.replace("~", keyword + " ")
                        
                        if test_pattern_attached in query or test_pattern_spaced in query:
                            found_keywords.append(keyword)
                            break
            
            result = {
                "domain": domain,
                "is_direct_reference": True,  # 시그니처 패턴이 있으므로 직접 참조
                "confidence": 1,  # 룰 기반이므로 항상 1
                "keywords_found": found_keywords,
                "priority": self.domains[domain].get("priority", 1)
            }
            results.append(result)
        
        # 우선순위에 따라 정렬
        results.sort(key=lambda x: x["priority"], reverse=True)
        
        return results
    
    def get_domain_candidates(self, query: str) -> List[str]:
        """
        질의에 대해 도메인 후보들을 반환합니다.
        시그니처 패턴이 있는 도메인만 후보로 반환합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            도메인 후보 리스트
        """
        domain_results = self.select_domains(query)
        return [result["domain"] for result in domain_results]
    
    def is_direct_reference(self, query: str, domain: str) -> bool:
        """
        특정 도메인에 대한 직접 참조 여부를 확인합니다.
        시그니처 패턴이 있는 경우만 True를 반환합니다.
        
        Args:
            query: 사용자 질의
            domain: 확인할 도메인명
            
        Returns:
            직접 참조 여부
        """
        domain_results = self.select_domains(query)
        for result in domain_results:
            if result["domain"] == domain:
                return result["is_direct_reference"]
        return False 