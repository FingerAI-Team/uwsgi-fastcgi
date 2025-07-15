# Domain Selector (도메인 셀렉터)

질의에 대한 도메인을 선택하는 룰 기반 모듈입니다. 형태소 분석을 통해 도메인 키워드를 찾고, 시그니처 패턴을 분석하여 직접 참조인지 일반 언급인지 판단합니다.

## 주요 기능

- **도메인 키워드 검출**: 질의에서 도메인 관련 키워드를 자동으로 찾습니다
- **시그니처 패턴 분석**: "~상에서", "~에 의하면" 등의 패턴을 분석하여 직접 참조 여부를 판단합니다
- **다중 도메인 지원**: 하나의 질의에서 여러 도메인을 동시에 처리할 수 있습니다
- **룰 기반 처리**: 신뢰도가 아닌 명확한 룰에 따라 0 또는 1의 신호를 출력합니다

## 파일 구조

```
domain_selector/
├── __init__.py              # 패키지 초기화
├── domain_selector.py       # 핵심 도메인 셀렉터 클래스
├── domain_service.py        # RAG 시스템 통합 서비스
├── domain_config.json       # 도메인 설정 파일
├── test_domain_selector.py  # 테스트 스크립트
└── README.md               # 이 파일
```

## 사용법

### 기본 사용법

```python
from domain_selector import DomainSelector

# 도메인 셀렉터 초기화
selector = DomainSelector()

# 질의에 대한 도메인 선택
query = "법률상에서 계약 해지 조건에 대해 알려줘"
results = selector.select_domains(query)

for result in results:
    print(f"도메인: {result['domain']}")
    print(f"직접 참조: {result['is_direct_reference']}")
    print(f"발견된 키워드: {result['keywords_found']}")
```

### RAG 시스템 통합

```python
from domain_selector.domain_service import DomainService

# 도메인 서비스 초기화
domain_service = DomainService()

# 질의 처리
query = "의료에 의하면 이 증상은 감기입니다"
result = domain_service.process_query(query)

print(f"도메인 후보: {result['domain_candidates']}")
print(f"직접 참조 여부: {result['has_direct_reference']}")
print(f"도메인 필터링 필요: {result['should_filter_by_domain']}")
```

## 설정 파일 (domain_config.json)

### 도메인 정의

```json
{
  "domains": {
    "법률": {
      "keywords": ["법률", "법", "조항", "규정", "법령"],
      "priority": 1
    },
    "의료": {
      "keywords": ["의료", "병원", "진료", "치료", "약"],
      "priority": 1
    }
  }
}
```

### 시그니처 패턴

```json
{
  "signature_patterns": [
    "~상에서",
    "~에 의하면",
    "~에 따르면",
    "~에 대해 검색해줘",
    "~에 관한 정보를 찾아줘"
  ]
}
```

## API 참조

### DomainSelector 클래스

#### `__init__(config_path=None)`
도메인 셀렉터를 초기화합니다.

#### `select_domains(query: str) -> List[Dict]`
질의에 대해 관련된 도메인들을 선택합니다.

**반환값:**
- `domain`: 도메인명
- `is_direct_reference`: 직접 참조 여부 (True/False)
- `confidence`: 신뢰도 (0 또는 1)
- `keywords_found`: 발견된 키워드들
- `priority`: 우선순위

#### `get_domain_candidates(query: str) -> List[str]`
질의에 대해 도메인 후보들을 반환합니다.

#### `is_direct_reference(query: str, domain: str) -> bool`
특정 도메인에 대한 직접 참조 여부를 확인합니다.

### DomainService 클래스

#### `process_query(query: str) -> Dict`
질의를 처리하여 도메인 정보를 반환합니다.

**반환값:**
- `domains`: 발견된 도메인 정보 리스트
- `has_direct_reference`: 직접 참조가 있는지 여부
- `domain_candidates`: 도메인 후보 리스트
- `should_filter_by_domain`: 도메인별 필터링이 필요한지 여부
- `original_query`: 원본 질의

#### `get_search_domains(query: str) -> List[str]`
검색에 사용할 도메인 리스트를 반환합니다.

#### `is_domain_specific_query(query: str) -> bool`
질의가 특정 도메인에 대한 것인지 확인합니다.

#### `get_domain_filter_query(query: str, domain: str) -> str`
특정 도메인에 대한 필터링된 검색 쿼리를 생성합니다.

#### `get_domain_context(query: str) -> Dict`
도메인 컨텍스트 정보를 반환합니다.


### 인터랙티브 테스트 (사용자 직접 입력)
```bash
cd prompt/domain_selector
python interactive_test.py
```

인터랙티브 테스트에서는 사용자가 직접 질의를 입력하여 도메인 셀렉터의 동작을 확인할 수 있습니다.
종료하려면 'quit', 'exit', '종료' 중 하나를 입력하세요.

## 확장 방법

1. **새로운 도메인 추가**: `domain_config.json`의 `domains` 섹션에 새로운 도메인을 추가
2. **새로운 시그니처 패턴 추가**: `signature_patterns` 섹션에 새로운 패턴 추가
3. **우선순위 조정**: 각 도메인의 `priority` 값을 조정하여 우선순위 변경

## 주의사항

- 도메인 키워드는 2글자 이상이어야 합니다
- 시그니처 패턴에서 `~`는 도메인 키워드로 대체됩니다
- 룰 기반이므로 신뢰도는 항상 1입니다
- 직접 참조가 있는 경우 해당 도메인에 대해서만 RAG 검색을 수행해야 합니다 