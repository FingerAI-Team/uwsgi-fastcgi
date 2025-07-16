# Query Rewrite 시스템

대화 컨텍스트를 기반으로 사용자 질문을 개선하는 Query Rewrite 시스템입니다.

## 개요

Query Rewrite 시스템은 사용자의 질문을 대화 맥락을 고려하여 더 명확하고 구체적으로 재작성합니다. 이를 통해 RAG 검색의 정확도를 향상시키고 더 관련성 높은 응답을 생성할 수 있습니다.

## 주요 기능

- **대화 컨텍스트 분석**: 최근 5개 대화 턴을 기반으로 질문의 맥락을 파악
- **질문 개선**: 대명사 해결, 생략된 정보 보완, 구체적인 표현으로 변환
- **신뢰도 평가**: 재작성된 질문의 품질을 0.0-1.0 범위로 평가
- **오류 처리**: 재작성 실패 시 원본 질문을 그대로 사용

## 시스템 구조

```
query_rewriter/
├── __init__.py
├── query_rewriter.py      # 메인 QueryRewriter 클래스
├── templates/
│   └── query_rewrite.txt  # 프롬프트 템플릿
├── test_query_rewriter.py # 테스트 스크립트
└── README.md
```

## 사용법

### 기본 사용법

```python
from query_rewriter.query_rewriter import QueryRewriter

# Query Rewriter 초기화
rewriter = QueryRewriter(
    ollama_endpoint="http://ollama-gpu:11434",
    default_model="gemma3:12b",
    temperature=0.3,
    max_history_turns=5
)

# 질문 재작성
result = rewriter.rewrite_query(
    current_query="그것은 어떻게 작동하나요?",
    session_data=session_data,
    model="gemma3:12b"
)

print(f"원본: {result['original_query']}")
print(f"재작성: {result['rewritten_query']}")
print(f"신뢰도: {result['confidence']}")
```

### RAG 챗봇 서비스와 통합

Query Rewrite는 `RagChatService`에 자동으로 통합되어 있습니다:

1. 사용자 질문 입력
2. 세션 데이터 로드
3. **Query Rewrite 수행** (새로 추가된 단계)
4. 도메인 셀렉터 및 RAG 검색
5. 시스템 프롬프트 로드
6. 검색 결과 포맷팅
7. 프롬프트 구성 및 LLM 호출

## 응답 형식

Query Rewrite 결과는 다음과 같은 정보를 포함합니다:

```json
{
  "original_query": "그것은 어떻게 작동하나요?",
  "rewritten_query": "인공지능은 어떻게 작동하나요?",
  "confidence": 0.85,
  "reasoning": "최근 3개 대화 턴을 기반으로 질문을 개선했습니다",
  "used_history": 3
}
```

## 설정 옵션

- `ollama_endpoint`: Ollama API 엔드포인트
- `default_model`: 기본 LLM 모델
- `temperature`: 생성 온도 (낮을수록 일관성 높음)
- `max_history_turns`: 참조할 최대 대화 턴 수

## 테스트

테스트 스크립트를 실행하여 Query Rewrite 시스템을 테스트할 수 있습니다:

```bash
cd prompt/query_rewriter
python test_query_rewriter.py
```

## 예시

### 입력
- 대화 기록: "인공지능에 대해 알려줘" → "인공지능(AI)은..."
- 현재 질문: "그것은 어떻게 작동하나요?"

### 출력
- 재작성된 질문: "인공지능은 어떻게 작동하나요?"
- 신뢰도: 0.85
- 이유: 대화 맥락에서 "그것"이 "인공지능"을 지칭함을 파악

## 성능 고려사항

- Query Rewrite는 추가적인 LLM 호출이 필요하므로 응답 시간이 약간 증가할 수 있습니다
- 대화 기록이 없거나 질문이 이미 명확한 경우 원본을 그대로 사용하여 불필요한 처리를 방지합니다
- 신뢰도가 낮은 경우 원본 질문을 우선적으로 사용합니다

## 로깅

Query Rewrite 과정은 상세한 로그를 제공합니다:

```
[Query Rewrite] 시작: 원본 질문='그것은 어떻게 작동하나요?'
[Query Rewrite] 결과: '그것은 어떻게 작동하나요?' → '인공지능은 어떻게 작동하나요?' (신뢰도: 0.85)
[Query Rewrite] 완료: '그것은 어떻게 작동하나요?' → '인공지능은 어떻게 작동하나요?' (신뢰도: 0.85)
``` 