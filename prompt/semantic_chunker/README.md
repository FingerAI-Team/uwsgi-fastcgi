# Semantic Chunker

현재 질의와 관련된 대화 히스토리만 선별하는 시멘틱 청커 모듈입니다.

## 기능

- **관련 히스토리 선별**: 현재 질의와 관련된 대화 턴만 선택
- **맥락 유지**: 대화의 논리적 흐름을 유지하면서 불필요한 정보 제거
- **성능 최적화**: Mistral-7B 모델을 사용한 빠른 처리
- **템플릿 기반**: 프롬프트를 별도 템플릿 파일로 관리

## 사용법

```python
from semantic_chunker import SemanticChunker

# 초기화
chunker = SemanticChunker()

# 관련 히스토리 선별
relevant_history = chunker.select_relevant_history(
    current_query="그것은 어떻게 작동하나요?",
    session_data={"history": [...]}
)
```

## 설정

- **VLLM 엔드포인트**: `http://vllm:8000`
- **모델**: `Mistral-7B-Instruct-v0.2`
- **최대 히스토리 턴**: 10턴
- **기본값**: 최근 3턴 (선별 실패 시)

## 성능

- **처리 시간**: ~0.5-0.6초 (Query Rewriter와 유사)
- **정확도**: 현재 질의와 관련된 대화 턴 선별
- **안정성**: 파싱 실패 시 기본값 반환 