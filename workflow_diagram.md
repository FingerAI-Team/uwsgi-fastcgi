# Prompt Chatbot → Enhanced Search 워크플로우 (Mermaid 플로우차트)

## 1. 전체 시스템 아키텍처

```mermaid
graph TB
    Client[클라이언트] --> Nginx[Nginx Proxy<br/>포트 12321]
    Nginx --> Prompt[Prompt Service<br/>FastCGI]
    Nginx --> RAG[RAG Service<br/>FastCGI]
    Nginx --> Reranker[Reranker Service<br/>FastCGI]
    
    Prompt --> LLM_Gateway[LLM Gateway<br/>Ollama/vLLM]
    RAG --> Milvus[Milvus Vector DB]
    RAG --> Embedding[BGE-M3<br/>임베딩 모델]
    Reranker --> FlashRank[FlashRank<br/>GTE-Multilingual]
    Reranker --> MRC[MRC<br/>Machine Reading Comprehension]
    
    LLM_Gateway --> Ollama[Ollama<br/>Gemma3:12b]
    LLM_Gateway --> VLLM[vLLM<br/>Mistral-7B]
    
    subgraph "GPU Resources"
        Ollama
        VLLM
        FlashRank
        MRC
    end
    
    subgraph "Vector Database"
        Milvus
        Embedding
    end
```

## 2. 상세 워크플로우 (Prompt Chatbot)

```mermaid
flowchart TD
    A[클라이언트 요청<br/>POST /prompt/chatbot] --> B{세션 ID 존재?}
    
    B -->|Yes| C[기존 세션 로드]
    B -->|No| D[새 세션 생성]
    
    C --> E[사용자 메시지 추가]
    D --> E
    
    E --> F[토큰 수 계산]
    F --> G{토큰 초과?}
    
    G -->|Yes| H[오래된 메시지 제거<br/>LRU 방식]
    G -->|No| I[Query Rewrite 수행]
    H --> I
    
    I --> J[vLLM 호출<br/>Mistral-7B]
    J --> K{신뢰도 > 0.7?}
    
    K -->|Yes| L[재작성된 쿼리 사용]
    K -->|No| M[원본 쿼리 사용]
    
    L --> N[도메인 셀렉터 실행]
    M --> N
    
    N --> O{사용자 지정 도메인?}
    O -->|Yes| P[사용자 도메인 사용]
    O -->|No| Q{도메인 셀렉터 결과?}
    
    Q -->|Yes| R[자동 선택 도메인]
    Q -->|No| S[전체 도메인 검색]
    
    P --> T[Enhanced Search API 호출]
    R --> T
    S --> T
    
    T --> U[RAG 검색 단계]
    U --> V[하이브리드 재랭킹]
    V --> W[결과 후처리]
    W --> X[컨텍스트 구성]
    X --> Y[시스템 프롬프트 선택]
    Y --> Z[최종 프롬프트 생성]
    Z --> AA[LLM 응답 생성]
    AA --> BB[응답 후처리]
    BB --> CC[세션 저장]
    CC --> DD[최종 응답 반환]
```

## 3. Enhanced Search 상세 플로우

```mermaid
flowchart TD
    A[Enhanced Search API<br/>POST /prompt/enhanced_search] --> B[파라미터 검증]
    
    B --> C[RAG 검색 요청]
    C --> D[BGE-M3 임베딩 생성]
    D --> E[Milvus 벡터 검색]
    E --> F{검색 결과 있음?}
    
    F -->|No| G[빈 결과 반환]
    F -->|Yes| H[검색 결과 전처리]
    
    H --> I[하이브리드 재랭킹 시작]
    I --> J[FlashRank 재랭킹<br/>GTE-Multilingual]
    I --> K[MRC 처리<br/>질문-답변 생성]
    I --> L[원본 점수 유지]
    
    J --> M[하이브리드 점수 계산<br/>FlashRank×0.5 + MRC×0.3 + 원본×0.2]
    K --> M
    L --> M
    
    M --> N[임계치 필터링<br/>score < threshold 제외]
    N --> O[상위 N개 결과 선택]
    O --> P[메타데이터 통합]
    P --> Q[중복 제거]
    Q --> R[순위 정렬]
    R --> S[응답 포맷팅]
    S --> T[최종 결과 반환]
```

## 4. 하이브리드 재랭킹 상세 플로우

```mermaid
flowchart TD
    A[재랭킹 요청] --> B{FlashRank 초기화됨?}
    
    B -->|No| C[FlashRank 모델 로드<br/>GTE-Multilingual]
    B -->|Yes| D[FlashRank 점수 계산]
    C --> D
    
    D --> E{MRC 모델 초기화됨?}
    E -->|No| F[MRC 모델 로드]
    E -->|Yes| G[MRC 질문-답변 생성]
    F --> G
    
    G --> H[MRC 신뢰도 점수 계산]
    H --> I[캐릭터 위치 정보 추출]
    I --> J[하이브리드 점수 계산]
    
    J --> K[가중 평균 적용<br/>FlashRank×0.5 + MRC×0.3 + 원본×0.2]
    K --> L[점수 정규화]
    L --> M[임계치 필터링]
    M --> N[상위 결과 선택]
    N --> O[결과 반환]
```

## 5. LLM 응답 생성 플로우

```mermaid
flowchart TD
    A[최종 프롬프트 생성] --> B{LLM_PROVIDER 설정}
    
    B -->|vllm| C[vLLM 서버 호출<br/>Gemma3 12B]
    B -->|ollama| D[Ollama 호출<br/>Gemma3:12b]
    
    C --> E{스트리밍 모드?}
    D --> E
    
    E -->|Yes| F[스트리밍 응답 생성<br/>실시간 토큰 전송]
    E -->|No| G[전체 응답 생성]
    
    F --> H[토큰 누적]
    H --> I{응답 완료?}
    I -->|No| F
    I -->|Yes| J[응답 파싱]
    
    G --> J
    
    J --> K[인용 패턴 추출<br/>📚[숫자]]
    K --> L[참고문헌 생성]
    L --> M[구조화된 응답 구성]
    M --> N[세션에 봇 응답 저장]
    N --> O[최종 응답 반환]
```

## 6. 병렬 처리 및 배치 처리

```mermaid
flowchart TD
    A[대량 요청] --> B[요청 큐]
    B --> C[워커 풀 분배]
    
    C --> D[임베딩 배치 처리<br/>50개씩]
    C --> E[검색 멀티스레드<br/>10개 스레드]
    C --> F[재랭킹 GPU 워커<br/>7개 워커]
    
    D --> G[벡터 DB 병렬 검색]
    E --> G
    F --> H[GPU 병렬 처리]
    
    G --> I[결과 집계]
    H --> I
    I --> J[응답 반환]
    
    subgraph "GPU 병렬 처리"
        F1[FlashRank 워커 1]
        F2[FlashRank 워커 2]
        F3[FlashRank 워커 3]
        M1[MRC 워커 1]
        M2[MRC 워커 2]
        M3[MRC 워커 3]
        M4[MRC 워커 4]
    end
```

## 7. 에러 처리 및 복구 플로우

```mermaid
flowchart TD
    A[요청 시작] --> B{서비스 상태 확인}
    
    B -->|정상| C[정상 처리]
    B -->|장애| D[장애 감지]
    
    D --> E{RAG 서비스 장애?}
    E -->|Yes| F[빈 검색 결과 사용]
    E -->|No| G{Reranker 장애?}
    
    G -->|Yes| H[원본 검색 결과 사용]
    G -->|No| I{LLM 서비스 장애?}
    
    I -->|Yes| J[에러 메시지 반환]
    I -->|No| K[정상 처리 계속]
    
    F --> L[부분 응답 생성]
    H --> L
    K --> L
    J --> M[클라이언트 에러 응답]
    L --> N[로깅 및 모니터링]
    N --> O[응답 반환]
```

## 8. 성능 모니터링 플로우

```mermaid
flowchart TD
    A[요청 시작] --> B[타이머 시작]
    B --> C[세션 로드 시간 측정]
    C --> D[Query Rewrite 시간 측정]
    D --> E[RAG 검색 시간 측정]
    E --> F[재랭킹 시간 측정]
    F --> G[LLM 응답 시간 측정]
    G --> H[총 처리 시간 계산]
    
    H --> I[성능 메트릭 수집]
    I --> J[토큰 수 추적]
    J --> K[GPU 사용률 모니터링]
    K --> L[메모리 사용량 추적]
    L --> M[로그 파일 저장]
    M --> N[통계 DB 저장]
    N --> O[대시보드 업데이트]
```

## 9. 세션 관리 플로우

```mermaid
flowchart TD
    A[세션 요청] --> B{세션 파일 존재?}
    
    B -->|Yes| C[세션 파일 로드]
    B -->|No| D[새 세션 생성]
    
    C --> E[토큰 수 계산]
    D --> E
    
    E --> F{토큰 제한 초과?}
    F -->|Yes| G[LRU 방식으로 오래된 메시지 제거]
    F -->|No| H[메시지 추가]
    
    G --> H
    H --> I[세션 파일 저장]
    I --> J[세션 만료 시간 설정]
    J --> K[세션 목록 업데이트]
    
    subgraph "세션 정리 프로세스"
        L[정기 세션 정리]
        L --> M{세션 만료?}
        M -->|Yes| N[세션 파일 삭제]
        M -->|No| O[세션 유지]
    end
```

## 10. 전체 시스템 데이터 플로우

```mermaid
graph LR
    A[사용자 입력] --> B[Query Rewrite]
    B --> C[도메인 셀렉터]
    C --> D[RAG 검색]
    D --> E[벡터 DB]
    E --> F[임베딩 모델]
    F --> G[검색 결과]
    G --> H[하이브리드 재랭킹]
    H --> I[FlashRank]
    H --> J[MRC]
    H --> K[원본 점수]
    I --> L[가중 평균]
    J --> L
    K --> L
    L --> M[필터링된 결과]
    M --> N[컨텍스트 구성]
    N --> O[프롬프트 생성]
    O --> P[LLM 응답]
    P --> Q[구조화된 응답]
    Q --> R[세션 저장]
    R --> S[최종 응답]
```
