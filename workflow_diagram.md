# Prompt Chatbot → Enhanced Search 완전 워크플로우 (Mermaid 플로우차트)

## 1. 전체 시스템 아키텍처 (상세)

```mermaid
graph TB
    Client[클라이언트] --> Nginx[Nginx Proxy<br/>포트 12321]
    
    Nginx --> Prompt[Prompt Service<br/>FastCGI /tmp/prompt.sock]
    Nginx --> RAG[RAG Service<br/>FastCGI /tmp/rag.sock]
    Nginx --> Reranker[Reranker Service<br/>FastCGI /tmp/reranker.sock]
    Nginx --> Vision[Vision Service<br/>FastCGI /tmp/vision.sock]
    
    Prompt --> LLM_Gateway[LLM Gateway<br/>Nginx Proxy]
    Prompt --> DomainSelector[도메인 셀렉터<br/>DomainService]
    Prompt --> QueryRewriter[쿼리 재작성<br/>QueryRewriter]
    Prompt --> SessionManager[세션 관리<br/>SessionManager]
    Prompt --> SemanticChunker[시멘틱 청커<br/>SemanticChunker]
    
    RAG --> Milvus[Milvus Vector DB<br/>etcd + minio]
    RAG --> Embedding[BGE-M3<br/>임베딩 모델]
    RAG --> Meilisearch[Meilisearch<br/>하이브리드 검색]
    RAG --> LegalSearch[통합 법령 검색<br/>IntegratedLegalSearch]
    
    Reranker --> FlashRank[FlashRank<br/>GTE-Multilingual]
    Reranker --> MRC[MRC<br/>Machine Reading Comprehension]
    Reranker --> HybridEngine[하이브리드 엔진<br/>HybridSearchEngine]
    
    LLM_Gateway --> Ollama[Ollama<br/>Gemma3:12b]
    LLM_Gateway --> VLLM[vLLM<br/>Mistral-7B]
    LLM_Gateway --> LLM_Server[LLM Server<br/>Gemma3 12B L40S]
    
    subgraph "GPU Resources"
        Ollama
        VLLM
        LLM_Server
        FlashRank
        MRC
        Vision
    end
    
    subgraph "Vector Database"
        Milvus
        Embedding
        Meilisearch
    end
    
    subgraph "Storage & Cache"
        SessionManager
        SemanticChunker
        DomainSelector
        QueryRewriter
    end
```

## 2. Prompt Chatbot 완전 워크플로우 (6단계 체인)

```mermaid
flowchart TD
    A[클라이언트 요청<br/>POST /prompt/chatbot] --> B[요청 파싱 및 검증]
    
    B --> C{세션 ID 존재?}
    C -->|Yes| D[기존 세션 파일 로드<br/>SessionManager.load_session]
    C -->|No| E[새 세션 생성<br/>SessionManager.create_session]
    
    D --> F[사용자 메시지 추가<br/>SessionManager.add_user_message]
    E --> F
    
    F --> G[토큰 수 계산<br/>tiktoken 사용]
    G --> H{토큰 제한 초과?<br/>max_total_tokens: 10000}
    
    H -->|Yes| I[LRU 방식으로 오래된 메시지 제거<br/>SessionManager.trim_history]
    H -->|No| J[Query Rewrite 수행<br/>QueryRewriter.rewrite_query]
    I --> J
    
    J --> K[vLLM 호출<br/>Mistral-7B 모델]
    K --> L[대화 기록 분석<br/>최근 3턴 추출]
    L --> M[쿼리 재작성 프롬프트 생성]
    M --> N[재작성 결과 파싱]
    N --> O{신뢰도 > 0.7?}
    
    O -->|Yes| P[재작성된 쿼리 사용<br/>rewritten_query]
    O -->|No| Q[원본 쿼리 사용<br/>original_query]
    
    P --> R[도메인 셀렉터 실행<br/>DomainService.process_query]
    Q --> R
    
    R --> S{사용자 지정 도메인?<br/>domains 파라미터}
    S -->|Yes| T[사용자 도메인 사용 ]
    S -->|No| U{도메인 셀렉터 결과?<br/>domain_candidates}
    
    U -->|Yes| V[자동 선택 도메인<br/>]
    U -->|No| W[전체 도메인 검색<br/>빈 도메인 리스트]
    
    T --> X[Enhanced Search API 호출<br/>POST /prompt/enhanced_search]
    V --> X
    W --> X
    
    X --> Y[RAG 검색 단계<br/>_perform_enhanced_search]
    Y --> Z[하이브리드 재랭킹<br/>FlashRank + MRC + 원본점수]
    Z --> AA[결과 후처리<br/>메타데이터 통합]
    AA --> BB[컨텍스트 구성<br/>format_context]
    BB --> CC{검색 결과 있음?}
    
    CC -->|Yes| DD[시스템 프롬프트 로드<br/>rag_chat_with_docs.txt]
    CC -->|No| EE[시스템 프롬프트 로드<br/>rag_chat_no_docs.txt]
    
    DD --> FF[최종 프롬프트 생성<br/>build_prompt]
    EE --> FF
    
    FF --> GG{시멘틱 청킹 사용?}
    GG -->|Yes| HH[시멘틱 청킹 적용<br/>build_prompt_context_with_semantic_chunking]
    GG -->|No| II[기본 프롬프트 구성<br/>build_prompt_context]
    
    HH --> JJ[LLM 응답 생성<br/>LLM Gateway API 호출]
    II --> JJ
    
    JJ --> KK{LLM_PROVIDER 설정}
    KK -->|vllm| LL[vLLM 서버 호출<br/>Gemma3 12B]
    KK -->|ollama| MM[Ollama 호출<br/>Gemma3:12b]
    
    LL --> NN{스트리밍 모드?}
    MM --> NN
    
    NN -->|Yes| OO[스트리밍 응답 생성<br/>실시간 토큰 전송]
    NN -->|No| PP[전체 응답 생성<br/>일반 HTTP 응답]
    
    OO --> QQ[토큰 누적<br/>accumulated_response]
    QQ --> RR{응답 완료?<br/>done: true}
    RR -->|No| OO
    RR -->|Yes| SS[응답 파싱<br/>parse_structured_response]
    
    PP --> SS
    
    SS --> TT[인용 패턴 추출<br/> 정규식]
    TT --> UU[참고문헌 생성<br/>references 배열]
    UU --> VV[구조화된 응답 구성<br/>JSON 형식]
    VV --> WW[세션에 봇 응답 저장<br/>SessionManager.add_bot_message]
    WW --> XX[세션 파일 저장<br/>SessionManager.save_session]
    XX --> YY[최종 응답 반환<br/>JSON/SSE]
```

## 3. Enhanced Search 완전 플로우 (RAG + Reranker)

```mermaid
flowchart TD
    A[Enhanced Search API<br/>POST /prompt/enhanced_search] --> B[파라미터 검증<br/>query, top_m, top_n, threshold]
    
    B --> C[검색 파라미터 구성<br/>search_params]
    C --> D[도메인 필터 적용<br/>domains, domain]
    D --> E[추가 필터 적용<br/>author, start_date, end_date, title, info_filter, tags_filter]
    
    E --> F[RAG 검색 요청<br/>POST /rag/search]
    F --> G[BGE-M3 임베딩 생성<br/>emb_model.encode]
    G --> H[임베딩 배치 처리<br/>50개씩]
    H --> I[Milvus 벡터 검색<br/>Collection.search]
    I --> J[멀티스레드 검색<br/>10개 스레드]
    J --> K{검색 결과 있음?}
    
    K -->|No| L[빈 결과 반환<br/>empty array]
    K -->|Yes| M[검색 결과 전처리<br/>전처리 및 정규화]
    
    M --> N[재랭킹 데이터 준비<br/>rerank_passages 배열]
    N --> O[원본 결과 매핑<br/>original_results_by_id]
    O --> P[하이브리드 재랭킹 시작<br/>POST /reranker/rerank]
    
    P --> Q{FlashRank 초기화됨?<br/>flashrank_model}
    Q -->|No| R[FlashRank 모델 로드<br/>GTE-Multilingual]
    Q -->|Yes| S[FlashRank 점수 계산<br/>flashrank_score]
    R --> S
    
    S --> T{MRC 모델 초기화됨?<br/>mrc_model}
    T -->|No| U[MRC 모델 로드<br/>MRC 모델]
    T -->|Yes| V[MRC 질문-답변 생성<br/>mrc_answer]
    U --> V
    
    V --> W[MRC 신뢰도 점수 계산<br/>mrc_score]
    W --> X[캐릭터 위치 정보 추출<br/>mrc_char_ids]
    X --> Y[하이브리드 점수 계산<br/>가중 평균]
    
    Y --> Z[가중 평균 적용<br/>FlashRank×0.5 + MRC×0.3 + 원본×0.2]
    Z --> AA[점수 정규화<br/>0-1 범위로 정규화]
    AA --> BB[임계치 필터링<br/>score < threshold 제외]
    BB --> CC[상위 N개 결과 선택<br/>top_n]
    CC --> DD[메타데이터 통합<br/>title, author, date, link]
    DD --> EE[중복 제거<br/>doc_id 기반]
    EE --> FF[순위 정렬<br/>hybrid_score 기준]
    FF --> GG[응답 포맷팅<br/>JSON 구조화]
    GG --> HH[성능 메트릭 수집<br/>처리 시간, 결과 수]
    HH --> II[최종 결과 반환<br/>enhanced_search_response]
```

## 4. 하이브리드 재랭킹 상세 플로우

```mermaid
flowchart TD
    A[재랭킹 요청<br/>POST /reranker/rerank] --> B[요청 데이터 파싱<br/>query, results, total]
    
    B --> C[재랭킹 파라미터 추출<br/>top_k, weight_flashrank, weight_mrc, weight_original]
    C --> D[결과 데이터 전처리<br/>passage_id, doc_id, text, score]
    
    D --> E{FlashRank 모델 로드됨?<br/>self.flashrank_model}
    E -->|No| F[FlashRank 모델 초기화<br/>gte-multilingual-reranker-base]
    E -->|Yes| G[FlashRank 점수 계산<br/>쿼리-문서 관련도]
    F --> G
    
    G --> H[FlashRank 배치 처리<br/>GPU 가속]
    H --> I[FlashRank 점수 정규화<br/>0-1 범위]
    I --> J{MRC 모델 로드됨?<br/>self.mrc_model}
    
    J -->|No| K[MRC 모델 초기화<br/>MRC 모델 로드]
    J -->|Yes| L[MRC 질문-답변 생성<br/>질문에 대한 직접 답변]
    K --> L
    
    L --> M[MRC 배치 처리<br/>GPU 병렬 처리]
    M --> N[MRC 신뢰도 점수 계산<br/>답변 품질 평가]
    N --> O[캐릭터 위치 정보 추출<br/>답변 위치 인덱스]
    O --> P[원본 점수 유지<br/>original_score]
    
    P --> Q[하이브리드 점수 계산<br/>가중 평균 공식]
    Q --> R[점수 정규화<br/>최종 점수 0-1]
    R --> S[임계치 필터링<br/>score < threshold 제외]
    S --> T[상위 결과 선택<br/>top_k개]
    T --> U[결과 정렬<br/>hybrid_score 기준 내림차순]
    U --> V[메타데이터 추가<br/>rerank_position, reranked: true]
    V --> W[응답 구성<br/>reranker_response]
    W --> X[로깅 및 모니터링<br/>처리 시간, 결과 수]
    X --> Y[최종 재랭킹 결과 반환]
```

## 5. Query Rewrite 상세 플로우

```mermaid
flowchart TD
    A[Query Rewrite 시작<br/>QueryRewriter.rewrite_query] --> B[대화 기록 분석<br/>최근 3턴 추출]
    
    B --> C[대화 컨텍스트 구성<br/>user/bot 메시지 쌍]
    C --> D[현재 쿼리 분석<br/>쿼리 유형 분류]
    D --> E[재작성 필요성 판단<br/>대화 맥락 기반]
    
    E --> F{재작성 필요?<br/>대화 연속성 체크}
    F -->|No| G[원본 쿼리 반환<br/>confidence: 0.0]
    F -->|Yes| H[재작성 프롬프트 생성<br/>query_rewrite.txt 템플릿]
    
    H --> I[vLLM 호출<br/>Mistral-7B 모델]
    I --> J[재작성 결과 파싱<br/>JSON 응답 파싱]
    J --> K{파싱 성공?}
    
    K -->|No| L[원본 쿼리 반환<br/>파싱 실패 처리]
    K -->|Yes| M[재작성된 쿼리 추출<br/>rewritten_query]
    
    M --> N[신뢰도 점수 계산<br/>confidence score]
    N --> O[재작성 이유 추출<br/>reasoning]
    O --> P[결과 검증<br/>쿼리 유효성 체크]
    
    P --> Q{검증 통과?}
    Q -->|No| R[원본 쿼리 반환<br/>검증 실패 처리]
    Q -->|Yes| S[재작성 결과 반환<br/>rewrite_result]
    
    S --> T[로깅<br/>재작성 과정 기록]
    T --> U[최종 결과 반환<br/>original_query, rewritten_query, confidence, reasoning]
```

## 6. 도메인 셀렉터 상세 플로우

```mermaid
flowchart TD
    A[도메인 셀렉터 시작<br/>DomainService.process_query] --> B[쿼리 텍스트 분석<br/>키워드 추출]
    
    B --> C[도메인 키워드 매칭<br/>intent_keywords.json]
    C --> D[의도 분류<br/>의도별 키워드 매칭]
    D --> E[도메인 후보 생성<br/>domain_candidates]
    
    E --> F{도메인 후보 있음?}
    F -->|No| G[전체 도메인 반환<br/>빈 도메인 리스트]
    F -->|Yes| H[도메인 우선순위 계산<br/>가중치 기반]
    
    H --> I[도메인 신뢰도 평가<br/>confidence score]
    I --> J[임계치 필터링<br/>confidence > threshold]
    J --> K[상위 도메인 선택<br/>top domains]
    
    K --> L[도메인 메타데이터 추가<br/>domain_info]
    L --> M[결과 구성<br/>domain_result]
    M --> N[로깅<br/>도메인 선택 과정 기록]
    N --> O[최종 도메인 결과 반환<br/>domain_candidates, confidence]
```

## 7. 세션 관리 상세 플로우

```mermaid
flowchart TD
    A[세션 요청<br/>SessionManager] --> B{세션 파일 존재?<br/>session_id.json}
    
    B -->|Yes| C[세션 파일 로드<br/>JSON 파싱]
    B -->|No| D[새 세션 생성<br/>기본 구조 생성]
    
    C --> E[세션 데이터 검증<br/>JSON 스키마 검증]
    D --> E
    
    E --> F[토큰 수 계산<br/>tiktoken 사용]
    F --> G{토큰 제한 초과?<br/>max_total_tokens: 10000}
    
    G -->|Yes| H[LRU 방식으로 오래된 메시지 제거<br/>가장 오래된 메시지부터 제거]
    G -->|No| I[메시지 추가<br/>user/bot 메시지 추가]
    H --> I
    
    I --> J[토큰 수 업데이트<br/>새로운 토큰 수 계산]
    J --> K{시멘틱 청킹 사용?}
    
    K -->|Yes| L[시멘틱 청킹 적용<br/>SemanticChunker.chunk_history]
    K -->|No| M[기본 메시지 처리<br/>순차적 메시지 처리]
    
    L --> N[중요 메시지 선택<br/>관련도 기반 선택]
    M --> N
    
    N --> O[세션 파일 저장<br/>JSON 형식으로 저장]
    O --> P[세션 만료 시간 설정<br/>TTL 설정]
    P --> Q[세션 목록 업데이트<br/>세션 인덱스 업데이트]
    
    subgraph "세션 정리 프로세스"
        R[정기 세션 정리<br/>cleanup_expired_sessions]
        R --> S{세션 만료?<br/>TTL 체크}
        S -->|Yes| T[세션 파일 삭제<br/>파일 시스템에서 제거]
        S -->|No| U[세션 유지<br/>활성 세션 유지]
    end
```

## 8. LLM 응답 생성 상세 플로우

```mermaid
flowchart TD
    A[LLM 응답 생성 시작<br/>LLM Gateway API] --> B{LLM_PROVIDER 설정<br/>환경변수 확인}
    
    B -->|vllm| C[vLLM 서버 호출<br/>Gemma3 12B]
    B -->|ollama| D[Ollama 호출<br/>Gemma3:12b]
    B -->|llm-server| E[LLM Server 호출<br/>Gemma3 12B L40S]
    
    C --> F{스트리밍 모드?<br/>stream 파라미터}
    D --> F
    E --> F
    
    F -->|Yes| G[스트리밍 응답 생성<br/>Server-Sent Events]
    F -->|No| H[전체 응답 생성<br/>일반 HTTP 응답]
    
    G --> I[스트리밍 연결 설정<br/>HTTP 1.1 keep-alive]
    I --> J[토큰 스트리밍 시작<br/>실시간 토큰 전송]
    J --> K[토큰 누적<br/>accumulated_response]
    K --> L{응답 완료?<br/>done: true}
    L -->|No| J
    L -->|Yes| M[스트리밍 종료<br/>최종 응답 전송]
    
    H --> N[전체 응답 대기<br/>타임아웃: 120초]
    N --> O[응답 텍스트 추출<br/>response 필드]
    
    M --> P[응답 파싱<br/>parse_structured_response]
    O --> P
    
    P --> Q[인용 패턴 추출<br/> 정규식]
    Q --> R[참고문헌 매핑<br/>검색 결과와 매핑]
    R --> S[구조화된 응답 구성<br/>JSON 형식]
    S --> T[세션에 봇 응답 저장<br/>SessionManager.add_bot_message]
    T --> U[최종 응답 반환<br/>JSON/SSE]
```

## 9. 병렬 처리 및 배치 처리 상세 플로우

```mermaid
flowchart TD
    A[대량 요청 처리<br/>Concurrent Requests] --> B[요청 큐<br/>Request Queue]
    B --> C[워커 풀 분배<br/>Worker Pool Distribution]
    
    C --> D[임베딩 배치 처리<br/>50개씩 배치]
    C --> E[검색 멀티스레드<br/>10개 스레드]
    C --> F[재랭킹 GPU 워커<br/>7개 워커]
    C --> G[LLM 병렬 처리<br/>GPU 병렬]
    
    D --> H[벡터 DB 병렬 검색<br/>Milvus 병렬 쿼리]
    E --> H
    F --> I[GPU 병렬 처리<br/>CUDA 병렬 연산]
    G --> I
    
    H --> J[결과 집계<br/>Result Aggregation]
    I --> J
    J --> K[응답 반환<br/>Response Return]
    
    subgraph "GPU 병렬 처리 상세"
        F1[FlashRank 워커 1<br/>GPU 0]
        F2[FlashRank 워커 2<br/>GPU 1]
        F3[FlashRank 워커 3<br/>GPU 2]
        M1[MRC 워커 1<br/>GPU 3]
        M2[MRC 워커 2<br/>GPU 4]
        M3[MRC 워커 3<br/>GPU 5]
        M4[MRC 워커 4<br/>GPU 6]
    end
    
    subgraph "스레드 풀 관리"
        T1[임베딩 스레드 1-5<br/>배치 처리]
        T2[검색 스레드 1-10<br/>벡터 검색]
        T3[후처리 스레드 1-3<br/>결과 정리]
    end
```

## 10. 에러 처리 및 복구 상세 플로우

```mermaid
flowchart TD
    A[요청 시작<br/>Request Start] --> B{서비스 상태 확인<br/>Health Check}
    
    B -->|정상| C[정상 처리<br/>Normal Processing]
    B -->|장애| D[장애 감지<br/>Failure Detection]
    
    D --> E{RAG 서비스 장애?<br/>Milvus 연결 실패}
    E -->|Yes| F[빈 검색 결과 사용<br/>Empty Search Results]
    E -->|No| G{Reranker 장애?<br/>GPU 메모리 부족}
    
    G -->|Yes| H[원본 검색 결과 사용<br/>Original Search Results]
    G -->|No| I{LLM 서비스 장애?<br/>Ollama/vLLM 연결 실패}
    
    I -->|Yes| J[에러 메시지 반환<br/>Error Response]
    I -->|No| K{임베딩 모델 장애?<br/>BGE-M3 로드 실패}
    
    K -->|Yes| L[키워드 기반 검색<br/>Keyword-based Search]
    K -->|No| M{도메인 셀렉터 장애?<br/>DomainService 오류}
    
    M -->|Yes| N[전체 도메인 검색<br/>All Domains Search]
    M -->|No| O{Query Rewrite 장애?<br/>vLLM 연결 실패}
    
    O -->|Yes| P[원본 쿼리 사용<br/>Original Query]
    O -->|No| Q[정상 처리 계속<br/>Continue Processing]
    
    F --> R[부분 응답 생성<br/>Partial Response]
    H --> R
    L --> R
    N --> R
    P --> R
    Q --> R
    J --> S[클라이언트 에러 응답<br/>Client Error Response]
    
    R --> T[로깅 및 모니터링<br/>Logging & Monitoring]
    S --> T
    T --> U[응답 반환<br/>Response Return]
```

## 11. 성능 모니터링 상세 플로우

```mermaid
flowchart TD
    A[요청 시작<br/>Request Start] --> B[타이머 시작<br/>Start Timer]
    B --> C[세션 로드 시간 측정<br/>Session Load Time]
    C --> D[Query Rewrite 시간 측정<br/>Query Rewrite Time]
    D --> E[도메인 셀렉터 시간 측정<br/>Domain Selector Time]
    E --> F[RAG 검색 시간 측정<br/>RAG Search Time]
    F --> G[재랭킹 시간 측정<br/>Reranking Time]
    G --> H[LLM 응답 시간 측정<br/>LLM Response Time]
    H --> I[총 처리 시간 계산<br/>Total Processing Time]
    
    I --> J[성능 메트릭 수집<br/>Performance Metrics]
    J --> K[토큰 수 추적<br/>Token Count Tracking]
    K --> L[GPU 사용률 모니터링<br/>GPU Utilization]
    L --> M[메모리 사용량 추적<br/>Memory Usage]
    M --> N[네트워크 지연 측정<br/>Network Latency]
    N --> O[디스크 I/O 모니터링<br/>Disk I/O]
    O --> P[로그 파일 저장<br/>Log File Storage]
    P --> Q[통계 DB 저장<br/>Statistics DB]
    Q --> R[대시보드 업데이트<br/>Dashboard Update]
    R --> S[알림 발송<br/>Alert Notification]
    S --> T[성능 리포트 생성<br/>Performance Report]
```

## 12. 전체 시스템 데이터 플로우 (완전)

```mermaid
graph LR
    A[사용자 입력<br/>User Input] --> B[Query Rewrite<br/>쿼리 재작성]
    B --> C[도메인 셀렉터<br/>Domain Selector]
    C --> D[RAG 검색<br/>RAG Search]
    D --> E[벡터 DB<br/>Vector Database]
    E --> F[임베딩 모델<br/>BGE-M3]
    F --> G[검색 결과<br/>Search Results]
    G --> H[하이브리드 재랭킹<br/>Hybrid Reranking]
    H --> I[FlashRank<br/>GTE-Multilingual]
    H --> J[MRC<br/>Machine Reading Comprehension]
    H --> K[원본 점수<br/>Original Score]
    I --> L[가중 평균<br/>Weighted Average]
    J --> L
    K --> L
    L --> M[필터링된 결과<br/>Filtered Results]
    M --> N[컨텍스트 구성<br/>Context Formation]
    N --> O[프롬프트 생성<br/>Prompt Generation]
    O --> P[LLM 응답<br/>LLM Response]
    P --> Q[구조화된 응답<br/>Structured Response]
    Q --> R[세션 저장<br/>Session Storage]
    R --> S[최종 응답<br/>Final Response]
    
    subgraph "중간 처리 단계"
        T[세션 관리<br/>Session Management]
        U[시멘틱 청킹<br/>Semantic Chunking]
        V[성능 모니터링<br/>Performance Monitoring]
        W[에러 처리<br/>Error Handling]
    end
```
