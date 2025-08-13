# RAG Legal API 명세서

## 개요
법령 문서를 위계형 구조로 인덱싱하고 검색하는 API입니다. Vector + BM25 하이브리드 검색, 고급 재랭킹, 위계 컨텍스트, 결과 설명을 제공합니다.

---

## 1. 법령 검색 API

### 1.1 통합 법령 검색 (`/rag/legal/search`)

**엔드포인트:** `POST /rag/legal/search`

**설명:** Vector + BM25 하이브리드 검색, 고급 재랭킹, 위계 컨텍스트, 결과 설명을 모두 제공하는 통합 검색 API

**Request Body:**
```json
{
    "query": "개인정보 처리에 관한 규정",
    "domains": ["nanet_related_law_cstt"],
    "top_k": 15,
    "enable_explanation": true,
    "enable_context": true,
    "filter_conditions": {},
    "explanation_mode": true
}
```

**파라미터:**
- `query` (필수): 검색 쿼리
- `domains` (선택): 검색할 도메인 목록 (기본값: `["nanet_related_law_cstt"]`)
- `top_k` (선택): 반환할 결과 수 (기본값: 15)
- `enable_explanation` (선택): 결과 설명 활성화 (기본값: true)
- `enable_context` (선택): 컨텍스트 강화 활성화 (기본값: true)
- `filter_conditions` (선택): 필터 조건
- `explanation_mode` (선택): 설명 모드 (기본값: true)

**Response:**
```json
{
    "status": "success",
    "data": {
        "query": "개인정보 처리에 관한 규정",
        "total_results": 10,
        "results": [
            {
                "node_id": "node_123",
                "content": "제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써...",
                "title": "개인정보보호법",
                "article_number": "제1조",
                "law_title": "개인정보보호법",
                "law_type": "법률",
                "hierarchy_path": "개인정보보호법 > 제1장 총칙 > 제1조",
                "final_rerank_score": 0.95,
                "final_rank": 1,
                "explanation": "이 결과는 개인정보 처리에 관한 핵심 규정으로...",
                "context": "관련 조문들과의 연관성..."
            }
        ],
        "api_response_time_ms": 1250
    }
}
```

### 1.2 간소화된 법령 검색 (`/rag/legal/search/simple`)

**엔드포인트:** `POST /rag/legal/search/simple`

**설명:** 빠른 응답을 위한 기본 검색 (컨텍스트 강화 및 상세 설명 제외)

**Request Body:**
```json
{
    "query": "개인정보 처리에 관한 규정",
    "top_k": 10,
    "collection_name": "legal_documents",
    "filter_conditions": {}
}
```

**Response:**
```json
{
    "status": "success",
    "data": {
        "query": "개인정보 처리에 관한 규정",
        "total_results": 8,
        "results": [
            {
                "node_id": "node_123",
                "content": "제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써...",
                "title": "개인정보보호법",
                "article_number": "제1조",
                "law_title": "개인정보보호법",
                "law_type": "법률",
                "hierarchy_path": "개인정보보호법 > 제1장 총칙 > 제1조",
                "score": 0.95,
                "rank": 1
            }
        ],
        "response_time_ms": 450,
        "search_type": "simple_legal_search"
    }
}
```

### 1.3 검색 시스템 상태 조회 (`/rag/legal/search/status`)

**엔드포인트:** `GET /rag/legal/search/status`

**설명:** 법령 검색 시스템 상태 조회

**Response:**
```json
{
    "status": "success",
    "data": {
        "system_status": "healthy",
        "components": {
            "milvus": "connected",
            "meilisearch": "connected",
            "reranker": "available"
        },
        "collections": ["nanet_related_law_cstt"],
        "total_documents": 15000
    }
}
```

### 1.4 검색 통계 조회 (`/rag/legal/search/stats`)

**엔드포인트:** `GET /rag/legal/search/stats`

**설명:** 법령 검색 시스템 통계 조회

**Response:**
```json
{
    "status": "success",
    "data": {
        "total_searches": 1250,
        "average_response_time_ms": 850,
        "success_rate": 0.98,
        "popular_queries": ["개인정보", "계약", "손해배상"],
        "search_distribution": {
            "integrated_search": 800,
            "simple_search": 450
        }
    }
}
```

---

## 2. 법령 인덱싱 API

### 2.1 법령 문서 인덱싱 (`/rag/legal/insert`)

**엔드포인트:** `POST /rag/legal/insert`

**설명:** 법령 문서를 위계형 구조로 인덱싱합니다. 문서 하나가 여러 노드로 자동 분할되어 인덱싱됩니다 (조문/항/호 단위).

**Request Body:**
```json
{
    "documents": [
        {
            "title": "개인정보보호법",
            "text": "제1장 총칙\n제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써...\n\n제2조(정의) ① 이 법에서 사용하는 용어의 뜻은 다음과 같다.\n1. 개인정보란...",
            "law_type": "법률",
            "law_number": "법률 제11690호",
            "domain": "nanet_related_law_cstt"
        }
    ],
    "ignore_duplicates": true,
    "enable_meilisearch": true
}
```

**파라미터:**
- `documents` (필수): 인덱싱할 문서 목록
  - `title` (필수): 문서 제목
  - `text` (필수): 문서 내용
  - `law_type` (선택): 법령 유형 (자동 추출 또는 기본값 "법률")
  - `law_number` (선택): 법률 번호 (자동 추출)
  - `domain` (필수): 도메인 (컬렉션명 역할)
- `ignore_duplicates` (선택): 중복 무시 (기본값: true)
- `enable_meilisearch` (선택): Meilisearch 인덱싱 (기본값: true)

**자동 생성 필드 (16개 중 10개):**
- `node_id`: 노드 고유 ID
- `document_id`: 문서 고유 ID
- `hierarchy_level`: 위계 레벨
- `parent_node_id`: 부모 노드 ID
- `hierarchy_path`: 위계 경로
- `content_embedding`: 내용 임베딩
- `created_at`: 생성 시간
- `article_number`: 조문 번호
- `paragraph_number`: 항 번호
- `item_number`: 호 번호

**Response:**
```json
{
    "status": "success",
    "summary": {
        "total_documents": 1,
        "successful_documents": 1,
        "failed_documents": 0,
        "total_nodes_indexed": 25,
        "collections_used": ["nanet_related_law_cstt"],
        "meilisearch_enabled": true
    },
    "results": [
        {
            "document_index": 0,
            "title": "개인정보보호법",
            "status": "success",
            "total_nodes": 25,
            "milvus_result": {
                "inserted_count": 25,
                "collection_name": "nanet_related_law_cstt"
            },
            "meilisearch_result": {
                "indexed_count": 25,
                "index_name": "nanet_related_law_cstt"
            },
            "processing_time_seconds": 2.5
        }
    ],
    "performance": {
        "total_api_time_seconds": 2.5,
        "average_time_per_document": 2.5
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

---

## 3. 법령 삭제 API

### 3.1 법령 문서/노드 삭제 (`/rag/legal/delete`)

**엔드포인트:** `DELETE /rag/legal/delete`

**설명:** 법령 문서 또는 노드 삭제

**Request Body:**
```json
{
    "collection_name": "legal_documents",
    "delete_type": "document",
    "target_ids": ["document_id_1", "document_id_2"],
    "delete_from_meilisearch": true
}
```

**파라미터:**
- `collection_name` (선택): 컬렉션명 (기본값: "legal_documents")
- `delete_type` (선택): 삭제 타입 - "document" 또는 "node" (기본값: "document")
- `target_ids` (필수): 삭제할 대상 ID 목록
- `delete_from_meilisearch` (선택): Meilisearch에서도 삭제 (기본값: true)

**Response:**
```json
{
    "status": "success",
    "summary": {
        "delete_type": "document",
        "total_targets": 2,
        "successful_deletes": 2,
        "failed_deletes": 0,
        "collection_name": "legal_documents",
        "meilisearch_enabled": true
    },
    "results": [
        {
            "document_id": "document_id_1",
            "status": "success",
            "result": {
                "deleted_nodes": 15,
                "collection_name": "legal_documents",
                "meilisearch_result": {
                    "deleted_count": 15
                }
            }
        }
    ],
    "performance": {
        "total_api_time_seconds": 1.2
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

---

## 4. 컬렉션 관리 API

### 4.1 컬렉션 목록 조회 (`/rag/legal/collections`)

**엔드포인트:** `GET /rag/legal/collections`

**설명:** 법령 컬렉션 목록 조회

**Response:**
```json
{
    "status": "success",
    "data": {
        "collections": [
            "nanet_related_law_cstt",
            "legal_documents"
        ],
        "total_count": 2
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

### 4.2 컬렉션 정보 조회 (`/rag/legal/collections/{collection_name}/info`)

**엔드포인트:** `GET /rag/legal/collections/{collection_name}/info`

**설명:** 특정 법령 컬렉션 정보 조회

**Response:**
```json
{
    "status": "success",
    "data": {
        "collection_name": "nanet_related_law_cstt",
        "total_documents": 15000,
        "total_nodes": 125000,
        "created_at": "2024-01-01T00:00:00",
        "last_updated": "2024-01-15T10:30:00",
        "schema": {
            "fields": 16,
            "indexed_fields": ["content", "title", "article_number"]
        }
    },
    "timestamp": "2024-01-15T10:30:00"
}
```

---

## 5. 에러 응답 형식

모든 API에서 오류 발생 시 다음과 같은 형식으로 응답합니다:

```json
{
    "status": "error",
    "message": "오류 메시지",
    "processing_time_seconds": 1.5,
    "timestamp": "2024-01-15T10:30:00"
}
```

---

## 6. 주요 특징

### 6.1 위계형 구조
- 법령 문서가 조문/항/호 단위로 자동 분할
- 위계 관계를 유지하며 인덱싱
- 부모-자식 관계 추적

### 6.2 하이브리드 검색
- Vector 검색 (의미적 유사성)
- BM25 검색 (키워드 매칭)
- 두 결과를 결합하여 최적화된 검색 결과 제공

### 6.3 고급 재랭킹
- 검색 결과의 품질 향상
- 법령 특화 재랭킹 알고리즘 적용

### 6.4 컨텍스트 강화
- 관련 조문들과의 연관성 분석
- 검색 결과의 맥락 제공

### 6.5 결과 설명
- 검색 결과에 대한 상세 설명
- 왜 해당 결과가 검색되었는지 설명

---

## 7. 사용 예시

### 7.1 법령 문서 인덱싱
```bash
curl -X POST http://localhost:5000/rag/legal/insert \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "title": "개인정보보호법",
        "text": "제1장 총칙\n제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써...",
        "domain": "nanet_related_law_cstt"
      }
    ]
  }'
```

### 7.2 법령 검색
```bash
curl -X POST http://localhost:5000/rag/legal/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "개인정보 처리에 관한 규정",
    "top_k": 10
  }'
```

### 7.3 법령 삭제
```bash
curl -X DELETE http://localhost:5000/rag/legal/delete \
  -H "Content-Type: application/json" \
  -d '{
    "delete_type": "document",
    "target_ids": ["document_id_1"]
  }'
```
