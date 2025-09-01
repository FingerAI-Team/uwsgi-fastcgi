# 간단한 위계형 RAG 시스템

기존 RAG 시스템과 완전히 호환되면서 위계형 구조만 지원하는 간단한 시스템입니다.

## 🎯 **핵심 특징**

1. **완전 호환성**: 기존 RAG 스키마와 100% 호환
2. **최소 추가**: 위계형 필드 3개만 추가 (`article_number`, `paragraph_number`, `item_number`)
3. **간단한 기능**: 조문 참조 검색만 추가, 나머지는 기존 검색 활용
4. **기존 코드 재사용**: InteractManager를 그대로 활용
5. **설정 파일 불필요**: 모든 복잡한 설정 제거

## 📁 **파일 구조**

```
hierarchical/
├── __init__.py              # 메인 모듈
├── simple_schema.py         # 간단한 위계형 스키마
├── simple_retriever.py      # 간단한 위계형 검색기
├── test_simple_system.py    # 테스트 파일
└── README.md               # 이 파일
```

## 🔧 **스키마 구조**

### **기존 RAG 필드 (완전 동일)**
- `passage_uid` (VARCHAR, 1024, PRIMARY KEY)
- `doc_id` (VARCHAR, 1024)
- `raw_doc_id` (VARCHAR, 1024)
- `passage_id` (INT64)
- `domain` (VARCHAR, 32)
- `title` (VARCHAR, 1024)
- `author` (VARCHAR, 128)
- `text` (VARCHAR, 10000)
- `text_emb` (FLOAT_VECTOR, 1024)
- `info` (JSON)
- `tags` (JSON)

### **추가된 위계형 필드 (3개만)**
- `article_number` (VARCHAR, 64) - 조 번호 (제1조, 제2조의2 등)
- `paragraph_number` (VARCHAR, 32) - 항 번호 (①, ②, ③ 등)
- `item_number` (VARCHAR, 32) - 호 번호 (1., 2., 3. 등)

## 🚀 **사용법**

### **1. 스키마 생성**
```python
from hierarchical import SimpleHierarchicalSchema

# 스키마 생성
schema = SimpleHierarchicalSchema()

# 호환 필드 가져오기
fields = schema.get_compatible_fields()

# 컬렉션 생성
collection = schema.create_compatible_collection(vectorenv, "legal_documents")
```

### **2. 검색기 사용**
```python
from hierarchical import SimpleHierarchicalRetriever

# 검색기 초기화 (기존 InteractManager와 함께)
retriever = SimpleHierarchicalRetriever(existing_interact_manager)

# 검색 실행
results = retriever.search(
    collection_name="legal_documents",
    query="제1조의 내용이 궁금해요",
    search_params={"top_k": 5}
)
```

### **3. API 사용**
```bash
# 간단한 위계형 검색
curl -X POST "http://localhost/rag/legal/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "제1조의 내용",
    "collection_name": "legal_documents",
    "top_k": 5
  }'
```

## 🔍 **검색 기능**

### **조문 참조 검색**
- **제1조**: 정확한 조문 매칭
- **제2조의2**: 조문의 조 매칭
- **제3항**: 항 매칭
- **제1호**: 호 매칭

### **일반 검색**
- 조문 참조가 없으면 기존 벡터 검색 사용
- InteractManager의 모든 기능 활용

## 🧪 **테스트**

```bash
# 테스트 실행
cd rag/hierarchical
python test_simple_system.py
```

## 📊 **기존 시스템과의 비교**

| 기능 | 기존 RAG | 복잡한 Hierarchical | 간단한 Hierarchical |
|------|----------|-------------------|-------------------|
| 스키마 호환성 | ✅ | ❌ | ✅ |
| 조문 참조 검색 | ❌ | ✅ | ✅ |
| 복잡한 설정 | ❌ | ✅ | ❌ |
| 기존 코드 재사용 | ✅ | ❌ | ✅ |
| 유지보수성 | ✅ | ❌ | ✅ |

## 🎯 **적용 시나리오**

1. **기존 RAG 시스템 확장**: 법령 조문 구조 지원이 필요한 경우
2. **점진적 마이그레이션**: 기존 시스템을 건드리지 않고 기능 추가
3. **간단한 위계형 검색**: 복잡한 설정 없이 조문 참조만 필요한 경우

## 🔄 **마이그레이션 가이드**

### **기존 RAG에서 간단한 Hierarchical로**

1. **스키마 업그레이드**
```python
# 기존 컬렉션에 위계형 필드 추가
schema = SimpleHierarchicalSchema()
fields = schema.get_compatible_fields()
# 기존 컬렉션에 필드 추가
```

2. **데이터 마이그레이션**
```python
# 기존 데이터에 위계형 필드 값 추가
# article_number, paragraph_number, item_number 필드 설정
```

3. **검색 코드 변경**
```python
# 기존 검색 → 간단한 위계형 검색
from hierarchical import SimpleHierarchicalRetriever
retriever = SimpleHierarchicalRetriever(interact_manager)
```

## 📝 **주의사항**

1. **기존 데이터**: 위계형 필드가 없는 기존 데이터는 `NULL` 값으로 처리
2. **성능**: 조문 참조 검색은 정확하지만, 일반 검색은 기존과 동일한 성능
3. **확장성**: 필요시 추가 위계형 필드 확장 가능

## 🤝 **기여**

이 시스템은 기존 RAG 시스템의 호환성을 최우선으로 설계되었습니다.
모든 변경사항은 기존 코드에 영향을 주지 않도록 구현되었습니다.
