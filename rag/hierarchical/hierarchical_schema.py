"""
위계형 스키마

기존 RAG 스키마와 완전히 호환되면서 최소한의 위계형 필드만 추가합니다.
"""

from typing import List, Dict, Any
from pymilvus import DataType, FieldSchema
import logging


class HierarchicalSchema:
    """위계형 스키마 클래스"""
    
    def __init__(self, vector_dim: int = 1024):
        """
        Args:
            vector_dim: 벡터 임베딩 차원 (기본값: 1024)
        """
        self.vector_dim = vector_dim
        self.logger = logging.getLogger(__name__)
    
    def get_compatible_fields(self) -> List[FieldSchema]:
        """
        기존 RAG 스키마와 완전히 호환되는 필드들
        
        기존 필드 + 위계형 필드 3개만 추가
        """
        try:
            # === 기존 RAG 필드들 (완전 동일) ===
            base_fields = [
                # 각 passage의 고유 식별자
                FieldSchema(
                    name="passage_uid", 
                    dtype=DataType.VARCHAR, 
                    max_length=1024, 
                    is_primary=True,
                    description="패시지 고유 식별자"
                ),
                # doc_id는 이제 일반 필드
                FieldSchema(
                    name="doc_id", 
                    dtype=DataType.VARCHAR, 
                    max_length=1024,
                    description="문서 ID"
                ),
                FieldSchema(
                    name="raw_doc_id", 
                    dtype=DataType.VARCHAR, 
                    max_length=1024,
                    description="원본 문서 ID"
                ),
                FieldSchema(
                    name="passage_id", 
                    dtype=DataType.INT64,
                    description="패시지 ID"
                ),
                FieldSchema(
                    name="domain", 
                    dtype=DataType.VARCHAR, 
                    max_length=32,
                    description="도메인"
                ),
                FieldSchema(
                    name="title", 
                    dtype=DataType.VARCHAR, 
                    max_length=1024,
                    description="제목"
                ),
                FieldSchema(
                    name="author", 
                    dtype=DataType.VARCHAR, 
                    max_length=128,
                    description="작성자"
                ),
                FieldSchema(
                    name="text", 
                    dtype=DataType.VARCHAR, 
                    max_length=10000,
                    description="텍스트 내용"
                ),
                FieldSchema(
                    name="text_emb", 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=self.vector_dim,
                    description="텍스트 임베딩 벡터"
                ),
                FieldSchema(
                    name="info", 
                    dtype=DataType.JSON,
                    description="추가 정보"
                ),
                FieldSchema(
                    name="tags", 
                    dtype=DataType.JSON,
                    description="태그 정보"
                ),
                
                # === 위계형 필드 4개 추가 (장 포함) ===
                FieldSchema(
                    name="chapter_number", 
                    dtype=DataType.VARCHAR, 
                    max_length=32,
                    description="장 번호 (제1장, 제2장 등)"
                ),
                FieldSchema(
                    name="chapter_title", 
                    dtype=DataType.VARCHAR, 
                    max_length=256,
                    description="장 제목"
                ),
                FieldSchema(
                    name="section_number", 
                    dtype=DataType.VARCHAR, 
                    max_length=32,
                    description="절 번호 (제1절, 제2절 등)"
                ),
                FieldSchema(
                    name="section_title", 
                    dtype=DataType.VARCHAR, 
                    max_length=256,
                    description="절 제목"
                ),
                FieldSchema(
                    name="division_number", 
                    dtype=DataType.VARCHAR, 
                    max_length=32,
                    description="관 번호 (제1관, 제2관 등)"
                ),
                FieldSchema(
                    name="division_title", 
                    dtype=DataType.VARCHAR, 
                    max_length=256,
                    description="관 제목"
                ),
                FieldSchema(
                    name="article_number", 
                    dtype=DataType.VARCHAR, 
                    max_length=64,
                    description="조 번호 (제1조, 제2조의2 등)"
                ),
                FieldSchema(
                    name="article_title", 
                    dtype=DataType.VARCHAR, 
                    max_length=256,
                    description="조 제목"
                ),
                FieldSchema(
                    name="paragraph_number", 
                    dtype=DataType.VARCHAR, 
                    max_length=32,
                    description="항 번호 (①, ②, ③ 등)"
                ),
                FieldSchema(
                    name="subparagraph_number", 
                    dtype=DataType.VARCHAR, 
                    max_length=32,
                    description="호 번호 (1., 2., 3. 등)"
                ),
                FieldSchema(
                    name="item_number", 
                    dtype=DataType.VARCHAR, 
                    max_length=32,
                    description="목 번호 (가., 나., 다., 라. 등)"
                ),
                FieldSchema(
                    name="is_omission", 
                    dtype=DataType.BOOL,
                    description="생략 여부 (true/false)"
                ),
                FieldSchema(
                    name="is_deletion", 
                    dtype=DataType.BOOL,
                    description="삭제 여부 (true/false)"
                ),
                FieldSchema(
                    name="is_amendment", 
                    dtype=DataType.BOOL,
                    description="개정/신설 여부 (true/false)"
                ),
                FieldSchema(
                    name="is_appendix", 
                    dtype=DataType.BOOL,
                    description="부칙 여부 (true/false)"
                ),
                FieldSchema(
                    name="is_attachment", 
                    dtype=DataType.BOOL,
                    description="별지 여부 (true/false)"
                ),
            ]
            
            self.logger.info(f"✅ 위계형 스키마 생성: {len(base_fields)}개 필드")
            return base_fields
            
        except Exception as e:
            self.logger.error(f"스키마 생성 중 오류: {e}")
            raise
    
    def create_compatible_collection(self, vectorenv, domain_name: str):
        """
        기존 RAG와 호환되는 컬렉션 생성
        
        Args:
            vectorenv: 벡터 환경 관리자
            domain_name: 도메인 이름
        """
        try:
            # 필드 스키마 생성
            schema_fields = self.get_compatible_fields()
            
            # 스키마 생성
            schema = vectorenv.create_schema(
                schema_fields, 
                'compatible hierarchical schema for fai-rag'
            )
            
            # 컬렉션 생성
            collection = vectorenv.create_collection(
                domain_name, 
                schema, 
                shards_num=2
            )
            
            # 인덱스 생성
            vectorenv.create_index(collection, field_name='text_emb')
            
            self.logger.info(f"✅ 호환 컬렉션 생성 완료: {domain_name}")
            return collection
            
        except Exception as e:
            self.logger.error(f"컬렉션 생성 중 오류: {e}")
            raise
