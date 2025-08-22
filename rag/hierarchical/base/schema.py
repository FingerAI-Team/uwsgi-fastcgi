"""
위계형 문서 스키마 베이스 클래스

기존 RAG 시스템과 호환되면서 위계형 구조를 지원하는 스키마를 제공합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pymilvus import DataType, FieldSchema, CollectionSchema
import logging


class BaseHierarchicalSchema(ABC):
    """위계형 문서 스키마 베이스 클래스"""
    
    def __init__(self, vector_dim: int = 1024):
        """
        Args:
            vector_dim: 벡터 임베딩 차원 (기본값: 1024, BGE-M3 모델과 호환)
        """
        self.vector_dim = vector_dim
        self.logger = logging.getLogger(__name__)
        
    def get_base_fields(self) -> List[FieldSchema]:
        """
        순수 위계형 필드들 (기존 시스템 호환성 제거)
        
        완전히 새로운 위계형 검색 시스템을 위한 스키마
        """
        try:
            base_fields = [
                # === 핵심 식별자 ===
                FieldSchema(
                    name="node_id", 
                    dtype=DataType.VARCHAR,
                    max_length=128, 
                    is_primary=True,
                    description="위계형 노드 고유 식별자"
                ),
                FieldSchema(
                    name="document_id", 
                    dtype=DataType.VARCHAR, 
                    max_length=256,
                    description="원본 문서 ID"
                ),
                
                # === 위계형 구조 핵심 필드들 ===
                FieldSchema(
                    name="hierarchy_level", 
                    dtype=DataType.INT64,
                    description="위계 레벨 (0=최상위, 1=1단계...)"
                ),
                FieldSchema(
                    name="parent_node_id", 
                    dtype=DataType.VARCHAR, 
                    max_length=128,
                    description="부모 노드 ID (None이면 최상위)"
                ),
                FieldSchema(
                    name="hierarchy_path", 
                    dtype=DataType.VARCHAR, 
                    max_length=2048,
                    description="위계 경로 (예: /root/level1/level2)"
                ),
                
                # === 내용 필드들 ===
                FieldSchema(
                    name="title", 
                    dtype=DataType.VARCHAR, 
                    max_length=1024,
                    description="노드 제목 (법령명/조문 제목 통합)"
                ),
                FieldSchema(
                    name="content", 
                    dtype=DataType.VARCHAR, 
                    max_length=15000,
                    description="노드 내용"
                ),
                
                # === 임베딩 필드 (단일) ===
                FieldSchema(
                    name="content_embedding", 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=self.vector_dim,
                    description="내용 임베딩 벡터"
                ),
                
                # === 기본 메타데이터 ===
                FieldSchema(
                    name="domain", 
                    dtype=DataType.VARCHAR, 
                    max_length=64,
                    description="도메인 분류"
                ),
                FieldSchema(
                    name="created_at", 
                    dtype=DataType.VARCHAR,
                    max_length=32,
                    description="생성 시간"
                ),
            ]
            
            self.logger.info(f"순수 위계형 필드 {len(base_fields)}개 생성 완료")
            return base_fields
            
        except Exception as e:
            self.logger.error(f"기본 필드 생성 중 오류: {e}")
            raise
    
    @abstractmethod
    def get_domain_fields(self) -> List[FieldSchema]:
        """
        도메인별 추가 필드 (서브클래스에서 구현)
        
        Returns:
            List[FieldSchema]: 도메인 전용 필드들
        """
        pass
    
    def get_complete_fields(self) -> List[FieldSchema]:
        """기본 필드 + 도메인 필드 결합"""
        try:
            base_fields = self.get_base_fields()
            domain_fields = self.get_domain_fields()
            
            all_fields = base_fields + domain_fields
            self.logger.info(f"전체 필드 {len(all_fields)}개 (기본: {len(base_fields)}, 도메인: {len(domain_fields)})")
            
            return all_fields
            
        except Exception as e:
            self.logger.error(f"필드 결합 중 오류: {e}")
            raise
    
    def create_schema(self, collection_name: str, description: str = None) -> CollectionSchema:
        """
        완전한 스키마 생성
        
        Args:
            collection_name: 컬렉션 이름
            description: 스키마 설명
            
        Returns:
            CollectionSchema: 생성된 스키마
        """
        try:
            all_fields = self.get_complete_fields()
            
            if description is None:
                description = f"Hierarchical collection schema for {collection_name}"
            
            schema = CollectionSchema(
                fields=all_fields,
                description=description,
                enable_dynamic_field=True  # 동적 필드 허용 (기존 호환성)
            )
            
            self.logger.info(f"스키마 생성 완료: {collection_name}")
            return schema
            
        except Exception as e:
            self.logger.error(f"스키마 생성 중 오류 ({collection_name}): {e}")
            raise
    
    def get_base_indexes(self) -> List[Dict[str, Any]]:
        """
        기본 인덱스 설정 (기존 시스템과 호환)
        
        Returns:
            List[Dict]: 인덱스 설정 리스트
        """
        try:
            base_indexes = [
                # 벡터 인덱스들
                {
                    "field_name": "content_embedding",
                    "index_type": "IVF_FLAT", 
                    "metric_type": "COSINE",
                    "params": {"nlist": 1024}
                },
                # 위계형 구조 인덱스들 (VARCHAR 필드는 FLAT 사용)
                {
                    "field_name": "hierarchy_level", 
                    "index_type": "FLAT"
                },
                {
                    "field_name": "parent_node_id",
                    "index_type": "FLAT"
                },
                {
                    "field_name": "domain",
                    "index_type": "FLAT"
                },
                {
                    "field_name": "document_id",
                    "index_type": "FLAT"
                },
                {
                    "field_name": "domain",
                    "index_type": "STL_SORT"
                },
                {
                    "field_name": "document_id",
                    "index_type": "STL_SORT"
                }
            ]
            
            self.logger.info(f"기본 인덱스 {len(base_indexes)}개 정의")
            return base_indexes
            
        except Exception as e:
            self.logger.error(f"기본 인덱스 정의 중 오류: {e}")
            raise
    
    @abstractmethod 
    def get_domain_indexes(self) -> List[Dict[str, Any]]:
        """
        도메인별 추가 인덱스 (서브클래스에서 구현)
        
        Returns:
            List[Dict]: 도메인 전용 인덱스들
        """
        pass
    
    def get_all_indexes(self) -> List[Dict[str, Any]]:
        """기본 인덱스 + 도메인 인덱스 결합"""
        try:
            base_indexes = self.get_base_indexes()
            domain_indexes = self.get_domain_indexes()
            
            all_indexes = base_indexes + domain_indexes
            self.logger.info(f"전체 인덱스 {len(all_indexes)}개 (기본: {len(base_indexes)}, 도메인: {len(domain_indexes)})")
            
            return all_indexes
            
        except Exception as e:
            self.logger.error(f"인덱스 결합 중 오류: {e}")
            raise
    
    def validate_schema(self) -> bool:
        """스키마 유효성 검증"""
        try:
            # 필수 필드 확인
            required_base_fields = [
                "node_id", "document_id", "hierarchy_level", 
                "title", "content", "content_embedding"
            ]
            
            all_fields = self.get_complete_fields()
            field_names = [field.name for field in all_fields]
            
            for required_field in required_base_fields:
                if required_field not in field_names:
                    self.logger.error(f"필수 필드 누락: {required_field}")
                    return False
            
            # 프라이머리 키 확인
            primary_fields = [field for field in all_fields if field.is_primary]
            if len(primary_fields) != 1:
                self.logger.error(f"프라이머리 키는 정확히 1개여야 함: {len(primary_fields)}개 발견")
                return False
            
            # 벡터 필드 확인 (1개: content_embedding)
            vector_fields = [field for field in all_fields if field.dtype == DataType.FLOAT_VECTOR]
            if len(vector_fields) != 1:
                self.logger.error(f"벡터 필드는 정확히 1개여야 함: {len(vector_fields)}개 발견")
                return False
            
            self.logger.info("스키마 유효성 검증 통과")
            return True
            
        except Exception as e:
            self.logger.error(f"스키마 유효성 검증 중 오류: {e}")
            return False
    
    def get_schema_info(self) -> Dict[str, Any]:
        """스키마 정보 요약"""
        try:
            all_fields = self.get_complete_fields()
            all_indexes = self.get_all_indexes()
            
            field_info = {}
            for field in all_fields:
                field_info[field.name] = {
                    "type": str(field.dtype),
                    "is_primary": field.is_primary,
                    "max_length": getattr(field, 'max_length', None),
                    "dim": getattr(field, 'dim', None)
                }
            
            return {
                "total_fields": len(all_fields),
                "total_indexes": len(all_indexes),
                "vector_dimension": self.vector_dim,
                "fields": field_info,
                "is_valid": self.validate_schema()
            }
            
        except Exception as e:
            self.logger.error(f"스키마 정보 생성 중 오류: {e}")
            return {"error": str(e)}

