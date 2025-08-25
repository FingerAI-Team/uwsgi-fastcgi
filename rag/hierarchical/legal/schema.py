"""
법령 전용 스키마 클래스

대한민국 법령 체계에 최적화된 스키마를 제공합니다.
법률, 시행령, 시행규칙 등의 위계 구조와 조문 체계를 지원합니다.
"""

from typing import List, Dict, Any
from pymilvus import DataType, FieldSchema
import logging

from ..base.schema import BaseHierarchicalSchema


class LegalSchema(BaseHierarchicalSchema):
    """법령 전용 스키마 클래스"""
    
    def __init__(self, vector_dim: int = 1024):
        """
        Args:
            vector_dim: 벡터 임베딩 차원 (기본값: 1024, BGE-M3 모델과 호환)
        """
        super().__init__(vector_dim)
        self.logger = logging.getLogger(__name__)
        
        # 법령 특화 설정
        self.legal_hierarchy_levels = {
            0: "법령",      # 최상위 (법률명)
            1: "편",       # 편
            2: "장",       # 장  
            3: "절",       # 절
            4: "조",       # 조
            5: "항",       # 항
            6: "호",       # 호
            7: "목",       # 목 (세부 항목)
        }
        
        self.legal_provision_types = [
            "본칙", "부칙", "별표", "별지", "부록"
        ]
        
        self.legal_keywords = [
            "법률", "시행령", "시행규칙", "고시", "훈령", "예규"
        ]
    
    def get_domain_fields(self) -> List[FieldSchema]:
        """
        법령 전용 핵심 필드들 (26개 → 5개로 대폭 축소)
        
        Returns:
            List[FieldSchema]: 법령 검색에 필수적인 필드들만
        """
        try:
            legal_fields = [
                # === 법령 식별 ===
                FieldSchema(
                    name="law_type",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                    description="법령 유형 (법률/시행령/시행규칙/고시)"
                ),
                FieldSchema(
                    name="law_name",
                    dtype=DataType.VARCHAR,
                    max_length=256,
                    description="법령명 (개인정보보호법, 전자상거래법 등)"
                ),
                FieldSchema(
                    name="law_number",
                    dtype=DataType.VARCHAR,
                    max_length=128,
                    description="법률 번호 (법률 제12345호)"
                ),
                
                # === 조문 구조 (핵심) ===
                FieldSchema(
                    name="article_number",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                    description="조 번호 (제1조, 제2조의2 등)"
                ),
                FieldSchema(
                    name="paragraph_number",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                    description="항 번호 (①, ②, ③ 등)"
                ),
                FieldSchema(
                    name="item_number",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                    description="호 번호 (1., 2., 3. 등)"
                ),
                
                # === 법령 날짜 (새로 추가) ===
                FieldSchema(
                    name="enactment_date",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                    description="제정일 (YYYY.MM.DD 형식)"
                ),
            ]
            
            self.logger.info(f"법령 필드 {len(legal_fields)}개 생성 (날짜 필드 포함)")
            return legal_fields
            
        except Exception as e:
            self.logger.error(f"법령 필드 생성 중 오류: {e}")
            raise
    
    def get_domain_indexes(self) -> List[Dict[str, Any]]:
        """
        법령 전용 인덱스 설정 (축소)
        
        Returns:
            List[Dict]: 핵심 검색용 인덱스들만
        """
        try:
            legal_indexes = [
                # === 필수 인덱스만 ===
                {
                    "field_name": "law_type",
                    "index_type": "FLAT"  # VARCHAR 필드는 FLAT 인덱스 사용
                },
                {
                    "field_name": "law_name",
                    "index_type": "FLAT"
                },
                {
                    "field_name": "law_number", 
                    "index_type": "FLAT"
                },
                {
                    "field_name": "article_number",
                    "index_type": "FLAT"
                },
                {
                    "field_name": "paragraph_number",
                    "index_type": "FLAT"
                },
                {
                    "field_name": "item_number",
                    "index_type": "FLAT"
                },
                {
                    "field_name": "enactment_date",
                    "index_type": "FLAT"
                }
            ]
            
            self.logger.info(f"법령 인덱스 {len(legal_indexes)}개 정의 (날짜 인덱스 포함)")
            return legal_indexes
            
        except Exception as e:
            self.logger.error(f"법령 인덱스 정의 중 오류: {e}")
            raise
    
    def validate_legal_schema(self) -> bool:
        """법령 스키마 특화 유효성 검증"""
        try:
            # 기본 검증
            if not self.validate_schema():
                return False
            
            # 법령 특화 검증
            all_fields = self.get_complete_fields()
            field_names = [field.name for field in all_fields]
            
            # 법령 필수 필드 확인 (날짜 필드 포함)
            legal_required_fields = [
                "law_type", "law_name", "law_number", "article_number", "enactment_date"
            ]
            
            for required_field in legal_required_fields:
                if required_field not in field_names:
                    self.logger.error(f"법령 필수 필드 누락: {required_field}")
                    return False
            
            self.logger.info("법령 스키마 유효성 검증 통과")
            return True
            
        except Exception as e:
            self.logger.error(f"법령 스키마 검증 중 오류: {e}")
            return False
    
    def get_legal_schema_info(self) -> Dict[str, Any]:
        """법령 스키마 정보 요약"""
        try:
            base_info = self.get_schema_info()
            
            legal_info = {
                **base_info,
                "schema_type": "legal",
                "supported_law_types": ["법률", "시행령", "시행규칙", "고시", "훈령", "예규"],
                "hierarchy_levels": self.legal_hierarchy_levels,
                "provision_types": self.legal_provision_types,
                "legal_validation": self.validate_legal_schema()
            }
            
            return legal_info
            
        except Exception as e:
            self.logger.error(f"법령 스키마 정보 생성 중 오류: {e}")
            return {"error": str(e)}
    
    def create_legal_collection_schema(self, collection_name: str) -> Any:
        """법령 컬렉션 전용 스키마 생성"""
        try:
            description = f"Legal document collection: {collection_name}"
            schema = self.create_schema(collection_name, description)
            
            self.logger.info(f"법령 컬렉션 스키마 생성 완료: {collection_name}")
            return schema
            
        except Exception as e:
            self.logger.error(f"법령 컬렉션 스키마 생성 실패: {e}")
            raise
    
    def get_sample_legal_document(self) -> Dict[str, Any]:
        """법령 문서 샘플 데이터 (축소된 스키마용)"""
        return {
            # === Base 필드 (10개) ===
            "node_id": "legal_privacy_law_art1",
            "document_id": "privacy_law_2011",
            "hierarchy_level": 4,  # 조 레벨
            "parent_node_id": "legal_privacy_law_chap1",
            "hierarchy_path": "/개인정보보호법/제1장/제1조",
            "title": "목적",  # 조문 제목
            "content": "이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써 개인의 자유와 권리를 보호하고, 나아가 개인의 존엄과 가치를 구현하기 위함을 목적으로 한다.",
            "text_emb": None,  # 임베딩은 별도 생성
            "domain": "legal",
            "created_at": "2024-01-15T10:30:00",
            
            # === Legal 필드 (6개) ===
            "law_type": "법률",
            "law_name": "개인정보보호법",
            "law_number": "법률 제11690호",
            "article_number": "제1조",
            "paragraph_number": "",  # 단일 조문이므로 비어있음
            "item_number": "",       # 단일 조문이므로 비어있음
            "enactment_date": "2011.09.30",  # 제정일
        }
