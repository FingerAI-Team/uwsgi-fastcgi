"""
위계형 RAG 시스템 메인 모듈

이 모듈은 기존 RAG 시스템을 확장하여 위계형 문서 구조를 지원합니다.
법령, 정책문서, 매뉴얼 등 계층적 구조를 가진 문서들을 효율적으로 처리합니다.

주요 구성요소:
- base/: 위계형 시스템의 베이스 클래스들
- legal/: 법령 전용 구현체들  
- utils/: 공통 유틸리티 함수들
"""

from .base import BaseHierarchicalSchema, BaseHierarchicalIndexer, BaseHierarchicalRetriever
from .legal import LegalSchema, LegalParser, LegalIndexer, LegalRetriever, LegalRAGSystem

__version__ = "1.0.0"
__all__ = [
    'BaseHierarchicalSchema',
    'BaseHierarchicalIndexer', 
    'BaseHierarchicalRetriever',
    'LegalSchema',
    'LegalParser',
    'LegalIndexer',
    'LegalRetriever',
    'LegalRAGSystem'
]
