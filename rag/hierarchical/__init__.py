"""
위계형 RAG 시스템

기존 RAG 시스템과 완전히 호환되면서 위계형 구조만 지원하는 시스템입니다.
기존 스키마에 최소한의 필드만 추가하여 법령의 조문 구조를 지원합니다.
"""

from .hierarchical_schema import HierarchicalSchema
from .hierarchical_retriever import HierarchicalRetriever
from .hierarchical_processor import HierarchicalProcessor

__version__ = "2.0.0"
__all__ = [
    'HierarchicalSchema',
    'HierarchicalRetriever',
    'HierarchicalProcessor'
]
