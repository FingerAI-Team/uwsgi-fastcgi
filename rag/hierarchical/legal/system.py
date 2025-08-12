"""
통합 법령 RAG 시스템

법령 스키마, 인덱서, 검색기를 통합하여 API에서 쉽게 사용할 수 있는 인터페이스를 제공합니다.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import time
from datetime import datetime

from .schema import LegalSchema
from .parser import LegalParser
from .indexer import LegalIndexer
from .retriever import LegalRetriever


class LegalRAGSystem:
    """법령 전용 RAG 시스템 - API에서 간단하게 호출 가능"""
    
    def __init__(self, existing_interact_manager=None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager 인스턴스 (배치/GPU 기능 재사용)
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            # 구성요소 초기화
            self.schema = LegalSchema()
            self.parser = LegalParser()
            self.indexer = LegalIndexer(existing_interact_manager)
            self.retriever = LegalRetriever(existing_interact_manager)
            
            # 시스템 설정
            self.default_collection_prefix = "legal"
            self.system_info = {
                "version": "1.0.0",
                "initialized_at": datetime.now().isoformat(),
                "components": {
                    "schema": "LegalSchema",
                    "parser": "LegalParser", 
                    "indexer": "LegalIndexer",
                    "retriever": "LegalRetriever"
                }
            }
            
            self.logger.info("법령 RAG 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"법령 RAG 시스템 초기화 실패: {e}")
            raise
    
    # === 컬렉션 관리 ===
    
    def create_legal_collection(self, collection_name: str, 
                               drop_existing: bool = False) -> Dict[str, Any]:
        """
        법령 컬렉션 생성
        
        Args:
            collection_name: 생성할 컬렉션 이름
            drop_existing: 기존 컬렉션 삭제 여부
            
        Returns:
            Dict: 생성 결과
        """
        try:
            start_time = time.time()
            self.logger.info(f"법령 컬렉션 생성 시작: {collection_name}")
            
            # 컬렉션 이름 정규화
            normalized_name = self._normalize_collection_name(collection_name)
            
            # 컬렉션 생성
            success = self.indexer.create_collection(normalized_name, drop_existing)
            
            end_time = time.time()
            
            result = {
                "success": success,
                "collection_name": normalized_name,
                "original_name": collection_name,
                "drop_existing": drop_existing,
                "creation_time": end_time - start_time,
                "schema_info": self.schema.get_legal_schema_info(),
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                self.logger.info(f"법령 컬렉션 생성 완료: {normalized_name}")
            else:
                self.logger.error(f"법령 컬렉션 생성 실패: {normalized_name}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"법령 컬렉션 생성 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            normalized_name = self._normalize_collection_name(collection_name)
            stats = self.indexer.get_legal_indexing_stats(normalized_name)
            
            return {
                "success": True,
                "collection_name": normalized_name,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"컬렉션 정보 조회 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name,
                "timestamp": datetime.now().isoformat()
            }
    
    # === 문서 인덱싱 ===
    
    def index_legal_document(self, collection_name: str, document: Dict[str, Any],
                           ignore_duplicates: bool = True) -> Dict[str, Any]:
        """
        단일 법령 문서 인덱싱
        
        Args:
            collection_name: 컬렉션 이름
            document: 인덱싱할 법령 문서
            ignore_duplicates: 중복 문서 무시 여부
            
        Returns:
            Dict: 인덱싱 결과
        """
        try:
            start_time = time.time()
            self.logger.info(f"법령 문서 인덱싱 시작: {document.get('title', 'Unknown')}")
            
            normalized_name = self._normalize_collection_name(collection_name)
            
            # 문서 검증
            validation_result = self._validate_legal_document(document)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": f"문서 검증 실패: {validation_result['errors']}",
                    "document_title": document.get("title", "Unknown"),
                    "timestamp": datetime.now().isoformat()
                }
            
            # 인덱싱 실행
            success = self.indexer.index_document(
                normalized_name, document, ignore_duplicates
            )
            
            end_time = time.time()
            
            # 파싱 통계 생성
            parsing_stats = None
            try:
                parsed_chunks = self.parser.parse_legal_document(document)
                parsing_stats = self.parser.get_parsing_stats(parsed_chunks)
            except Exception as e:
                self.logger.warning(f"파싱 통계 생성 실패: {e}")
            
            result = {
                "success": success,
                "collection_name": normalized_name,
                "document_title": document.get("title", "Unknown"),
                "document_id": document.get("doc_id", ""),
                "law_number": document.get("law_number", ""),
                "processing_time": end_time - start_time,
                "parsing_stats": parsing_stats,
                "validation_result": validation_result,
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                self.logger.info(f"법령 문서 인덱싱 완료: {document.get('title', 'Unknown')}")
            else:
                self.logger.error(f"법령 문서 인덱싱 실패: {document.get('title', 'Unknown')}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"법령 문서 인덱싱 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_title": document.get("title", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }
    
    def index_legal_documents_batch(self, collection_name: str, documents: List[Dict[str, Any]],
                                   ignore_duplicates: bool = True) -> Dict[str, Any]:
        """
        다중 법령 문서 배치 인덱싱
        
        Args:
            collection_name: 컬렉션 이름
            documents: 인덱싱할 법령 문서들
            ignore_duplicates: 중복 문서 무시 여부
            
        Returns:
            Dict: 배치 인덱싱 결과
        """
        try:
            start_time = time.time()
            self.logger.info(f"법령 문서 배치 인덱싱 시작: {len(documents)}개 문서")
            
            normalized_name = self._normalize_collection_name(collection_name)
            
            # 문서들 검증
            validation_results = []
            valid_documents = []
            
            for doc in documents:
                validation = self._validate_legal_document(doc)
                validation_results.append(validation)
                if validation["valid"]:
                    valid_documents.append(doc)
            
            if not valid_documents:
                return {
                    "success": False,
                    "error": "유효한 문서가 없습니다",
                    "total_documents": len(documents),
                    "valid_documents": 0,
                    "validation_results": validation_results,
                    "timestamp": datetime.now().isoformat()
                }
            
            # 배치 인덱싱 실행
            batch_result = self.indexer.index_documents_batch(
                normalized_name, valid_documents, ignore_duplicates
            )
            
            end_time = time.time()
            
            result = {
                "success": batch_result.get("success", False),
                "collection_name": normalized_name,
                "total_documents": len(documents),
                "valid_documents": len(valid_documents),
                "invalid_documents": len(documents) - len(valid_documents),
                "processing_time": end_time - start_time,
                "batch_result": batch_result,
                "validation_summary": self._summarize_validations(validation_results),
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"법령 문서 배치 인덱싱 완료: {batch_result}")
            return result
            
        except Exception as e:
            self.logger.error(f"법령 문서 배치 인덱싱 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_documents": len(documents) if documents else 0,
                "timestamp": datetime.now().isoformat()
            }
    
    # === 문서 검색 ===
    
    def search_legal_documents(self, collection_name: str, query: str,
                              search_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        법령 문서 검색
        
        Args:
            collection_name: 검색할 컬렉션
            query: 검색 쿼리
            search_params: 검색 옵션
            
        Returns:
            Dict: 검색 결과
        """
        try:
            start_time = time.time()
            self.logger.info(f"법령 문서 검색 시작: {query}")
            
            normalized_name = self._normalize_collection_name(collection_name)
            
            # 기본 검색 파라미터 설정
            default_params = {
                "search_mode": "hybrid",
                "target_node_types": ["article", "paragraph", "item"],
                "include_context": True,
                "top_k": 10
            }
            
            if search_params:
                default_params.update(search_params)
            
            # 검색 실행
            search_results = self.retriever.search_legal_documents(
                normalized_name, query, default_params
            )
            
            end_time = time.time()
            
            result = {
                "success": True,
                "collection_name": normalized_name,
                "query": query,
                "search_params": default_params,
                "total_results": len(search_results),
                "results": search_results,
                "search_time": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"법령 문서 검색 완료: {len(search_results)}개 결과")
            return result
            
        except Exception as e:
            self.logger.error(f"법령 문서 검색 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "collection_name": collection_name,
                "timestamp": datetime.now().isoformat()
            }
    
    def search_by_article_number(self, collection_name: str, article_number: str) -> Dict[str, Any]:
        """조문 번호로 검색"""
        try:
            normalized_name = self._normalize_collection_name(collection_name)
            
            context = self.retriever.get_article_context(normalized_name, article_number)
            
            return {
                "success": True,
                "collection_name": normalized_name,
                "article_number": article_number,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"조문 번호 검색 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "article_number": article_number,
                "timestamp": datetime.now().isoformat()
            }
    
    # === 유틸리티 메서드들 ===
    
    def _normalize_collection_name(self, collection_name: str) -> str:
        """컬렉션 이름 정규화"""
        try:
            # 기본 접두사 추가
            if not collection_name.startswith(self.default_collection_prefix):
                return f"{self.default_collection_prefix}_{collection_name}"
            return collection_name
        except Exception:
            return f"{self.default_collection_prefix}_default"
    
    def _validate_legal_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """법령 문서 검증"""
        try:
            errors = []
            warnings = []
            
            # 필수 필드 확인
            required_fields = ["title", "text"]
            for field in required_fields:
                if not document.get(field):
                    errors.append(f"필수 필드 누락: {field}")
            
            # 텍스트 길이 확인
            text = document.get("text", "")
            if len(text) < 10:
                errors.append("텍스트가 너무 짧습니다 (최소 10자)")
            elif len(text) > 100000:
                warnings.append("텍스트가 매우 깁니다 (100,000자 초과)")
            
            # 법령 특화 필드 확인
            if not document.get("law_type"):
                warnings.append("법령 유형이 설정되지 않았습니다")
            
            if not document.get("law_number"):
                warnings.append("법령 번호가 설정되지 않았습니다")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "total_issues": len(errors) + len(warnings)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"검증 중 오류: {str(e)}"],
                "warnings": [],
                "total_issues": 1
            }
    
    def _summarize_validations(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """검증 결과 요약"""
        try:
            total = len(validation_results)
            valid = sum(1 for v in validation_results if v.get("valid", False))
            invalid = total - valid
            
            all_errors = []
            all_warnings = []
            
            for result in validation_results:
                all_errors.extend(result.get("errors", []))
                all_warnings.extend(result.get("warnings", []))
            
            return {
                "total_documents": total,
                "valid_documents": valid,
                "invalid_documents": invalid,
                "total_errors": len(all_errors),
                "total_warnings": len(all_warnings),
                "common_errors": self._get_common_issues(all_errors),
                "common_warnings": self._get_common_issues(all_warnings)
            }
            
        except Exception as e:
            return {"error": f"검증 요약 중 오류: {str(e)}"}
    
    def _get_common_issues(self, issues: List[str]) -> List[Dict[str, Any]]:
        """공통 이슈 분석"""
        try:
            issue_counts = {}
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            # 빈도순 정렬
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            
            return [{"issue": issue, "count": count} for issue, count in sorted_issues[:5]]
            
        except Exception:
            return []
    
    # === 시스템 정보 ===
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 조회"""
        try:
            return {
                "success": True,
                "system_info": self.system_info,
                "schema_info": self.schema.get_legal_schema_info(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_sample_legal_document(self) -> Dict[str, Any]:
        """샘플 법령 문서 반환"""
        try:
            sample = self.schema.get_sample_legal_document()
            return {
                "success": True,
                "sample_document": sample,
                "usage_note": "이 샘플을 참고하여 법령 문서를 구성하세요",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # === 고급 기능 ===
    
    def analyze_legal_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """법령 문서 분석 (인덱싱 없이 파싱만)"""
        try:
            start_time = time.time()
            
            # 문서 파싱
            parsed_chunks = self.parser.parse_legal_document(document)
            
            # 파싱 통계
            stats = self.parser.get_parsing_stats(parsed_chunks)
            
            end_time = time.time()
            
            return {
                "success": True,
                "document_title": document.get("title", "Unknown"),
                "total_chunks": len(parsed_chunks),
                "parsing_stats": stats,
                "analysis_time": end_time - start_time,
                "sample_chunks": parsed_chunks[:3],  # 샘플 청크 3개
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"법령 문서 분석 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_title": document.get("title", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }
