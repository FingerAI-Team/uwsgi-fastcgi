"""
위계형 문서 인덱싱 베이스 클래스

기존 RAG 시스템의 배치 처리 및 GPU 관리 기능을 재사용하면서
위계형 구조 처리를 추가한 인덱서입니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging
import time
import uuid
import hashlib
from pymilvus import Collection, utility

# Meilisearch 클라이언트 임포트 (선택적)
try:
    from ..engines.meilisearch_client import MeilisearchClient
    MEILISEARCH_AVAILABLE = True
except ImportError:
    MEILISEARCH_AVAILABLE = False


class BaseHierarchicalIndexer(ABC):
    """위계형 문서 인덱싱 베이스 클래스"""
    
    def __init__(self, existing_interact_manager=None, schema_handler=None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager 인스턴스 (배치/GPU 기능 재사용)
            schema_handler: 스키마 핸들러
        """
        self.interact_manager = existing_interact_manager
        self.schema = schema_handler
        self.logger = logging.getLogger(__name__)
        
        # 위계형 처리 설정
        self.max_hierarchy_depth = 10  # 최대 위계 깊이
        
        # Meilisearch 클라이언트 초기화 (사용 가능한 경우)
        self.meilisearch_client = None
        if MEILISEARCH_AVAILABLE:
            try:
                self.meilisearch_client = MeilisearchClient()
                self.logger.info("Meilisearch 클라이언트 초기화 완료")
            except Exception as e:
                self.logger.warning(f"Meilisearch 클라이언트 초기화 실패: {e}")
                self.meilisearch_client = None
        
    def create_collection(self, collection_name: str, drop_existing: bool = False) -> bool:
        """
        위계형 컬렉션 생성
        
        기존 create_domain 기능을 확장하여 위계형 필드를 포함한 컬렉션 생성
        
        Args:
            collection_name: 생성할 컬렉션 이름
            drop_existing: 기존 컬렉션 삭제 여부
            
        Returns:
            bool: 생성 성공 여부
        """
        try:
            self.logger.info(f"위계형 컬렉션 생성 시작: {collection_name}")
            
            # 기존 컬렉션 확인 및 삭제
            if drop_existing and self._collection_exists(collection_name):
                self._drop_collection(collection_name)
                self.logger.info(f"기존 컬렉션 삭제: {collection_name}")
            
            # 스키마 생성
            if not self.schema:
                raise ValueError("스키마 핸들러가 설정되지 않았습니다")
                
            schema = self.schema.create_schema(
                collection_name, 
                f"Hierarchical collection for {collection_name}"
            )
            
            # 컬렉션 생성 (기존 방식 활용)
            if self.interact_manager and hasattr(self.interact_manager, 'vectorenv'):
                # 기존 create_domain과 유사한 방식
                collection = Collection(
                    name=collection_name,
                    schema=schema,
                    using='default',
                    shards_num=2
                )
                self.logger.info(f"컬렉션 생성 완료: {collection_name}")
            else:
                raise ValueError("InteractManager가 설정되지 않았습니다")
            
            # 인덱스 생성
            self._create_indexes(collection_name)
            
            # 컬렉션 로드
            collection.load()
            self.logger.info(f"컬렉션 로드 완료: {collection_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"컬렉션 생성 실패 ({collection_name}): {e}")
            return False
    
    def _collection_exists(self, collection_name: str) -> bool:
        """컬렉션 존재 여부 확인"""
        try:
            available_collections = utility.list_collections()
            return collection_name in available_collections
        except Exception as e:
            self.logger.error(f"컬렉션 존재 확인 실패: {e}")
            return False
    
    def _drop_collection(self, collection_name: str) -> bool:
        """컬렉션 삭제"""
        try:
            utility.drop_collection(collection_name)
            return True
        except Exception as e:
            self.logger.error(f"컬렉션 삭제 실패 ({collection_name}): {e}")
            return False
    
    def _create_indexes(self, collection_name: str) -> bool:
        """인덱스 생성"""
        try:
            if not self.schema:
                return False
                
            indexes = self.schema.get_all_indexes()
            collection = Collection(collection_name)
            
            for index_config in indexes:
                try:
                    collection.create_index(
                        field_name=index_config["field_name"],
                        index_params={
                            "index_type": index_config["index_type"],
                            "metric_type": index_config.get("metric_type", "COSINE"),
                            "params": index_config.get("params", {})
                        }
                    )
                    self.logger.info(f"인덱스 생성: {index_config['field_name']}")
                    
                except Exception as e:
                    self.logger.warning(f"인덱스 생성 실패 ({index_config['field_name']}): {e}")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"인덱스 생성 중 오류: {e}")
            return False
    
    @abstractmethod
    def parse_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        문서를 위계형 구조로 파싱 (서브클래스에서 구현)
        
        Args:
            document: 원본 문서 데이터
            
        Returns:
            List[Dict]: 파싱된 위계형 청크들
        """
        pass
    
    def index_document(self, collection_name: str, document: Dict[str, Any], 
                      ignore_duplicates: bool = True) -> bool:
        """
        단일 문서 인덱싱
        
        Args:
            collection_name: 컬렉션 이름
            document: 인덱싱할 문서
            ignore_duplicates: 중복 문서 무시 여부
            
        Returns:
            bool: 인덱싱 성공 여부
        """
        try:
            self.logger.info(f"문서 인덱싱 시작: {document.get('title', 'Untitled')}")
            
            # 1. 위계형 파싱
            parsed_chunks = self.parse_document(document)
            if not parsed_chunks:
                self.logger.warning("파싱된 청크가 없습니다")
                return False
            
            # 2. 필수 필드 보완
            processed_chunks = self._prepare_chunks_for_insertion(parsed_chunks, document)
            
            # 3. 임베딩 생성 (기존 시스템 활용)
            if not self._generate_embeddings(processed_chunks):
                self.logger.error("임베딩 생성 실패")
                return False
            
            # 4. 배치 삽입 (기존 시스템 활용)
            success = self._batch_insert_chunks(collection_name, processed_chunks)
            
            if success:
                self.logger.info(f"문서 인덱싱 완료: {len(processed_chunks)}개 청크")
            else:
                self.logger.error("배치 삽입 실패")
                
            return success
            
        except Exception as e:
            self.logger.error(f"문서 인덱싱 중 오류: {e}")
            return False
    
    def index_documents_batch(self, collection_name: str, documents: List[Dict[str, Any]], 
                             ignore_duplicates: bool = True) -> Dict[str, Any]:
        """
        다중 문서 배치 인덱싱
        
        Args:
            collection_name: 컬렉션 이름
            documents: 인덱싱할 문서들
            ignore_duplicates: 중복 문서 무시 여부
            
        Returns:
            Dict: 인덱싱 결과 통계
        """
        try:
            start_time = time.time()
            self.logger.info(f"배치 인덱싱 시작: {len(documents)}개 문서")
            
            total_chunks = 0
            successful_docs = 0
            failed_docs = 0
            
            all_chunks = []
            
            # 1. 모든 문서 파싱
            for doc_idx, document in enumerate(documents):
                try:
                    parsed_chunks = self.parse_document(document)
                    if parsed_chunks:
                        processed_chunks = self._prepare_chunks_for_insertion(parsed_chunks, document)
                        all_chunks.extend(processed_chunks)
                        total_chunks += len(processed_chunks)
                        successful_docs += 1
                    else:
                        failed_docs += 1
                        self.logger.warning(f"문서 파싱 실패: {doc_idx}")
                        
                except Exception as e:
                    failed_docs += 1
                    self.logger.error(f"문서 {doc_idx} 처리 실패: {e}")
            
            # 2. 배치 임베딩 생성
            if all_chunks:
                embedding_success = self._generate_embeddings(all_chunks)
                if not embedding_success:
                    self.logger.error("배치 임베딩 생성 실패")
                    return {"success": False, "error": "임베딩 생성 실패"}
                
                # 3. 배치 삽입
                insert_success = self._batch_insert_chunks(collection_name, all_chunks)
                if not insert_success:
                    self.logger.error("배치 삽입 실패")
                    return {"success": False, "error": "배치 삽입 실패"}
            
            end_time = time.time()
            duration = end_time - start_time
            
            result = {
                "success": True,
                "total_documents": len(documents),
                "successful_documents": successful_docs,
                "failed_documents": failed_docs,
                "total_chunks": total_chunks,
                "processing_time": duration,
                "chunks_per_second": total_chunks / duration if duration > 0 else 0
            }
            
            self.logger.info(f"배치 인덱싱 완료: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"배치 인덱싱 중 오류: {e}")
            return {"success": False, "error": str(e)}
    
    def _prepare_chunks_for_insertion(self, chunks: List[Dict[str, Any]], 
                                    original_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """청크를 삽입을 위해 준비 (필수 필드 보완)"""
        try:
            prepared_chunks = []
            
            for chunk_idx, chunk in enumerate(chunks):
                # 기본 필드 보완
                prepared_chunk = {
                    # 기존 시스템 호환 필드들
                    "passage_uid": self._generate_passage_uid(original_document, chunk_idx),
                    "doc_id": self._generate_doc_id(original_document),
                    "passage_id": chunk_idx + 1,
                    "domain": original_document.get("domain", "legal"),
                    "title": original_document.get("title", ""),
                    "author": original_document.get("author", ""),
                    "text": chunk.get("text", ""),
                    "info": original_document.get("info", {}),
                    "tags": original_document.get("tags", {}),
                    
                    # 위계형 필드들 (청크에서 가져오기)
                    "hierarchy_level": chunk.get("hierarchy_level", 0),
                    "parent_id": chunk.get("parent_id", ""),
                    "hierarchy_path": chunk.get("hierarchy_path", "/"),
                    "section_type": chunk.get("section_type", "content"),
                    "section_number": chunk.get("section_number", ""),
                    "hierarchy_metadata": chunk.get("metadata", {}),
                }
                
                # 청크의 다른 필드들도 병합
                for key, value in chunk.items():
                    if key not in prepared_chunk:
                        prepared_chunk[key] = value
                
                prepared_chunks.append(prepared_chunk)
            
            return prepared_chunks
            
        except Exception as e:
            self.logger.error(f"청크 준비 중 오류: {e}")
            return []
    
    def _generate_passage_uid(self, document: Dict[str, Any], chunk_idx: int) -> str:
        """고유 패시지 ID 생성"""
        doc_id = document.get("doc_id", str(uuid.uuid4()))
        return f"{doc_id}_{chunk_idx}"
    
    def _generate_doc_id(self, document: Dict[str, Any]) -> str:
        """문서 ID 생성 (기존 방식과 호환)"""
        if "doc_id" in document:
            return document["doc_id"]
        
        # 제목과 내용으로 해시 생성
        content = f"{document.get('title', '')}{document.get('text', '')}"
        return hashlib.blake2b(content.encode('utf-8'), digest_size=32).hexdigest()
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        임베딩 생성 (기존 시스템의 배치 처리 + GPU 세마포어 + 캐시 활용)
        
        Args:
            chunks: 임베딩을 생성할 청크들
            
        Returns:
            bool: 성공 여부
        """
        try:
            if not self.interact_manager or not hasattr(self.interact_manager, 'emb_model'):
                self.logger.error("임베딩 모델이 설정되지 않았습니다")
                return False
            
            # 텍스트 추출
            texts = [chunk.get("content", "") or chunk.get("text", "") for chunk in chunks]
            texts = [text for text in texts if text.strip()]  # 빈 텍스트 제거
            
            if not texts:
                self.logger.warning("임베딩할 텍스트가 없습니다")
                return False
            
            # 🔥 기존 배치 임베딩 시스템 완전 활용 
            # - GPU 세마포어 자동 적용
            # - LRU 캐시 (1000개) 자동 활용  
            # - 배치 크기 최적화 자동 적용
            self.logger.info(f"🚀 배치 임베딩 시작: {len(texts)}개 텍스트 (GPU 세마포어 + 캐시 활용)")
            embeddings = self.interact_manager.emb_model.bge_batch_embed_data(texts)
            
            if not embeddings or len(embeddings) != len(texts):
                self.logger.error(f"임베딩 생성 불일치: {len(embeddings)} vs {len(texts)}")
                return False
            
            # 임베딩 결과를 청크에 추가 (위계형 스키마에 맞게)
            for chunk, embedding in zip(chunks, embeddings):
                chunk["content_embedding"] = embedding  # 위계형 스키마 필드명
                # 하위 호환성을 위해 기존 필드도 유지
                chunk["text_emb"] = embedding
            
            self.logger.info(f"✅ 배치 임베딩 완료: {len(embeddings)}개 (캐시 히트율 자동 적용됨)")
            return True
            
        except Exception as e:
            self.logger.error(f"임베딩 생성 중 오류: {e}")
            return False
    
    def _batch_insert_chunks(self, collection_name: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        배치 삽입 (기존 시스템의 고급 배치 처리 완전 활용)
        
        기존 시스템의 다음 최적화 기능들이 자동으로 적용됩니다:
        - 대용량 배치 자동 분할 처리
        - 스레드 안전 배치 큐 관리  
        - 문서별 청크 그룹화
        - 개별 항목 재시도 메커니즘
        - 필수 필드 검증 및 자동 추가
        
        Args:
            collection_name: 컬렉션 이름
            chunks: 삽입할 청크들
            
        Returns:
            bool: 성공 여부
        """
        try:
            if not self.interact_manager:
                self.logger.error("InteractManager가 설정되지 않았습니다")
                return False
            
            # 🔥 기존 고급 배치 삽입 시스템 완전 활용
            # - 배치 크기 자동 조정 (BATCH_SIZE 환경변수)
            # - 글로벌 배치 큐 관리 (global_batch_queue)
            # - 배치 워커 스레드 활용
            # - 오류 발생 시 개별 재시도
            # - 메모리 최적화 처리
            self.logger.info(f"🚀 고급 배치 삽입 시작: {len(chunks)}개 청크 (최적화 적용)")
            success = self.interact_manager.batch_insert_data(collection_name, chunks)
            
            if success:
                self.logger.info(f"✅ 고급 배치 삽입 완료 (자동 분할, 재시도, 그룹화 적용됨)")
            else:
                self.logger.error(f"❌ 배치 삽입 실패")
                
            return success
            
        except Exception as e:
            self.logger.error(f"배치 삽입 중 오류: {e}")
            return False
    
    def get_indexing_stats(self, collection_name: str) -> Dict[str, Any]:
        """인덱싱 통계 조회"""
        try:
            if not self._collection_exists(collection_name):
                return {"error": f"컬렉션 {collection_name}이 존재하지 않습니다"}
            
            collection = Collection(collection_name)
            collection.load()
            
            stats = {
                "collection_name": collection_name,
                "total_entities": collection.num_entities,
                "is_empty": collection.is_empty,
                "schema_description": collection.description,
            }
            
            # 위계 레벨별 통계 (간단한 예시)
            try:
                # 실제로는 더 복잡한 쿼리가 필요할 수 있음
                stats["hierarchy_info"] = "위계 통계는 구현 예정"
            except Exception as e:
                stats["hierarchy_info"] = f"통계 조회 실패: {e}"
            
            return stats
            
        except Exception as e:
            self.logger.error(f"통계 조회 중 오류: {e}")
            return {"error": str(e)}
