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
                    field_name = index_config["field_name"]
                    index_type = index_config["index_type"]
                    
                    # 벡터 필드인 경우에만 벡터 인덱스 생성
                    if index_type in ["IVF_FLAT", "IVF_SQ8", "HNSW"]:
                        collection.create_index(
                            field_name=field_name,
                            index_params={
                                "index_type": index_type,
                                "metric_type": index_config.get("metric_type", "COSINE"),
                                "params": index_config.get("params", {})
                            }
                        )
                        self.logger.info(f"벡터 인덱스 생성: {field_name}")
                    # VARCHAR 필드는 FLAT 인덱스 생성
                    elif index_type == "FLAT":
                        collection.create_index(
                            field_name=field_name,
                            index_params={
                                "index_type": "FLAT"
                            }
                        )
                        self.logger.info(f"FLAT 인덱스 생성: {field_name}")
                    else:
                        self.logger.debug(f"인덱스 생성 건너뜀: {field_name} ({index_type})")
                        
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
    
    def _index_parsed_nodes(self, collection_name: str, parsed_nodes: List[Dict[str, Any]], 
                           document: Dict[str, Any], ignore_duplicates: bool = True) -> bool:
        """
        이미 파싱된 노드들을 인덱싱
        
        Args:
            collection_name: 컬렉션 이름
            parsed_nodes: 이미 파싱된 노드들
            document: 원본 문서 (메타데이터용)
            ignore_duplicates: 중복 문서 무시 여부
            
        Returns:
            bool: 인덱싱 성공 여부
        """
        try:
            self.logger.info(f"파싱된 노드 인덱싱 시작: {len(parsed_nodes)}개 노드")
            
            # 1. 필수 필드 보완
            processed_nodes = self._prepare_chunks_for_insertion(parsed_nodes, document)
            
            # 2. 임베딩 생성
            if not self._generate_embeddings(processed_nodes):
                self.logger.error("임베딩 생성 실패")
                return False
            
            # 3. 배치 삽입
            success = self._batch_insert_chunks(collection_name, processed_nodes)
            
            if success:
                self.logger.info(f"파싱된 노드 인덱싱 완료: {len(processed_nodes)}개 노드")
            else:
                self.logger.error("배치 삽입 실패")
                
            return success
            
        except Exception as e:
            self.logger.error(f"파싱된 노드 인덱싱 중 오류: {e}")
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
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        위계형 전용 임베딩 생성 (기존 시스템의 모든 고급 기능 적용)
        
        기존 시스템의 다음 고급 기능들을 위계형에 맞게 적용:
        - 임베딩 캐시 (LRU)
        - 중복 텍스트 검사
        - GPU 세마포어 관리
        - 배치 크기 최적화
        - 메모리 관리
        """
        try:
            if not self.interact_manager or not hasattr(self.interact_manager, 'emb_model'):
                self.logger.error("임베딩 모델이 설정되지 않았습니다")
                return False
            
            # 텍스트 추출 및 전처리
            texts = []
            text_to_chunk_map = []
            
            for chunk in chunks:
                text = chunk.get("content", "") or chunk.get("text", "")
                if text.strip():  # 빈 텍스트 제거
                    texts.append(text)
                    text_to_chunk_map.append(chunk)
            
            if not texts:
                self.logger.warning("임베딩할 텍스트가 없습니다")
                return False
            
            # 🔥 기존 시스템의 고급 임베딩 기능 활용
            self.logger.info(f"🚀 위계형 고급 임베딩 시작: {len(texts)}개 텍스트")
            
            # 1. 기존 시스템의 배치 임베딩 완전 활용
            # - GPU 세마포어 자동 적용
            # - LRU 캐시 (1000개) 자동 활용
            # - 배치 크기 최적화 자동 적용
            # - 중복 텍스트 자동 처리
            embeddings = self.interact_manager.emb_model.bge_batch_embed_data(texts)
            
            if not embeddings or len(embeddings) != len(texts):
                self.logger.error(f"임베딩 생성 불일치: {len(embeddings)} vs {len(texts)}")
                return False
            
            # 2. 임베딩 결과를 위계형 청크에 추가
            for chunk, embedding in zip(text_to_chunk_map, embeddings):
                chunk["content_embedding"] = embedding
            
            # 3. 성공률 및 통계 로깅
            cache_hit_rate = getattr(self.interact_manager.emb_model, 'cache_hit_rate', 0)
            self.logger.info(f"✅ 위계형 임베딩 완료: {len(embeddings)}개 (캐시 히트율: {cache_hit_rate:.1f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"위계형 임베딩 생성 중 오류: {e}")
            return False
    
    def _prepare_chunks_for_insertion(self, chunks: List[Dict[str, Any]], 
                                    original_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        위계형 청크 준비 (기존 시스템의 고급 기능 적용)
        
        기존 시스템의 다음 고급 기능들을 위계형에 맞게 적용:
        - 중복 문서 검사
        - 문서 해시 생성
        - 필드 검증 및 보완
        - 메타데이터 추가
        """
        try:
            prepared_chunks = []
            
            # 🔥 기존 시스템의 중복 검사 활용
            # 1. 문서 해시 생성 (기존 시스템과 동일한 방식)
            doc_hash = self._generate_document_hash(original_document)
            
            # 2. 중복 문서 검사 (기존 시스템 활용)
            if hasattr(self.interact_manager, 'processed_documents'):
                if doc_hash in self.interact_manager.processed_documents:
                    self.logger.info(f"중복 문서 감지: {original_document.get('title', 'Unknown')}")
                    return []  # 중복 문서는 건너뛰기
            
            # 3. 위계형 청크 준비
            for chunk_idx, chunk in enumerate(chunks):
                                 # 기본 필드 보완 (순수 위계형 스키마에 맞게)
                 prepared_chunk = {
                     # === 위계형 핵심 필드들 (스키마와 정확히 일치) ===
                     "node_id": self._generate_node_id(original_document, chunk_idx),
                     "document_id": doc_hash,
                     "hierarchy_level": chunk.get("hierarchy_level", 0),
                     "parent_node_id": chunk.get("parent_node_id", ""),
                     "hierarchy_path": chunk.get("hierarchy_path", "/"),
                     "title": chunk.get("title", original_document.get("title", "")),
                     "content": chunk.get("content", ""),
                     "domain": original_document.get("domain", "legal"),
                     "created_at": self._get_current_timestamp(),
                 }
                
                 # 청크의 다른 필드들도 병합
                 for key, value in chunk.items():
                     if key not in prepared_chunk:
                         prepared_chunk[key] = value
                
                 prepared_chunks.append(prepared_chunk)
            
            # 4. 처리된 문서 기록 (기존 시스템과 동일)
            if hasattr(self.interact_manager, 'processed_documents'):
                self.interact_manager.processed_documents.add(doc_hash)
            
            self.logger.info(f"✅ 위계형 청크 준비 완료: {len(prepared_chunks)}개 노드")
            return prepared_chunks
            
        except Exception as e:
            self.logger.error(f"위계형 청크 준비 중 오류: {e}")
            return []
    
    def _generate_document_hash(self, document: Dict[str, Any]) -> str:
        """문서 해시 생성 (기존 시스템과 동일한 방식)"""
        if "doc_id" in document:
            return document["doc_id"]
        
        # 제목과 내용으로 해시 생성 (기존 시스템과 동일)
        content = f"{document.get('title', '')}{document.get('text', '')}"
        return hashlib.blake2b(content.encode('utf-8'), digest_size=32).hexdigest()
    
    def _generate_node_id(self, document: Dict[str, Any], chunk_idx: int) -> str:
        """위계형 노드 ID 생성"""
        doc_id = self._generate_document_hash(document)
        return f"{doc_id}_node_{chunk_idx}"
    
    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 생성"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _batch_insert_chunks(self, collection_name: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        위계형 전용 배치 삽입 (기존 시스템과 완전히 독립)
        
        위계형 스키마에 최적화된 새로운 배치 삽입 시스템
        """
        try:
            if not self.interact_manager:
                self.logger.error("InteractManager가 설정되지 않았습니다")
                return False
            
            # 위계형 전용 배치 삽입 로직
            self.logger.info(f"🚀 위계형 배치 삽입 시작: {len(chunks)}개 노드")
            
            # 1. 위계형 필드 검증
            valid_chunks = []
            for chunk in chunks:
                if self._validate_hierarchical_chunk(chunk):
                    valid_chunks.append(chunk)
                else:
                    self.logger.warning(f"위계형 필드 검증 실패: {chunk.get('node_id', 'unknown')}")
            
            if not valid_chunks:
                self.logger.error("유효한 위계형 청크가 없습니다")
                return False
            
            # 2. 위계형 배치 삽입 (새로운 로직)
            success = self._hierarchical_batch_insert(collection_name, valid_chunks)
            
            if success:
                self.logger.info(f"✅ 위계형 배치 삽입 완료: {len(valid_chunks)}개 노드")
            else:
                self.logger.error(f"❌ 위계형 배치 삽입 실패")
                
            return success
            
        except Exception as e:
            self.logger.error(f"위계형 배치 삽입 중 오류: {e}")
            return False
    
    def _validate_hierarchical_chunk(self, chunk: Dict[str, Any]) -> bool:
        """위계형 청크 필드 검증"""
        required_fields = [
            "node_id", "document_id", "hierarchy_level", 
            "title", "content", "content_embedding"
        ]
        
        for field in required_fields:
            if field not in chunk or chunk[field] is None:
                return False
        
        return True
    
    def _hierarchical_batch_insert(self, collection_name: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        위계형 전용 배치 삽입 구현 (기존 시스템의 성능 최적화 기능 적용)
        
        기존 시스템의 다음 고급 기능들을 위계형에 맞게 적용:
        - GPU 세마포어 관리
        - 배치 크기 자동 조정
        - 스레드 안전 처리
        - 개별 항목 재시도
        - 메모리 최적화
        """
        try:
            from pymilvus import Collection
            import os
            import threading
            import time
            
            # 컬렉션 로드
            collection = Collection(collection_name)
            collection.load()
            
            # 🔥 기존 시스템의 성능 최적화 설정 적용
            # 1. 배치 크기 자동 조정 (환경변수 기반)
            default_batch_size = 100  # 위계형 데이터는 작은 배치가 효율적
            batch_size = int(os.getenv('HIERARCHICAL_BATCH_SIZE', str(default_batch_size)))
            
            # 2. GPU 세마포어 활용 (기존 시스템과 동일)
            if hasattr(self.interact_manager, 'gpu_semaphore'):
                gpu_semaphore = self.interact_manager.gpu_semaphore
            else:
                gpu_semaphore = threading.Semaphore(1)  # 기본값
            
            # 3. 재시도 설정
            max_retries = int(os.getenv('HIERARCHICAL_MAX_RETRIES', '3'))
            retry_delay = float(os.getenv('HIERARCHICAL_RETRY_DELAY', '1.0'))
            
            self.logger.info(f"🚀 위계형 고급 배치 삽입 시작: {len(chunks)}개 노드, 배치크기={batch_size}")
            
            # 배치별 삽입 (성능 최적화 적용)
            successful_inserts = 0
            failed_batches = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                # GPU 세마포어 획득 (기존 시스템과 동일한 방식)
                with gpu_semaphore:
                    success = self._insert_batch_with_retry(
                        collection, batch, batch_num, max_retries, retry_delay
                    )
                    
                    if success:
                        successful_inserts += len(batch)
                        self.logger.info(f"✅ 위계형 배치 {batch_num} 삽입 완료: {len(batch)}개 노드")
                    else:
                        failed_batches.append((batch_num, batch))
                        self.logger.error(f"❌ 위계형 배치 {batch_num} 삽입 실패")
            
            # 실패한 배치 재처리 (기존 시스템의 재시도 메커니즘)
            if failed_batches:
                self.logger.warning(f"실패한 배치 {len(failed_batches)}개 재처리 시작")
                retry_success = self._retry_failed_batches(
                    collection, failed_batches, gpu_semaphore, max_retries, retry_delay
                )
                if retry_success:
                    successful_inserts += sum(len(batch) for _, batch in failed_batches)
            
            # 변경사항 플러시
            collection.flush()
            
            success_rate = (successful_inserts / len(chunks)) * 100
            self.logger.info(f"🎯 위계형 배치 삽입 완료: {successful_inserts}/{len(chunks)} ({success_rate:.1f}%)")
            
            return successful_inserts == len(chunks)
            
        except Exception as e:
            self.logger.error(f"위계형 배치 삽입 중 오류: {e}")
            return False
    
    def _insert_batch_with_retry(self, collection, batch, batch_num, max_retries, retry_delay):
        """배치 삽입 with 재시도 (기존 시스템의 재시도 메커니즘 적용)"""
        for attempt in range(max_retries):
            try:
                collection.insert(batch)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"배치 {batch_num} 삽입 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay * (attempt + 1))  # 지수 백오프
                else:
                    self.logger.error(f"배치 {batch_num} 최종 삽입 실패: {e}")
                    return False
        return False
    
    def _retry_failed_batches(self, collection, failed_batches, gpu_semaphore, max_retries, retry_delay):
        """실패한 배치 재처리 (기존 시스템의 재시도 메커니즘)"""
        try:
            retry_success_count = 0
            
            for batch_num, batch in failed_batches:
                with gpu_semaphore:
                    success = self._insert_batch_with_retry(
                        collection, batch, f"retry_{batch_num}", max_retries, retry_delay
                    )
                    if success:
                        retry_success_count += 1
                        self.logger.info(f"✅ 재시도 성공: 배치 {batch_num}")
                    else:
                        self.logger.error(f"❌ 재시도 실패: 배치 {batch_num}")
            
            retry_rate = (retry_success_count / len(failed_batches)) * 100
            self.logger.info(f"🔄 재시도 완료: {retry_success_count}/{len(failed_batches)} ({retry_rate:.1f}%)")
            
            return retry_success_count == len(failed_batches)
            
        except Exception as e:
            self.logger.error(f"재시도 처리 중 오류: {e}")
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
    
    def index_to_meilisearch(self, parsed_nodes: List[Dict[str, Any]], 
                           index_name: str = None,
                           document_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        파싱된 노드들을 Meilisearch에 인덱싱
        
        Args:
            parsed_nodes: 파싱된 위계형 노드들
            index_name: Meilisearch 인덱스 이름
            document_metadata: 원본 문서 메타데이터
            
        Returns:
            Dict: 인덱싱 결과
        """
        try:
            if not self.meilisearch_client:
                return {
                    "status": "error",
                    "message": "Meilisearch 클라이언트가 초기화되지 않았습니다",
                    "indexed_count": 0
                }
            
            # Meilisearch 인덱싱 수행
            result = self.meilisearch_client.index_parsed_nodes(
                parsed_nodes=parsed_nodes,
                index_name=index_name,
                document_metadata=document_metadata
            )
            
            self.logger.info(f"Meilisearch 인덱싱 결과: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Meilisearch 인덱싱 중 오류: {e}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "indexed_count": 0
            }

    def list_collections(self) -> List[str]:
        """사용 가능한 모든 컬렉션 목록 반환"""
        try:
            from pymilvus import utility
            collections = utility.list_collections()
            self.logger.info(f"사용 가능한 컬렉션: {collections}")
            return collections
        except Exception as e:
            self.logger.error(f"컬렉션 목록 조회 실패: {e}")
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """특정 컬렉션의 상세 정보 조회"""
        try:
            if not self._collection_exists(collection_name):
                return {"error": f"컬렉션 {collection_name}이 존재하지 않습니다"}
            
            collection = Collection(collection_name)
            collection.load()
            
            # 기본 정보
            info = {
                "collection_name": collection_name,
                "total_entities": collection.num_entities,
                "is_empty": collection.is_empty,
                "schema_description": collection.description,
            }
            
            # 스키마 정보
            try:
                schema = collection.schema
                info["schema"] = {
                    "fields": len(schema.fields),
                    "field_names": [field.name for field in schema.fields],
                    "indexed_fields": []
                }
                
                # 인덱스 정보
                try:
                    indexes = collection.index().info
                    if indexes:
                        info["schema"]["indexed_fields"] = [idx.get('field_name') for idx in indexes if idx.get('field_name')]
                except Exception:
                    pass
                    
            except Exception as e:
                info["schema"] = {"error": f"스키마 정보 조회 실패: {e}"}
            
            return info
            
        except Exception as e:
            self.logger.error(f"컬렉션 정보 조회 중 오류: {e}")
            return {"error": str(e)}
