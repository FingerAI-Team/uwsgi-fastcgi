"""
Meilisearch 클라이언트 및 BM25 검색 엔진

Vector 검색과 결합하여 하이브리드 검색을 제공합니다.
"""

import meilisearch
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class MeiliSearchConfig:
    """Meilisearch 설정"""
    host: str = "http://milvus-meilisearch:7700"
    master_key: str = "legal-search-hybrid-key-2024"
    index_name: str = "legal_documents"
    primary_key: str = "node_id"


class MeilisearchEngine:
    """Meilisearch BM25 검색 엔진"""
    
    def __init__(self, config: Optional[MeiliSearchConfig] = None):
        """
        Args:
            config: Meilisearch 설정
        """
        self.config = config or MeiliSearchConfig()
        self.logger = logging.getLogger(__name__)
        
        # 클라이언트 초기화
        self.client = None
        self.index = None
        self._initialize_client()
        
        # 검색 설정
        self.search_settings = {
            "limit": 100,
            "attributesToHighlight": ["content", "title", "law_name"],
            "highlightPreTag": "<mark>",
            "highlightPostTag": "</mark>",
            "attributesToCrop": ["content"],
            "cropLength": 200,
            "showMatchesPosition": True
        }
    
    def _initialize_client(self):
        """클라이언트 초기화"""
        try:
            self.client = meilisearch.Client(
                self.config.host, 
                self.config.master_key
            )
            
            # 연결 테스트
            health = self.client.health()
            self.logger.info(f"Meilisearch 연결 성공: {health}")
            
            # 인덱스 초기화
            self._initialize_index()
            
        except Exception as e:
            self.logger.error(f"Meilisearch 클라이언트 초기화 실패: {e}")
            raise
    
    def _initialize_index(self):
        """인덱스 초기화 및 설정"""
        try:
            # 인덱스 생성 또는 가져오기
            try:
                self.index = self.client.index(self.config.index_name)
                self.logger.info(f"기존 인덱스 사용: {self.config.index_name}")
            except:
                # 인덱스가 없으면 생성
                task = self.client.create_index(
                    self.config.index_name, 
                    {"primaryKey": self.config.primary_key}
                )
                self.client.wait_for_task(task.task_uid)
                self.index = self.client.index(self.config.index_name)
                self.logger.info(f"새 인덱스 생성: {self.config.index_name}")
            
            # 인덱스 설정
            self._configure_index()
            
        except Exception as e:
            self.logger.error(f"인덱스 초기화 실패: {e}")
            raise
    
    def _configure_index(self):
        """인덱스 설정 구성"""
        try:
            # 검색 가능한 속성 설정 (축소)
            searchable_attributes = [
                "content",          # 본문 (가장 중요)
                "title",           # 제목 (조문 제목)
                "law_name",        # 법령명 (새로 추가)
                "article_number",  # 조문 번호
            ]
            
            # 필터 가능한 속성 설정 (축소)
            filterable_attributes = [
                "law_type",
                "law_name",        # 법령명 필터링 추가
                "domain", 
                "hierarchy_level",
                "law_number",
                "article_number",
                "paragraph_number",
                "created_at"
            ]
            
            # 정렬 가능한 속성 설정
            sortable_attributes = [
                "hierarchy_level",
                "article_number", 
                "paragraph_number",
                "created_at"
            ]
            
            # 표시 가능한 속성 설정 (축소된 필드만)
            displayed_attributes = [
                "node_id",
                "content",
                "title", 
                "law_name",        # 법령명 표시 추가
                "article_number",
                "paragraph_number",
                "item_number",
                "law_type",
                "law_number",
                "domain",
                "hierarchy_level",
                "hierarchy_path",
                "created_at"
            ]
            
            # 설정 적용
            settings_tasks = []
            
            settings_tasks.append(
                self.index.update_searchable_attributes(searchable_attributes)
            )
            settings_tasks.append(
                self.index.update_filterable_attributes(filterable_attributes)
            )
            settings_tasks.append(
                self.index.update_sortable_attributes(sortable_attributes)
            )
            settings_tasks.append(
                self.index.update_displayed_attributes(displayed_attributes)
            )
            
            # 랭킹 규칙 설정 (BM25 최적화)
            ranking_rules = [
                "words",      # 일치하는 단어 수
                "typo",       # 오타 허용도
                "proximity",  # 단어 간 근접성
                "attribute",  # 속성 중요도
                "sort",       # 정렬
                "exactness"   # 정확도
            ]
            settings_tasks.append(
                self.index.update_ranking_rules(ranking_rules)
            )
            
            # 모든 설정 작업 완료 대기
            for task in settings_tasks:
                self.client.wait_for_task(task.task_uid)
            
            self.logger.info("Meilisearch 인덱스 설정 완료")
            
        except Exception as e:
            self.logger.error(f"인덱스 설정 실패: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """문서 추가"""
        try:
            if not documents:
                return True
            
            # 문서 형식 변환
            meili_documents = []
            for doc in documents:
                meili_doc = self._convert_to_meili_format(doc)
                if meili_doc:
                    meili_documents.append(meili_doc)
            
            if not meili_documents:
                self.logger.warning("변환된 문서가 없습니다")
                return False
            
            # 배치 추가
            self.logger.info(f"Meilisearch에 {len(meili_documents)}개 문서 추가 시작")
            task = self.index.add_documents(meili_documents)
            
            # 작업 완료 대기
            self.client.wait_for_task(task.task_uid)
            
            # 결과 확인
            task_info = self.client.get_task(task.task_uid)
            if task_info.status == "succeeded":
                self.logger.info(f"Meilisearch 문서 추가 성공: {len(meili_documents)}개")
                return True
            else:
                self.logger.error(f"Meilisearch 문서 추가 실패: {task_info}")
                return False
                
        except Exception as e:
            self.logger.error(f"Meilisearch 문서 추가 중 오류: {e}")
            return False
    
    def index_parsed_nodes(self, parsed_nodes: List[Dict[str, Any]], 
                          index_name: str = None, 
                          document_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        파싱된 위계형 노드들을 Meilisearch에 인덱싱
        
        Args:
            parsed_nodes: 파싱된 위계형 노드들
            index_name: 인덱스 이름 (기본값: 설정된 인덱스)
            document_metadata: 원본 문서 메타데이터
            
        Returns:
            Dict: 인덱싱 결과
        """
        try:
            if not parsed_nodes:
                return {
                    "status": "error",
                    "message": "파싱된 노드가 없습니다",
                    "indexed_count": 0
                }
            
            # 인덱스 이름 설정
            target_index = index_name or self.config.index_name
            
            # Meilisearch 형식으로 변환
            meili_documents = []
            for i, node in enumerate(parsed_nodes):
                # 노드에 메타데이터 추가
                enriched_node = node.copy()
                if document_metadata:
                    enriched_node.update({
                        "law_type": document_metadata.get("law_type", ""),
                        "law_name": document_metadata.get("law_name", ""),  # 법령명 추가
                        "law_number": document_metadata.get("law_number", ""),
                        "domain": document_metadata.get("domain", "legal"),
                        "created_at": datetime.now().isoformat()
                    })
                
                # Meilisearch 형식으로 변환
                meili_doc = self._convert_to_meili_format(enriched_node)
                if meili_doc:
                    meili_documents.append(meili_doc)
            
            if not meili_documents:
                return {
                    "status": "error", 
                    "message": "변환된 문서가 없습니다",
                    "indexed_count": 0
                }
            
            # 배치 추가
            self.logger.info(f"Meilisearch 인덱스 '{target_index}'에 {len(meili_documents)}개 노드 인덱싱 시작")
            task = self.index.add_documents(meili_documents)
            
            # 작업 완료 대기
            self.client.wait_for_task(task.task_uid)
            
            # 결과 확인
            task_info = self.client.get_task(task.task_uid)
            if task_info.status == "succeeded":
                self.logger.info(f"Meilisearch 인덱싱 성공: {len(meili_documents)}개 노드")
                return {
                    "status": "success",
                    "message": f"{len(meili_documents)}개 노드 인덱싱 완료",
                    "indexed_count": len(meili_documents),
                    "index_name": target_index,
                    "task_uid": task.task_uid
                }
            else:
                error_msg = f"Meilisearch 인덱싱 실패: {task_info}"
                self.logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "indexed_count": 0,
                    "task_uid": task.task_uid
                }
                
        except Exception as e:
            error_msg = f"Meilisearch 인덱싱 중 오류: {e}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "indexed_count": 0
            }
    
    def _convert_to_meili_format(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Milvus 노드를 Meilisearch 형식으로 변환"""
        try:
            meili_doc = {
                "id": node.get("node_id", ""),
                "document_id": node.get("document_id", ""),
                "hierarchy_level": node.get("hierarchy_level", 0),
                "parent_node_id": node.get("parent_node_id", ""),
                "hierarchy_path": node.get("hierarchy_path", ""),
                "title": node.get("title", ""),  # 조문 제목
                "content": node.get("content", ""),
                "domain": node.get("domain", ""),
                "created_at": node.get("created_at", ""),
                
                # 법령 특화 필드
                "law_type": node.get("law_type", ""),
                "law_name": node.get("law_name", ""),  # 법령명
                "law_number": node.get("law_number", ""),
                "article_number": node.get("article_number", ""),
                "paragraph_number": node.get("paragraph_number", ""),
                "item_number": node.get("item_number", ""),
                "enactment_date": node.get("enactment_date", ""),
            }
            
            return meili_doc
            
        except Exception as e:
            self.logger.error(f"Meilisearch 형식 변환 중 오류: {e}")
            return node
    
    def search(self, query: str, limit: int = 50, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """BM25 검색 수행"""
        try:
            if not query.strip():
                return []
            
            # 검색 옵션 구성
            search_options = {
                "limit": limit,
                **self.search_settings
            }
            
            # 필터 적용
            if filters:
                filter_expressions = []
                for key, values in filters.items():
                    if not values:  # 빈 값 스킵
                        continue
                    
                    # domains 배열을 다중 도메인 필터로 변환
                    if key == "domains" and isinstance(values, list) and len(values) > 0:
                        if len(values) == 1:
                            # 단일 도메인
                            domain_value = values[0]
                            filter_expressions.append(f"domain = '{domain_value}'")
                            self.logger.info(f"🌐 단일 도메인: {domain_value}")
                        else:
                            # 다중 도메인 - OR 조건으로 처리
                            domain_list = "', '".join(str(v) for v in values if v)
                            filter_expressions.append(f"domain IN ['{domain_list}']")
                            self.logger.info(f"🌐 다중 도메인 검색: {values}")
                    elif isinstance(values, list):
                        if len(values) == 1:
                            filter_expressions.append(f"{key} = '{values[0]}'")
                        else:
                            value_list = "', '".join(str(v) for v in values if v)  # 빈 값 제외
                            if value_list:
                                filter_expressions.append(f"{key} IN ['{value_list}']")
                    else:
                        filter_expressions.append(f"{key} = '{values}'")
                
                if filter_expressions:
                    search_options["filter"] = " AND ".join(filter_expressions)
                    self.logger.info(f"🔍 Meilisearch 필터: {search_options['filter']}")
            
            self.logger.debug(f"Meilisearch 검색: '{query}' with options: {search_options}")
            
            # 검색 실행
            start_time = time.time()
            results = self.index.search(query, search_options)
            search_time = time.time() - start_time
            
            # 결과 처리
            hits = results.get("hits", [])
            self.logger.info(f"Meilisearch BM25 검색 완료: {len(hits)}개 결과, {search_time:.3f}초")
            
            # 결과 형식 변환
            formatted_results = []
            for hit in hits:
                formatted_hit = self._format_search_result(hit, query)
                if formatted_hit:
                    formatted_results.append(formatted_hit)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Meilisearch 검색 중 오류: {e}")
            return []
    
    def _format_search_result(self, hit: Dict[str, Any], query: str) -> Dict[str, Any]:
        """검색 결과 형식 변환"""
        try:
            # 기본 점수 (0-1 범위로 정규화)
            raw_score = hit.get("_rankingScore", 0.0)
            normalized_score = min(raw_score / 100.0, 1.0) if raw_score > 0 else 0.0
            
            formatted_result = {
                # 식별 정보
                "node_id": hit.get("node_id"),
                "content": hit.get("content", ""),
                "title": hit.get("title", ""),
                
                # 검색 점수
                "bm25_score": normalized_score,
                "raw_bm25_score": raw_score,
                "search_strategy": "bm25",
                
                # 하이라이트 정보
                "highlights": hit.get("_formatted", {}),
                "matches_position": hit.get("_matchesPosition", {}),
                
                # 메타데이터
                "article_number": hit.get("article_number", ""),
                "paragraph_number": hit.get("paragraph_number", ""),  # VARCHAR 필드이므로 빈 문자열로 변경
                "hierarchy_level": hit.get("hierarchy_level", 0),
                "hierarchy_path": hit.get("hierarchy_path", ""),
                "law_type": hit.get("law_type", ""),
                "law_name": hit.get("law_name", ""),  # 누락된 필드 추가
                "domain": hit.get("domain", ""),
                
                # 검색 관련 정보
                "query": query,
                "search_timestamp": time.time()
            }
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"검색 결과 형식 변환 중 오류: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """인덱스 통계 조회"""
        try:
            stats = self.index.get_stats()
            return {
                "total_documents": stats.get("numberOfDocuments", 0),
                "index_size": stats.get("databaseSize", 0),
                "last_update": stats.get("updatedAt", ""),
                "is_indexing": stats.get("isIndexing", False)
            }
        except Exception as e:
            self.logger.error(f"통계 조회 중 오류: {e}")
            return {}
    
    def delete_all_documents(self) -> bool:
        """모든 문서 삭제"""
        try:
            task = self.index.delete_all_documents()
            self.client.wait_for_task(task.task_uid)
            self.logger.info("Meilisearch 모든 문서 삭제 완료")
            return True
        except Exception as e:
            self.logger.error(f"문서 삭제 중 오류: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        try:
            health = self.client.health()
            stats = self.get_stats()
            
            return {
                "status": "healthy" if health.get("status") == "available" else "unhealthy",
                "meilisearch_status": health,
                "index_stats": stats,
                "config": {
                    "host": self.config.host,
                    "index_name": self.config.index_name
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# 전역 인스턴스
_meilisearch_engine = None

def get_meilisearch_engine() -> MeilisearchEngine:
    """전역 Meilisearch 엔진 인스턴스 조회"""
    global _meilisearch_engine
    if _meilisearch_engine is None:
        _meilisearch_engine = MeilisearchEngine()
    return _meilisearch_engine
