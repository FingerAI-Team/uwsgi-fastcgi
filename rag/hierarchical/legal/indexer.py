"""
법령 문서 인덱서 클래스

법령 파서와 베이스 인덱서를 결합하여 법령 문서를 위계형 구조로 인덱싱합니다.
"""

from typing import Dict, List, Any, Optional
import logging
import time
import hashlib
from datetime import datetime

from ..base.indexer import BaseHierarchicalIndexer
from .parser import LegalParser
from .schema import LegalSchema


class LegalIndexer(BaseHierarchicalIndexer):
    """법령 전용 인덱서 클래스"""
    
    def __init__(self, existing_interact_manager=None):
        """
        Args:
            existing_interact_manager: 기존 InteractManager 인스턴스 (배치/GPU 기능 재사용)
        """
        # 법령 전용 스키마와 파서 초기화
        legal_schema = LegalSchema()
        super().__init__(existing_interact_manager, legal_schema)
        
        self.legal_parser = LegalParser()
        self.logger = logging.getLogger(__name__)
        
        # 법령 인덱싱 설정
        self.node_id_prefix = "legal"
        self.enable_summary_generation = True
        self.enable_keyword_extraction = True
        
        self.logger.info("법령 인덱서 초기화 완료")
    
    def parse_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        법령 문서를 위계형 구조로 파싱
        
        Args:
            document: 원본 법령 문서
            
        Returns:
            List[Dict]: 파싱된 위계형 노드들
        """
        try:
            self.logger.info(f"법령 문서 파싱 시작: {document.get('title', 'Unknown')}")
            
            # 1. 법령 파서로 기본 파싱
            parsed_chunks = self.legal_parser.parse_legal_document(document)
            
            if not parsed_chunks:
                self.logger.warning("파싱된 청크가 없습니다")
                return []
            
            # 2. 위계형 노드로 변환
            hierarchical_nodes = self._convert_to_hierarchical_nodes(parsed_chunks, document)
            
            # 3. 노드 관계 설정
            linked_nodes = self._establish_node_relationships(hierarchical_nodes)
            
            self.logger.info(f"법령 파싱 완료: {len(linked_nodes)}개 노드")
            return linked_nodes
            
        except Exception as e:
            self.logger.error(f"법령 문서 파싱 중 오류: {e}")
            return []
    
    def _convert_to_hierarchical_nodes(self, chunks: List[Dict[str, Any]], 
                                     document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """청크를 위계형 노드로 변환"""
        try:
            nodes = []
            current_time = datetime.now().isoformat()
            document_id = self._generate_document_id(document)
            
            for chunk_idx, chunk in enumerate(chunks):
                node = {
                    # === Base 필드 (10개) ===
                    "node_id": self._generate_node_id(document_id, chunk_idx, chunk),
                    "document_id": document_id,
                    "hierarchy_level": chunk.get("hierarchy_level", 0),
                    "parent_node_id": "",  # 나중에 설정
                    "hierarchy_path": chunk.get("hierarchy_path", "/"),
                    "title": chunk.get("article_title", "") or chunk.get("section_number", "") or document.get("title", ""),
                    "content": chunk.get("text", ""),
                    "content_embedding": None,  # 나중에 생성
                    "domain": "legal",
                    "created_at": current_time,
                    
                    # === Legal 필드 (5개) ===
                    "law_type": chunk.get("law_type", ""),
                    "law_number": chunk.get("law_number", ""),
                    "article_number": chunk.get("article_number", ""),
                    "paragraph_number": chunk.get("paragraph_number", ""),
                    "item_number": chunk.get("item_number", ""),
                }
                
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"노드 변환 중 오류: {e}")
            return []
    
    def _generate_node_id(self, document_id: str, chunk_idx: int, chunk: Dict[str, Any]) -> str:
        """위계형 노드 ID 생성"""
        try:
            # 법령 특화 ID 생성
            law_number = chunk.get("law_number", "").replace(" ", "_")
            article_number = chunk.get("article_number", "").replace("제", "").replace("조", "")
            paragraph_number = chunk.get("paragraph_number", "")
            item_number = chunk.get("item_number", "")
            
            id_parts = [self.node_id_prefix]
            
            if law_number:
                id_parts.append(law_number)
            else:
                id_parts.append(document_id[:8])
            
            if article_number:
                id_parts.append(f"art_{article_number}")
                
            if paragraph_number:
                id_parts.append(f"para_{paragraph_number}")
                
            if item_number:
                id_parts.append(f"item_{item_number}")
            
            # 청크 인덱스로 고유성 보장
            id_parts.append(f"chunk_{chunk_idx}")
            
            return "_".join(id_parts)
            
        except Exception as e:
            self.logger.error(f"노드 ID 생성 중 오류: {e}")
            return f"{self.node_id_prefix}_{document_id[:8]}_{chunk_idx}"
    
    def _generate_document_id(self, document: Dict[str, Any]) -> str:
        """문서 ID 생성"""
        try:
            # 법령 번호가 있으면 사용
            law_number = document.get("law_number", "")
            if law_number:
                return law_number.replace(" ", "_").replace("제", "").replace("호", "")
            
            # 없으면 제목으로 해시 생성
            title = document.get("title", "")
            content_preview = document.get("text", "")[:100]
            combined = f"{title}_{content_preview}"
            
            return hashlib.blake2b(combined.encode('utf-8'), digest_size=16).hexdigest()
            
        except Exception as e:
            self.logger.error(f"문서 ID 생성 중 오류: {e}")
            return "unknown_document"
    
    def _generate_content_hash(self, content: str) -> str:
        """내용 해시 생성"""
        try:
            return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        except Exception as e:
            self.logger.error(f"내용 해시 생성 중 오류: {e}")
            return "unknown_hash"
    
    def _establish_node_relationships(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """노드 간 관계 설정"""
        try:
            # 레벨별로 노드 그룹화
            nodes_by_level = {}
            for node in nodes:
                level = node["hierarchy_level"]
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)
            
            # 부모-자식 관계 설정 (개선된 로직)
            for level in sorted(nodes_by_level.keys()):
                if level == 0:
                    continue  # 최상위 레벨은 부모가 없음
                
                current_level_nodes = nodes_by_level[level]
                
                # 모든 상위 레벨에서 부모 찾기 (중간 레벨 건너뛰기 대응)
                potential_parents = []
                for parent_level in range(level - 1, -1, -1):  # level-1부터 0까지 역순
                    if parent_level in nodes_by_level:
                        potential_parents.extend(nodes_by_level[parent_level])
                
                for node in current_level_nodes:
                    parent = self._find_parent_node(node, potential_parents)
                    if parent:
                        node["parent_node_id"] = parent["node_id"]
                        # child_count 계산 제거 (사용하지 않음)
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"노드 관계 설정 중 오류: {e}")
            return nodes
    
    def _find_parent_node(self, child_node: Dict[str, Any], 
                         potential_parents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """자식 노드의 부모 노드 찾기"""
        try:
            child_path = child_node.get("hierarchy_path", "")
            
            # 가장 구체적으로 매칭되는 부모 찾기
            best_parent = None
            max_match_length = 0
            
            for parent in potential_parents:
                parent_path = parent.get("hierarchy_path", "")
                
                # 자식 경로가 부모 경로로 시작하는지 확인
                if child_path.startswith(parent_path) and len(parent_path) > max_match_length:
                    best_parent = parent
                    max_match_length = len(parent_path)
            
            return best_parent
            
        except Exception as e:
            self.logger.error(f"부모 노드 검색 중 오류: {e}")
            return None
    
    # 제거됨: _calculate_descendant_counts (descendant_count 필드 삭제)
    
    # 제거됨: _calculate_tree_depths (tree_depth 필드 삭제)
    
    def _enrich_nodes(self, nodes: List[Dict[str, Any]], 
                     document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """노드 메타데이터 보강"""
        try:
            enriched_nodes = []
            
            for node in nodes:
                # 요약 생성 (옵션)
                if self.enable_summary_generation:
                    node["summary"] = self._generate_summary(node["content"])
                
                # 키워드 추출 (옵션)
                if self.enable_keyword_extraction:
                    node["keywords"] = self._extract_keywords(node["content"])
                
                enriched_nodes.append(node)
            
            return enriched_nodes
            
        except Exception as e:
            self.logger.error(f"노드 보강 중 오류: {e}")
            return nodes
    
    def _generate_summary(self, content: str) -> str:
        """내용 요약 생성 (간단한 구현)"""
        try:
            if len(content) <= 100:
                return content
            
            # 첫 번째 문장이나 처음 100자를 요약으로 사용
            sentences = content.split('.')
            if sentences and len(sentences[0]) <= 200:
                return sentences[0].strip() + '.'
            else:
                return content[:100] + "..."
                
        except Exception as e:
            self.logger.error(f"요약 생성 중 오류: {e}")
            return content[:50] + "..."
    
    def _extract_keywords(self, content: str) -> List[str]:
        """키워드 추출 (간단한 구현)"""
        try:
            # 기존 법령 파서의 키워드 추출 활용
            keywords = self.legal_parser._extract_legal_keywords(content)
            
            # 추가 키워드 (길이 기반)
            words = content.split()
            long_words = [word for word in words if len(word) >= 3 and word.isalpha()]
            
            # 중복 제거하고 상위 10개만
            all_keywords = list(set(keywords + long_words[:5]))
            return all_keywords[:10]
            
        except Exception as e:
            self.logger.error(f"키워드 추출 중 오류: {e}")
            return []
    
    def _generate_embeddings(self, nodes: List[Dict[str, Any]]) -> bool:
        """
        임베딩 생성 (제목과 내용 별도)
        
        Args:
            nodes: 임베딩을 생성할 노드들
            
        Returns:
            bool: 성공 여부
        """
        try:
            if not self.interact_manager or not hasattr(self.interact_manager, 'emb_model'):
                self.logger.error("임베딩 모델이 설정되지 않았습니다")
                return False
            
            # 제목과 내용 텍스트 분리
            titles = []
            contents = []
            valid_nodes = []
            
            for node in nodes:
                title = node.get("title", "").strip()
                content = node.get("content", "").strip()
                
                if title or content:
                    titles.append(title if title else content[:100])
                    contents.append(content if content else title)
                    valid_nodes.append(node)
            
            if not valid_nodes:
                self.logger.warning("임베딩할 텍스트가 없습니다")
                return False
            
            self.logger.info(f"임베딩 생성 시작: {len(valid_nodes)}개 노드")
            
            # 제목 임베딩 생성
            if titles:
                title_embeddings = self.interact_manager.emb_model.bge_batch_embed_data(titles)
                if title_embeddings and len(title_embeddings) == len(titles):
                    for node, embedding in zip(valid_nodes, title_embeddings):
                        node["title_embedding"] = embedding
                        
            # 내용 임베딩 생성
            if contents:
                content_embeddings = self.interact_manager.emb_model.bge_batch_embed_data(contents)
                if content_embeddings and len(content_embeddings) == len(contents):
                    for node, embedding in zip(valid_nodes, content_embeddings):
                        node["content_embedding"] = embedding
            
            self.logger.info("임베딩 생성 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"임베딩 생성 중 오류: {e}")
            return False
    
    def get_legal_indexing_stats(self, collection_name: str) -> Dict[str, Any]:
        """법령 인덱싱 통계 조회"""
        try:
            base_stats = self.get_indexing_stats(collection_name)
            
            if "error" in base_stats:
                return base_stats
            
            # 법령 특화 통계 추가
            legal_stats = {
                **base_stats,
                "legal_specific": {
                    "parser_stats": "법령 파싱 통계는 구현 예정",
                    "node_types": "노드 타입별 통계는 구현 예정",
                    "hierarchy_distribution": "위계 분포 통계는 구현 예정"
                }
            }
            
            return legal_stats
            
        except Exception as e:
            self.logger.error(f"법령 통계 조회 중 오류: {e}")
            return {"error": str(e)}
