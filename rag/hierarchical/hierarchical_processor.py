"""
위계형 데이터 프로세서

기존 pipe.py의 InteractManager를 확장하여 조항 단위 청킹과 위계 필드를 지원합니다.
기존의 모든 배치 처리, GPU 관리, 데이터 파이프라인을 그대로 활용합니다.
"""

import re
import logging
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from src.pipe import InteractManager


class HierarchicalProcessor(InteractManager):
    """위계형 데이터 프로세서 - 기존 InteractManager 확장"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger('hierarchical')
        
        # 조항 패턴 정의 (장 포함 + 생략 패턴)
        self.article_patterns = {
            "chapter": r"제(\d+)장",      # 제1장, 제2장 등
            "main_article": r"제(\d+)조",
            "sub_article": r"제(\d+)조의(\d+)",
            "paragraph": r"(\d+)\.",  # 1., 2., 3. 등
            "item": r"(\d+)\)",       # 1), 2), 3) 등
            "omission": r"(?:제\d+조부터\s+제\d+조까지는\s+생략한다?|이하\s+생략|생략한다?|\.\.\.)",  # 생략 패턴
        }
        
        # 위계형 스키마 초기화
        from .hierarchical_schema import HierarchicalSchema
        self.hierarchical_schema = HierarchicalSchema()
        

        
        self.logger.info("✅ 위계형 프로세서 초기화 완료")
    
    def _get_collection_instance(self, collection_name):
        """컬렉션 인스턴스를 가져오는 헬퍼 메서드"""
        if hasattr(self, 'vectorenv') and self.vectorenv:
            if hasattr(self.vectorenv, 'get_collection'):
                self.logger.info(f"📚 DataMilVus.get_collection 사용: {collection_name}")
                return self.vectorenv.get_collection(collection_name)
            else:
                self.logger.info(f"📚 MilvusEnvManager - 직접 Collection 생성: {collection_name}")
                from pymilvus import Collection
                collection = Collection(collection_name)
                collection.load()
                return collection
        else:
            self.logger.info(f"📚 InteractManager get_collection 사용 (폴백): {collection_name}")
            return self.get_collection(collection_name)
    
    def chunk_by_articles(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        라인 시작 기반 위계형 청킹 (장/절/관/조/항/호/목 단위)
        - 조: 제10조, 제10조의2, (제목) 지원
        - 항: ①, (1), 1. 지원  
        - 호: 1), (1), 가), (가), 가. 지원
        - 목: 가., 나., 다., 라. 지원
        - 생략/삭제: '…생략', '[삭제]' 태깅
        - 개정/신설: '[전문개정]', '[신설]' 태깅
        - 부칙: 부칙<제112호> 형태 지원
        """
        import re

        try:
            self.logger.info(f"🔧 위계형 청킹 시작: {len(text)}자")

            # -------------------------
            # 0) 정규식/정규화 설정
            # -------------------------
            FLAGS = re.UNICODE | re.MULTILINE

            # 동그라미 숫자 → (n) 통일
            CIRCLED_MAP = {chr(0x2460 + i): f"({i+1})" for i in range(20)}  # ①~⑳
            def normalize_line(s: str) -> str:
                s = s.replace("\t", " ").replace("\r", "")
                for k, v in CIRCLED_MAP.items():
                    s = s.replace(k, v)
                return s.strip()

            # 페이지 번호 등 단독 숫자 라인 제거
            PAT_PAGE_NO = re.compile(r"^\s*[-–—]*\s*\d+\s*[-–—]*\s*$", FLAGS)

            # 상·중·하위 구조
            PAT_CHAPTER = re.compile(r"^\s*제(?P<num>\d+)장\s*(?:\((?P<title>[^)]+)\))?\s*$", FLAGS)
            PAT_SECTION = re.compile(r"^\s*제(?P<num>\d+)절\s*(?:\((?P<title>[^)]+)\))?\s*$", FLAGS)
            PAT_DIVISION = re.compile(r"^\s*제(?P<num>\d+)관\s*(?:\((?P<title>[^)]+)\))?\s*$", FLAGS)

            # 조(의조+제목) - 의조 패턴 강화
            PAT_ARTICLE = re.compile(
                r"^\s*제(?P<num>\d+)조(?:의(?P<sub>\d+))?\s*(?:\((?P<title>[^)]+)\))?\s*$", FLAGS
            )

            # 항: (1) / 1.    (동그라미 숫자는 normalize에서 (n)으로 변환됨)
            PAT_PARAGRAPH = re.compile(r"^\s*(?:\((?P<p>\d+)\)|(?P<p2>\d+)\.)\s+", FLAGS)

            # 호: 1. / 1) / (1) / 가. / (가)
            PAT_SUBPARA = re.compile(
                r"^\s*(?:(?P<n1>\d+)[\.\)]|\((?P<n2>\d+)\)|(?P<ko1>[가-힣])[\.\)]|\((?P<ko2>[가-힣])\))\s+",
                FLAGS
            )

            # 목: 가., 나., 다., 라. (호보다 더 하위)
            PAT_ITEM = re.compile(
                r"^\s*(?:(?P<ko>[가-힣])\.|\((?P<ko2>[가-힣])\))\s+", FLAGS
            )

            # 생략/삭제/개정/신설
            PAT_OMISSION = re.compile(
                r"(?:제\d+조부터\s+제\d+조까지(?:는)?\s*생략한다?|이하\s*생략|생략한다?|\[\s*삭제\s*[^\]]*\]|삭제)\s*$",
                FLAGS
            )
            
            PAT_AMENDMENT = re.compile(
                r"\[(?:전문개정|개정|신설|삭제)\s+[^\]]+\]", FLAGS
            )

            # 부칙 패턴 - 한자 포함 및 패턴 확장
            PAT_APPENDIX = re.compile(
                r"^(?:부칙|附則)\s*(?:\([^)]+\))?\s*(?:<[^>]+>)?", FLAGS
            )

            # 별지 패턴
            PAT_ATTACHMENT = re.compile(
                r"^\[별지\s+제\s*\d+\s*호[^\]]*\]", FLAGS
            )

            # 부칙 내부 조항 패턴 (부칙 내에서만 사용)
            PAT_APPENDIX_ARTICLE = re.compile(
                r"^\s*제(?P<num>\d+)조\s*(?:\((?P<title>[^)]+)\))?\s*$", FLAGS
            )

            # -------------------------
            # 1) 유틸: 플러시/메타 복제
            # -------------------------
            chunks: List[Tuple[str, Dict[str, Any]]] = []
            buf: List[str] = []

            meta = {
                "chapter_number": "",
                "chapter_title": "",
                "section_number": "",
                "section_title": "",
                "division_number": "",
                "division_title": "",
                "article_number": "",
                "article_title": "",
                "paragraph_number": "",
                "subparagraph_number": "",
                "item_number": "",
                "is_omission": False,
                "is_deletion": False,
                "is_amendment": False,
                "is_appendix": False,
                "is_attachment": False,
                "appendix_type": "header",  # header: 헤더, main: 메인컨텐츠, 부칙: 부칙, 별지: 별지
            }

            def flush(reason: str):
                # 버퍼 내용을 하나의 청크로 저장
                txt = "\n".join([x for x in buf]).strip()
                if txt:
                    chunks.append((txt, meta.copy()))
                buf.clear()

            # 하위 위계 초기화 헬퍼
            def reset_below(level: str):
                """level 이하 하위 위계를 초기화"""
                order = ["chapter", "section", "division", "article", "paragraph", "subparagraph", "item"]
                idx = order.index(level)
                for lv in order[idx+1:]:
                    if lv == "chapter":
                        meta["chapter_number"] = ""; meta["chapter_title"] = ""
                    elif lv == "section":
                        meta["section_number"] = ""; meta["section_title"] = ""
                    elif lv == "division":
                        meta["division_number"] = ""; meta["division_title"] = ""
                    elif lv == "article":
                        meta["article_number"] = ""; meta["article_title"] = ""
                    elif lv == "paragraph":
                        meta["paragraph_number"] = ""
                    elif lv == "subparagraph":
                        meta["subparagraph_number"] = ""
                    elif lv == "item":
                        meta["item_number"] = ""
                
                # 상태 플래그도 초기화
                meta["is_omission"] = False
                meta["is_deletion"] = False
                meta["is_amendment"] = False
                meta["is_appendix"] = False
                meta["is_attachment"] = False
                meta["appendix_type"] = "main"  # 메인컨텐츠로 초기화

            # -------------------------
            # 2) 본문 라인 순회
            # -------------------------
            lines = [normalize_line(x) for x in text.split("\n") if x.strip() and not PAT_PAGE_NO.match(x)]
            main_content_started = False  # 메인컨텐츠 시작 여부
            
            for line in lines:
                # 메인컨텐츠 시작점 판단 (첫 번째 위계 구조)
                if not main_content_started and any([
                    PAT_CHAPTER.match(line),      # 장
                    PAT_SECTION.match(line),      # 절  
                    PAT_DIVISION.match(line),     # 관
                    PAT_ARTICLE.match(line)       # 조
                ]):
                    # 메인컨텐츠 시작! 헤더 청크 완성
                    flush("header_complete")
                    main_content_started = True
                    meta["appendix_type"] = "main"  # 메인컨텐츠로 변경
                
                # 2-1) 부칙/별지 먼저 체크 (상위 구조)
                if PAT_APPENDIX.match(line):
                    flush("new_appendix")
                    meta["is_appendix"] = True
                    # 부칙 정보 추가
                    meta["appendix_type"] = "부칙"
                    # 🚨 중요: 메인 법령 정보는 유지 (reset_below 호출하지 않음)
                    # 부칙은 메인 법령의 파생물이므로 연결성 보존
                    buf.append(line)
                    continue
                    
                if PAT_ATTACHMENT.match(line):
                    flush("new_attachment")
                    meta["is_attachment"] = True
                    meta["appendix_type"] = "별지"
                    # 별지도 메인 법령과 연결
                    buf.append(line)
                    continue

                # 2-2) 생략/삭제/개정 체크
                if PAT_OMISSION.search(line):
                    # 현재 버퍼 flush
                    flush("before_omission")
                    # 생략/삭제 라인 자체도 하나의 청크로 기록
                    buf.append(line)
                    meta["is_omission"] = True
                    if "삭제" in line:
                        meta["is_deletion"] = True
                    flush("omission_line")
                    # 생략 태그는 라인별 의미이므로 다시 false로 세팅
                    meta["is_omission"] = False
                    meta["is_deletion"] = False
                    continue

                # 2-3) 개정/신설 체크
                if PAT_AMENDMENT.search(line):
                    # 현재 버퍼 flush
                    flush("before_amendment")
                    # 개정 라인 자체도 하나의 청크로 기록
                    buf.append(line)
                    meta["is_amendment"] = True
                    flush("amendment_line")
                    # 개정 태그는 라인별 의미이므로 다시 false로 세팅
                    meta["is_amendment"] = False
                    continue

                # 2-4) 상위 구조: 장/절/관
                m = PAT_CHAPTER.match(line)
                if m:
                    buf.append(line)  # 🚨 중요: 장 라인을 먼저 버퍼에 추가
                    meta["chapter_number"] = f"제{m.group('num')}장"
                    meta["chapter_title"] = (m.group('title') or "").strip()
                    # 장이 바뀌면 하위 초기화
                    reset_below("chapter")
                    flush("new_chapter")  # 🚨 그 다음에 flush
                    continue

                m = PAT_SECTION.match(line)
                if m:
                    buf.append(line)  # 🚨 중요: 절 라인을 먼저 버퍼에 추가
                    meta["section_number"] = f"제{m.group('num')}절"
                    meta["section_title"] = (m.group('title') or "").strip()
                    reset_below("section")
                    flush("new_section")  # 🚨 그 다음에 flush
                    continue

                m = PAT_DIVISION.match(line)
                if m:
                    buf.append(line)  # 🚨 중요: 관 라인을 먼저 버퍼에 추가
                    meta["division_number"] = f"제{m.group('num')}관"
                    meta["division_title"] = (m.group('title') or "").strip()
                    reset_below("division")
                    flush("new_division")  # 🚨 그 다음에 flush
                    continue

                # 2-5) 조(의조 포함) - 부칙 내부 조항도 처리
                m = PAT_ARTICLE.match(line)
                if m:
                    buf.append(line)  # 🚨 중요: 조 라인을 먼저 버퍼에 추가
                    num = m.group("num")
                    sub = m.group("sub")
                    meta["article_number"] = f"제{num}조" + (f"의{sub}" if sub else "")
                    meta["article_title"] = (m.group("title") or "").strip()
                    # 조가 바뀌면 하위 초기화
                    reset_below("article")
                    flush("new_article")  # 🚨 그 다음에 flush
                    continue

                # 부칙 내부 조항 패턴 체크 (부칙 내에서만)
                m = PAT_APPENDIX_ARTICLE.match(line)
                if meta.get("is_appendix") and m:
                    buf.append(line)  # 🚨 중요: 조 라인을 먼저 버퍼에 추가
                    num = m.group("num")
                    title = m.group("title") or ""
                    meta["article_number"] = f"제{num}조"
                    meta["article_title"] = title.strip()
                    # 부칙 내 조항이므로 하위 초기화
                    reset_below("article")
                    flush("new_appendix_article")  # 🚨 그 다음에 flush
                    continue

                # 2-6) 항
                m = PAT_PARAGRAPH.match(line)
                if m:
                    # 새 항이 시작되면 이전 버퍼를 항 단위로 flush
                    flush("new_paragraph")
                    pnum = m.group("p") or m.group("p2")
                    meta["paragraph_number"] = pnum
                    # 항 하위 초기화
                    meta["subparagraph_number"] = ""
                    meta["item_number"] = ""
                    # 항 마커 제거 후 본문만 버퍼에 넣음
                    line_wo_marker = PAT_PARAGRAPH.sub("", line, count=1).strip()
                    buf.append(line_wo_marker)  # 🚨 항 본문을 버퍼에 추가
                    continue

                # 2-7) 호
                m = PAT_SUBPARA.match(line)
                if m:
                    flush("new_subpara")
                    if m.group("n1") or m.group("n2"):
                        sp = m.group("n1") or m.group("n2")  # 숫자 호
                    else:
                        sp = (m.group("ko1") or m.group("ko2") or "").strip()  # 한글 호(가, 나…)
                    meta["subparagraph_number"] = sp
                    # 호 하위 초기화
                    meta["item_number"] = ""
                    line_wo_marker = PAT_SUBPARA.sub("", line, count=1).strip()
                    buf.append(line_wo_marker)  # 🚨 호 본문을 버퍼에 추가
                    continue

                # 2-8) 목 (호보다 더 하위)
                m = PAT_ITEM.match(line)
                if m:
                    flush("new_item")
                    item = (m.group("ko") or m.group("ko2") or "").strip()
                    meta["item_number"] = item
                    line_wo_marker = PAT_ITEM.sub("", line, count=1).strip()
                    buf.append(line_wo_marker)  # 🚨 목 본문을 버퍼에 추가
                    continue

                # 2-9) 그 외 일반 본문 라인
                buf.append(line)

            # 3) 마지막 청크 플러시
            flush("eof")
            self.logger.info(f"✅ 위계형 청킹 완료: {len(chunks)}개 청크")
            return chunks

        except Exception as e:
            self.logger.error(f"조항 단위 청킹 중 오류: {e}")
            return self._fallback_chunking(text)
    
    def _fallback_chunking(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """기존 청킹 방식으로 폴백"""
        try:
            # 기존 data_p의 chunk_text 사용
            chunked_texts = self.data_p.chunk_text(text)
            
            # 위계 정보 없이 반환 (장 필드 포함)
            chunks = []
            for chunk in chunked_texts:
                chunks.append((chunk, {
                    "chapter_number": "",
                    "article_number": "",
                    "paragraph_number": "",
                    "item_number": "",
                    "is_omission": False
                }))
            
            self.logger.info(f"폴백 청킹 완료: {len(chunks)}개 청크")
            return chunks
            
        except Exception as e:
            self.logger.error(f"폴백 청킹 중 오류: {e}")
            return [(text, {
                "chapter_number": "",
                "article_number": "",
                "paragraph_number": "",
                "item_number": "",
                "is_omission": False
            })]
    
    def _load_new_collection(self, collection_name):
        """위계형 스키마로 새 컬렉션을 로드하고 캐시에 추가합니다."""
        try:
            print(f"[DEBUG] Loading new collection: {collection_name}")
            
            # 캐시 크기 제한 확인 및 관리
            if len(self.loaded_collections) >= self.max_cached_collections:
                # 가장 적게 접근된 컬렉션 찾기
                least_used = min(
                    self.collection_access_count.items(), 
                    key=lambda x: x[1] if x[0] in self.loaded_collections else float('inf')
                )[0]
                
                if least_used in self.loaded_collections:
                    print(f"[DEBUG] Removing least used collection from cache: {least_used}")
                    del self.loaded_collections[least_used]
            
            # 컬렉션 존재 여부 확인
            from pymilvus import utility
            if not utility.has_collection(collection_name):
                print(f"[DEBUG] Collection {collection_name} does not exist, creating with hierarchical schema")
                self._create_hierarchical_collection(collection_name)
            
            # 새 컬렉션 로드
            from pymilvus import Collection
            collection = Collection(collection_name)
            collection.load()
            
            # 캐시에 추가
            self.loaded_collections[collection_name] = collection
            print(f"[DEBUG] Collection {collection_name} successfully loaded and cached")
            return collection
            
        except Exception as e:
            print(f"[ERROR] Error loading collection {collection_name}: {str(e)}")
            raise
    
    def _create_hierarchical_collection(self, collection_name):
        """위계형 스키마로 컬렉션을 생성합니다."""
        try:
            self.logger.info(f"🔧 위계형 컬렉션 생성: {collection_name}")
            
            # 위계형 스키마 필드 가져오기
            schema_fields = self.hierarchical_schema.get_compatible_fields()
            
            # 스키마 생성
            from pymilvus import CollectionSchema
            schema = CollectionSchema(
                fields=schema_fields,
                description=f"위계형 RAG 스키마 - {collection_name}",
                enable_dynamic_field=True
            )
            
            # 컬렉션 생성
            from pymilvus import Collection
            collection = Collection(
                name=collection_name,
                schema=schema,
                using='default',
                shards_num=2
            )
            
            # 벡터 인덱스 생성
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(
                field_name="text_emb",
                index_params=index_params
            )
            
            self.logger.info(f"✅ 위계형 컬렉션 생성 완료: {collection_name}")
            return collection
            
        except Exception as e:
            self.logger.error(f"위계형 컬렉션 생성 중 오류: {e}")
            raise
    
    def list_collections(self):
        """사용 가능한 컬렉션 목록을 반환합니다."""
        try:
            from pymilvus import utility
            collections = utility.list_collections()
            return collections
        except Exception as e:
            self.logger.error(f"컬렉션 목록 조회 중 오류: {e}")
            return []
    
    def get_collection_info(self, collection_name):
        """특정 컬렉션의 정보를 반환합니다."""
        self.logger.info(f"🔍 컬렉션 정보 조회 시작: {collection_name}")
        start_time = time.time()
        
        try:
            # 이미 연결된 Milvus 인스턴스 사용 (vectorenv)
            self.logger.info(f"📚 기존 Milvus 연결 사용: {collection_name}")
            
            # 헬퍼 메서드 사용하여 컬렉션 인스턴스 가져오기
            collection = self._get_collection_instance(collection_name)
            
            # 스키마 정보 수집
            self.logger.info(f"📋 컬렉션 스키마 정보 수집: {collection_name}")
            schema_fields = []
            for field in collection.schema.fields:
                schema_fields.append({
                    "name": field.name,
                    "type": str(field.dtype),
                    "description": field.description
                })
            
            # 인덱스 정보 수집
            self.logger.info(f"🔍 컬렉션 인덱스 정보 수집: {collection_name}")
            indexes = []
            for idx in collection.indexes:
                index_info = {
                    "name": idx.name if hasattr(idx, 'name') else str(idx),
                    "type": str(type(idx).__name__),
                    "params": getattr(idx, 'params', {})
                }
                indexes.append(index_info)
            
            # 파티션 정보 수집
            self.logger.info(f"📦 컬렉션 파티션 정보 수집: {collection_name}")
            partitions = [p.name for p in collection.partitions]
            
            # 엔티티 수 조회
            self.logger.info(f"📊 컬렉션 엔티티 수 조회: {collection_name}")
            num_entities = collection.num_entities
            
            info = {
                "name": collection.name,
                "num_entities": num_entities,
                "schema": {
                    "fields": schema_fields
                },
                "indexes": indexes,
                "partitions": partitions
            }
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.info(f"✅ 컬렉션 정보 조회 성공: {collection_name}")
            self.logger.info(f"📊 컬렉션 정보 - 엔티티 수: {info.get('num_entities', 'N/A')}, 필드 수: {len(info.get('schema', {}).get('fields', []))}, 파티션 수: {len(info.get('partitions', []))}")
            self.logger.info(f"⏱️ 처리 시간: {processing_time:.3f}초")
            
            return info
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.error(f"❌ 컬렉션 정보 조회 실패: {collection_name}")
            self.logger.error(f"🚨 오류 내용: {str(e)}")
            self.logger.error(f"🚨 오류 타입: {type(e).__name__}")
            self.logger.error(f"⏱️ 처리 시간: {processing_time:.3f}초")
            
            return {"error": str(e)}
    
    def get_collection_sample(self, collection_name, sample_size=10):
        """컬렉션의 샘플 데이터를 반환합니다."""
        self.logger.info(f"🔍 컬렉션 샘플 조회 시작: {collection_name}, 샘플 크기: {sample_size}")
        start_time = time.time()
        
        try:
            # 이미 연결된 Milvus 인스턴스 사용 (vectorenv)
            self.logger.info(f"📚 기존 Milvus 연결 사용: {collection_name}")
            
            # 헬퍼 메서드 사용하여 컬렉션 인스턴스 가져오기
            collection = self._get_collection_instance(collection_name)
            
            # 샘플 데이터 조회
            self.logger.info(f"🔍 샘플 데이터 조회 시작: {collection_name}")
            sample_data = collection.query(
                expr="",
                output_fields=["*"],
                limit=sample_size
            )
            self.logger.info(f"✅ 샘플 데이터 조회 완료: {collection_name}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.info(f"✅ 컬렉션 샘플 조회 성공: {collection_name}")
            self.logger.info(f"📊 샘플 데이터 - 요청: {sample_size}, 실제: {len(sample_data)}")
            self.logger.info(f"⏱️ 처리 시간: {processing_time:.3f}초")
            
            return {
                "collection_name": collection_name,
                "sample_size": len(sample_data),
                "entities": sample_data
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.error(f"❌ 컬렉션 샘플 조회 실패: {collection_name}")
            self.logger.error(f"🚨 오류 내용: {str(e)}")
            self.logger.error(f"🚨 오류 타입: {type(e).__name__}")
            self.logger.error(f"⏱️ 처리 시간: {processing_time:.3f}초")
            
            return {"error": str(e)}
    
    def search_in_collection(self, collection_name, query, field="text", limit=20):
        """컬렉션에서 키워드 검색을 수행합니다."""
        self.logger.info(f"🔍 컬렉션 검색 시작: {collection_name}, 쿼리: '{query}', 필드: {field}, 제한: {limit}")
        start_time = time.time()
        
        try:
            # 이미 연결된 Milvus 인스턴스 사용 (vectorenv)
            self.logger.info(f"📚 기존 Milvus 연결 사용: {collection_name}")
            
            # 헬퍼 메서드 사용하여 컬렉션 인스턴스 가져오기
            collection = self._get_collection_instance(collection_name)
            
            # 키워드 검색 수행
            search_expr = f'{field} like "%{query}%"'
            self.logger.info(f"🔍 검색 표현식: {search_expr}")
            
            self.logger.info(f"🔍 검색 실행 시작: {collection_name}")
            search_results = collection.query(
                expr=search_expr,
                output_fields=["*"],
                limit=limit
            )
            self.logger.info(f"✅ 검색 실행 완료: {collection_name}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.info(f"✅ 컬렉션 검색 성공: {collection_name}")
            self.logger.info(f"📊 검색 결과 - 쿼리: '{query}', 결과 수: {len(search_results)}")
            self.logger.info(f"⏱️ 처리 시간: {processing_time:.3f}초")
            
            return {
                "collection_name": collection_name,
                "query": query,
                "field": field,
                "total_count": len(search_results),
                "entities": search_results
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.error(f"❌ 컬렉션 검색 실패: {collection_name}")
            self.logger.error(f"🚨 오류 내용: {str(e)}")
            self.logger.error(f"🚨 오류 타입: {type(e).__name__}")
            self.logger.error(f"🚨 처리 시간: {processing_time:.3f}초")
            
            return {"error": str(e)}
    
    def get_collection_data(self, collection_name, limit=100, offset=0, include_embeddings=False):
        """컬렉션의 데이터를 조회합니다."""
        self.logger.info(f"🔍 컬렉션 데이터 조회 시작: {collection_name}, 제한: {limit}, 오프셋: {offset}, 임베딩 포함: {include_embeddings}")
        start_time = time.time()
        
        try:
            # 이미 연결된 Milvus 인스턴스 사용 (vectorenv)
            self.logger.info(f"📚 기존 Milvus 연결 사용: {collection_name}")
            
            # 헬퍼 메서드 사용하여 컬렉션 인스턴스 가져오기
            collection = self._get_collection_instance(collection_name)
            
            # === 새로운 위계형 필드들 완전 포함 ===
            output_fields = [
                "passage_uid", "doc_id", "raw_doc_id", "passage_id", "domain", 
                "title", "author", "text", "info", "tags",
                
                # === 위계형 필드들 ===
                "chapter_number", "chapter_title", 
                "section_number", "section_title",
                "division_number", "division_title",
                "article_number", "article_title", 
                "paragraph_number", "subparagraph_number",
                "item_number",
                
                # === 상태 플래그들 ===
                "is_omission", "is_deletion", "is_amendment", 
                "is_appendix", "is_attachment"
            ]
            
            if include_embeddings:
                output_fields.append("text_emb")
                self.logger.info(f"📊 임베딩 필드 포함: text_emb")
            
            self.logger.info(f"📋 출력 필드 설정 완료: {len(output_fields)}개 필드")
            self.logger.info(f"🔍 데이터 조회 시작: {collection_name}")
            
            # 데이터 조회
            data = collection.query(
                expr="",
                output_fields=output_fields,
                limit=limit,
                offset=offset
            )
            self.logger.info(f"✅ 데이터 조회 완료: {collection_name}")
            
            total_count = collection.num_entities
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.info(f"✅ 컬렉션 데이터 조회 성공: {collection_name}")
            self.logger.info(f"📊 데이터 정보 - 전체: {total_count}, 반환: {len(data)}")
            self.logger.info(f"⏱️ 처리 시간: {processing_time:.3f}초")
            
            return {
                "collection_name": collection_name,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "entities": data
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.error(f"❌ 컬렉션 데이터 조회 실패: {collection_name}")
            self.logger.error(f"🚨 오류 내용: {str(e)}")
            self.logger.error(f"🚨 오류 타입: {type(e).__name__}")
            self.logger.error(f"⏱️ 처리 시간: {processing_time:.3f}초")
            
            return {"error": str(e)}
    
    def delete_node(self, node_id, collection_name):
        """특정 노드(passage)를 삭제합니다."""
        try:
            # 헬퍼 메서드 사용하여 컬렉션 인스턴스 가져오기
            collection = self._get_collection_instance(collection_name)
            
            # 노드 삭제
            collection.delete(f'passage_uid == "{node_id}"')
            collection.flush()
            
            self.logger.info(f"✅ 노드 삭제 완료: {node_id}")
            return True
        except Exception as e:
            self.logger.error(f"노드 삭제 중 오류: {e}")
            return False
    

    
    def insert_hierarchical_data(self, domain, doc_id, title, author, text, info, tags, ignore=True):
        """
        위계형 데이터 삽입 - 기존 insert_data를 완전히 복제하여 조항 단위 청킹 적용
        
        기존의 모든 배치 처리, GPU 관리, 데이터 파이프라인을 그대로 활용합니다.
        """
        try:
            # 시간 로깅을 위한 로거 설정
            import logging
            import threading
            import time
            import json
            import concurrent.futures
            import os
            from pymilvus import Collection
            
            timing_logger = logging.getLogger('timing')
            
            # DB 세마포어 및 배치 처리 락 초기화 (한 번만)
            if self.__class__.db_semaphore is None:
                max_db_connections = int(os.getenv('MAX_DB_CONNECTIONS', '20'))
                batch_size = int(os.getenv('BATCH_SIZE', '10'))
                self.__class__.db_semaphore = threading.BoundedSemaphore(max_db_connections)
                self.__class__.batch_lock = threading.Lock()
                self.__class__.batch_size = batch_size
                print(f"[DEBUG] Initialized DB connection semaphore with max {max_db_connections} connections and batch size {batch_size}")
            
            print(f"[DEBUG] Original text length: {len(text)}")
            
            # doc_id 해시 처리 (기존과 동일)
            hashed_doc_id = self.data_p.hash_text(doc_id, hash_type='blake')
            try:
                date = tags.get('date', '00000000').replace('-','')
                raw_doc_id = f"{date}-{title}-{author}"
                if len(raw_doc_id.encode('utf-8')) > 1024:
                    raw_doc_id = raw_doc_id[:200] + "..."
            except Exception as e:
                print(f"[WARNING] Error creating raw_doc_id: {str(e)}")
                raw_doc_id = f"unknown_doc_{hashed_doc_id[:8]}"
            print(f"[DEBUG] Hashed doc_id: {hashed_doc_id}, Raw doc_id: {raw_doc_id}")
            
            # 중복 문서 체크 (기존과 동일)
            duplicate_results = self.check_duplicates([hashed_doc_id], domain)
            print(f"[DEBUG] 중복 체크 결과: {duplicate_results}")
            
            # 중복된 문서가 존재하는 경우 (기존과 동일)
            if duplicate_results and hashed_doc_id in duplicate_results:
                # duplicate_results가 리스트인 경우 처리
                if isinstance(duplicate_results, list):
                    existing_chunks = len(duplicate_results)  # 리스트 길이로 청크 수 추정
                else:
                    # 딕셔너리인 경우 기존 방식 사용
                    existing_chunks = len(duplicate_results.get(hashed_doc_id, []))
                
                print(f"[DEBUG] Document with doc_id {hashed_doc_id} already exists in domain {domain} with at least {existing_chunks} chunks")
                timing_logger.info(f"DUPLICATE_FOUND - doc_id: {hashed_doc_id}, chunks: {existing_chunks}, ignore: {ignore}")
                
                if ignore:
                    print(f"[DEBUG] Skipping document due to ignore=True")
                    timing_logger.info(f"DUPLICATE_SKIPPED - doc_id: {hashed_doc_id}")
                    return "skipped"
                else:
                    print(f"[DEBUG] Deleting existing document due to ignore=False")
                    
                    delete_start_time = time.time()
                    timing_logger.info(f"DELETE_START - doc_id: {hashed_doc_id}, existing_chunks: {existing_chunks}")
                    
                    try:
                        delete_success = self.delete_data(domain, hashed_doc_id)
                        if not delete_success:
                            print(f"[ERROR] Failed to delete existing document")
                            return "error"
                        
                        delete_end_time = time.time()
                        delete_duration = delete_end_time - delete_start_time
                        timing_logger.info(f"DELETE_END - doc_id: {hashed_doc_id}, duration: {delete_duration:.4f}s")
                        print(f"[DEBUG] Successfully deleted existing document in {delete_duration:.4f}s")
                        
                    except Exception as delete_error:
                        delete_error_time = time.time()
                        delete_duration = delete_error_time - delete_start_time
                        timing_logger.error(f"DELETE_ERROR - doc_id: {hashed_doc_id}, duration: {delete_duration:.4f}s, error: {str(delete_error)}")
                        print(f"[ERROR] Failed to delete existing document: {str(delete_error)}")
                        return "error"
            
            # === 위계형 청킹 시작 (여기서만 변경) ===
            chunk_split_start = time.time()
            timing_logger.info(f"HIERARCHICAL_CHUNK_SPLIT_START - doc_id: {hashed_doc_id}")
            
            # 조항 단위 청킹
            chunked_data = self.chunk_by_articles(text)
            
            chunk_split_end = time.time()
            chunk_split_duration = chunk_split_end - chunk_split_start
            timing_logger.info(f"HIERARCHICAL_CHUNK_SPLIT_END - doc_id: {hashed_doc_id}, chunks: {len(chunked_data)}, duration: {chunk_split_duration:.4f}s")
            print(f"[DEBUG] Number of hierarchical chunks: {len(chunked_data)}")
            
            # info와 tags가 문자열인 경우 파싱 (기존과 동일)
            print(f"[DEBUG] Original info type: {type(info)}, value: {info}")
            print(f"[DEBUG] Original tags type: {type(tags)}, value: {tags}")
            
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                    print(f"[DEBUG] Parsed info: {info}")
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse info JSON: {e}")
                    info = {}
            elif info is None:
                print(f"[DEBUG] Info is None, setting to empty dict")
                info = {}
                
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                    print(f"[DEBUG] Parsed tags: {tags}")
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse tags JSON: {e}")
                    tags = {}
            elif tags is None:
                print(f"[DEBUG] Tags is None, setting to empty dict")
                tags = {}
            
            # 청크 처리를 위한 병렬 처리 함수 (위계형 버전)
            def process_hierarchical_chunk(chunk_data):
                try:
                    i, (chunk_text, hierarchy) = chunk_data
                    total_chunks = len(chunked_data)
                    print(f"[DEBUG] Processing hierarchical chunk {i+1}/{total_chunks} in thread")
                    
                    # 텍스트 길이 체크 (위계형은 더 큰 청크 허용)
                    chunk_bytes = len(chunk_text.encode('utf-8'))
                    if chunk_bytes > 10000:  # 위계형은 10KB까지 허용
                        error_msg = f"Text chunk too large: {chunk_bytes} bytes exceeds maximum 10000 bytes"
                        print(f"[ERROR] {error_msg}")
                        raise ValueError(error_msg)
                    
                    # passage의 고유 식별자 생성 (기존과 동일)
                    passage_id = i + 1
                    passage_uid = f"{hashed_doc_id}_{passage_id}"
                    
                    # 개별 청크 임베딩 시작 (기존과 동일)
                    chunk_emb_start = time.time()
                    
                    # 임베딩 생성 (GPU 제한 적용됨)
                    chunk_emb = self.emb_model.bge_embed_data(chunk_text)
                    
                    chunk_emb_end = time.time()
                    chunk_emb_duration = chunk_emb_end - chunk_emb_start
                    
                    # 임베딩 결과 검증
                    if not chunk_emb or len(chunk_emb) == 0:
                        raise ValueError(f"Empty embedding generated for chunk {i+1}")
                    
                    # === 위계형 데이터 구조 (위계 필드 추가) ===
                    data_item = {
                        "passage_uid": passage_uid,
                        "doc_id": hashed_doc_id, 
                        "raw_doc_id": raw_doc_id,
                        "passage_id": passage_id, 
                        "domain": domain, 
                        "title": title, 
                        "author": author,
                        "text": chunk_text, 
                        "text_emb": chunk_emb, 
                        "info": info, 
                        "tags": tags,
                        
                        # === 위계형 필드 추가 ===
                        "chapter_number": hierarchy.get("chapter_number", ""),
                        "chapter_title": hierarchy.get("chapter_title", ""),
                        "section_number": hierarchy.get("section_number", ""),
                        "section_title": hierarchy.get("section_title", ""),
                        "division_number": hierarchy.get("division_number", ""),
                        "division_title": hierarchy.get("division_title", ""),
                        "article_number": hierarchy.get("article_number", ""),
                        "article_title": hierarchy.get("article_title", ""),
                        "paragraph_number": hierarchy.get("paragraph_number", ""),
                        "subparagraph_number": hierarchy.get("subparagraph_number", ""),
                        "item_number": hierarchy.get("item_number", ""),
                        "is_omission": hierarchy.get("is_omission", False),
                        "is_deletion": hierarchy.get("is_deletion", False),
                        "is_amendment": hierarchy.get("is_amendment", False),
                        "is_appendix": hierarchy.get("is_appendix", False),
                        "is_attachment": hierarchy.get("is_attachment", False),
                        "appendix_type": "main",  # main: 메인 법령, 부칙: 부칙, 별지: 별지
                    }
                    
                    print(f"[DEBUG] Data item info field: {data_item['info']} (type: {type(data_item['info'])})")
                    print(f"[DEBUG] Data item tags field: {data_item['tags']} (type: {type(data_item['tags'])})")
                    
                    data = [data_item]        
                    
                    # DB 삽입 시작 (배치 처리 사용 - 기존과 동일)
                    db_insert_start = time.time()
                    print(f"[DEBUG] Preparing hierarchical chunk {i+1} with passage_uid: {passage_uid} for batch insert")
                    print(f"[DEBUG] Hierarchical info: chapter={hierarchy.get('chapter_number', 'N/A')}({hierarchy.get('chapter_title', '')}), article={hierarchy.get('article_number', 'N/A')}({hierarchy.get('article_title', '')}), paragraph={hierarchy.get('paragraph_number', 'N/A')}, subpara={hierarchy.get('subparagraph_number', 'N/A')}, item={hierarchy.get('item_number', 'N/A')}")
                    print(f"[DEBUG] Status flags: omission={hierarchy.get('is_omission', False)}, deletion={hierarchy.get('is_deletion', False)}, amendment={hierarchy.get('is_amendment', False)}, appendix={hierarchy.get('is_appendix', False)}, attachment={hierarchy.get('is_attachment', False)}")
                    
                    try:
                        # 배치에 추가하고 필요시 삽입
                        data_item = data[0]
                        print(f"[DEBUG] About to call _add_to_batch_and_insert with data_item keys: {list(data_item.keys())}")
                        print(f"[DEBUG] Data item sample: passage_uid={data_item.get('passage_uid')}, title={data_item.get('title')}")
                        batch_inserted = self._add_to_batch_and_insert(data_item, domain)
                        
                        db_insert_end = time.time()
                        db_insert_duration = db_insert_end - db_insert_start
                        
                        if batch_inserted:
                            print(f"[DEBUG] Hierarchical chunk {i+1} triggered a batch insert")
                        else:
                            print(f"[DEBUG] Hierarchical chunk {i+1} added to batch (will be inserted later)")
                            
                        print(f"[TIMING] Hierarchical chunk {i+1} - embedding: {chunk_emb_duration:.4f}s, batch_process: {db_insert_duration:.4f}s")
                        return f"hierarchical_chunk_{i+1}_success"
                        
                    except Exception as db_error:
                        db_insert_error_time = time.time()
                        db_insert_error_duration = db_insert_error_time - db_insert_start
                        error_msg = f"DB insert failed for hierarchical chunk {i+1}: {str(db_error)} (duration: {db_insert_error_duration:.4f}s)"
                        print(f"[ERROR] {error_msg}")
                        raise Exception(error_msg)
                    
                except Exception as e:
                    error_msg = f"Error processing hierarchical chunk {i+1}: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    raise Exception(error_msg)
            
            # 임베딩 및 DB 삽입 시작 (기존과 동일)
            embedding_start_time = time.time()
            timing_logger.info(f"HIERARCHICAL_EMBEDDING_START - doc_id: {hashed_doc_id}, chunks: {len(chunked_data)}")
            
            # 청크별 임베딩 생성을 병렬 처리 (기존과 동일)
            max_workers = min(
                int(os.getenv('INSERT_CHUNK_THREADS', '10')),
                len(chunked_data),
            )
            print(f"[DEBUG] Using {max_workers} threads for hierarchical chunk embedding processing (total chunks: {len(chunked_data)})")
            
            chunk_start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 청크 데이터와 인덱스를 함께 전달
                chunk_data_list = [(i, chunk_data) for i, chunk_data in enumerate(chunked_data)]
                
                # 모든 청크를 병렬로 처리
                future_to_chunk = {executor.submit(process_hierarchical_chunk, chunk_data): chunk_data for chunk_data in chunk_data_list}
                
                # 결과 수집 및 오류 처리 (기존과 동일)
                successful_chunks = 0
                failed_chunks = 0
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_data = future_to_chunk[future]
                    chunk_index = chunk_data[0]
                    try:
                        result = future.result()
                        print(f"[DEBUG] {result}")
                        successful_chunks += 1
                    except Exception as exc:
                        failed_chunks += 1
                        error_msg = f"Hierarchical chunk {chunk_index + 1} processing failed: {exc}"
                        print(f"[ERROR] {error_msg}")
                        timing_logger.error(f"HIERARCHICAL_CHUNK_ERROR - chunk: {chunk_index + 1}, error: {str(exc)}")
            
            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - chunk_start_time
            timing_logger.info(f"HIERARCHICAL_CHUNK_PROCESSING_END - doc_id: {hashed_doc_id}, successful: {successful_chunks}, failed: {failed_chunks}, duration: {chunk_duration:.4f}s")
            
            embedding_end_time = time.time()
            embedding_duration = embedding_end_time - embedding_start_time
            timing_logger.info(f"HIERARCHICAL_EMBEDDING_END - doc_id: {hashed_doc_id}, duration: {embedding_duration:.4f}s")
            
            # 최종 결과 반환 (기존과 동일)
            if failed_chunks > 0:
                error_msg = f"Failed to process {failed_chunks} out of {len(chunked_data)} hierarchical chunks"
                print(f"[ERROR] {error_msg}")
                timing_logger.error(f"HIERARCHICAL_INSERT_PARTIAL_FAILURE - doc_id: {hashed_doc_id}, failed: {failed_chunks}/{len(chunked_data)}")
                return "partial_failure"
            
            print(f"[DEBUG] Successfully processed all {successful_chunks} hierarchical chunks")
            timing_logger.info(f"HIERARCHICAL_INSERT_SUCCESS - doc_id: {hashed_doc_id}, chunks: {successful_chunks}")
            return "success"
            
        except Exception as e:
            self.logger.error(f"위계형 데이터 삽입 중 오류: {e}")
            timing_logger.error(f"HIERARCHICAL_INSERT_ERROR - doc_id: {hashed_doc_id if 'hashed_doc_id' in locals() else 'unknown'}, error: {str(e)}")
            raise
    
    def search_hierarchical_data(self, query: str, top_k: int, 
                               filter_conditions: Dict = None) -> List[Dict[str, Any]]:
        """
        위계형 데이터 검색 - 기존 retrieve_data를 확장
        
        조문 참조가 있으면 정확한 검색, 없으면 기존 벡터 검색 사용
        """
        try:
            self.logger.info(f"🔍 위계형 데이터 검색 시작: {query}")
            
            # 조문 참조 분석
            legal_refs = self._extract_legal_references(query)
            
            if legal_refs["has_references"]:
                # 조문 참조가 있으면 정확한 검색
                results = self._search_by_legal_references(query, top_k, legal_refs, filter_conditions)
            else:
                # 조문 참조가 없으면 기존 벡터 검색
                results = super().retrieve_data(query, top_k, filter_conditions)
            
            self.logger.info(f"✅ 위계형 데이터 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            self.logger.error(f"위계형 데이터 검색 중 오류: {e}")
            return []
    
    def _extract_legal_references(self, query: str) -> Dict[str, Any]:
        """쿼리에서 조문 참조 추출"""
        try:
            refs = {
                "has_references": False,
                "articles": [],
                "paragraphs": [],
                "items": []
            }
            
            # 조문 패턴 매칭
            article_matches = re.finditer(self.article_patterns["main_article"], query)
            for match in article_matches:
                refs["articles"].append(f"제{match.group(1)}조")
                refs["has_references"] = True
            
            # 조문의 조 패턴 매칭
            sub_article_matches = re.finditer(self.article_patterns["sub_article"], query)
            for match in sub_article_matches:
                refs["articles"].append(f"제{match.group(1)}조의{match.group(2)}")
                refs["has_references"] = True
            
            # 항 패턴 매칭
            paragraph_matches = re.finditer(self.article_patterns["paragraph"], query)
            for match in paragraph_matches:
                refs["paragraphs"].append(f"{match.group(1)}.")
                refs["has_references"] = True
            
            # 호 패턴 매칭
            item_matches = re.finditer(self.article_patterns["item"], query)
            for match in item_matches:
                refs["items"].append(f"{match.group(1)})")
                refs["has_references"] = True
            
            return refs
            
        except Exception as e:
            self.logger.error(f"조문 참조 추출 중 오류: {e}")
            return {"has_references": False, "articles": [], "paragraphs": [], "items": []}
    
    def _search_by_legal_references(self, query: str, top_k: int, 
                                  legal_refs: Dict[str, Any], 
                                  filter_conditions: Dict = None) -> List[Dict[str, Any]]:
        """조문 참조 기반 정확한 검색"""
        try:
            # 기본 도메인 설정
            domain = "legal"
            if filter_conditions and "domain" in filter_conditions:
                domain = filter_conditions["domain"]
            
            # 조문 참조로 필터링 조건 구성
            expr_parts = []
            
            # 조문 필터
            if legal_refs["articles"]:
                article_expr = " || ".join([f'article_number == "{article}"' for article in legal_refs["articles"]])
                expr_parts.append(f"({article_expr})")
            
            # 항 필터
            if legal_refs["paragraphs"]:
                paragraph_expr = " || ".join([f'paragraph_number == "{paragraph}"' for paragraph in legal_refs["paragraphs"]])
                expr_parts.append(f"({paragraph_expr})")
            
            # 호 필터
            if legal_refs["items"]:
                item_expr = " || ".join([f'item_number == "{item}"' for item in legal_refs["items"]])
                expr_parts.append(f"({item_expr})")
            
            # 기존 필터 조건 추가
            if filter_conditions:
                if "domain" in filter_conditions:
                    expr_parts.append(f'domain == "{filter_conditions["domain"]}"')
            
            # 최종 검색 표현식
            expr = " && ".join(expr_parts) if expr_parts else None
            
            # 기존 retrieve_data의 검색 로직 활용
            results = super().retrieve_data(query, top_k, filter_conditions)
            
            # 조문 참조가 있는 결과만 필터링
            if expr:
                filtered_results = []
                for result in results:
                    # 조문 참조와 일치하는지 확인
                    if self._matches_legal_references(result, legal_refs):
                        filtered_results.append(result)
                return filtered_results
            
            return results
            
        except Exception as e:
            self.logger.error(f"조문 참조 검색 중 오류: {e}")
            return []
    
    def _matches_legal_references(self, result: Dict[str, Any], legal_refs: Dict[str, Any]) -> bool:
        """결과가 조문 참조와 일치하는지 확인"""
        try:
            # 조문 매칭
            if legal_refs["articles"]:
                result_article = result.get("article_number", "")
                if not any(article in result_article for article in legal_refs["articles"]):
                    return False
            
            # 항 매칭
            if legal_refs["paragraphs"]:
                result_paragraph = result.get("paragraph_number", "")
                if not any(paragraph in result_paragraph for paragraph in legal_refs["paragraphs"]):
                    return False
            
            # 호 매칭
            if legal_refs["items"]:
                result_item = result.get("item_number", "")
                if not any(item in result_item for item in legal_refs["items"]):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"조문 참조 매칭 중 오류: {e}")
            return True  # 오류 시 매칭된 것으로 처리
