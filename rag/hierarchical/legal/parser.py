"""
법령 문서 파싱 클래스

대한민국 법령 구조에 맞춰 조문을 파싱하고 위계형 구조로 변환합니다.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import logging


class LegalParser:
    """법령 문서 파싱 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 설정 로더 초기화
        from ..config.config_loader import HierarchicalConfigLoader
        self.config_loader = HierarchicalConfigLoader()
        
        # 법령 패턴 정의
        self.patterns = self._init_legal_patterns()
        
        # 법령 구조 매핑
        self.hierarchy_mapping = {
            "law": 0,       # 법령명
            "part": 1,      # 편
            "chapter": 2,   # 장
            "section": 3,   # 절
            "article": 4,   # 조
            "paragraph": 5, # 항
            "item": 6,      # 호
            "subitem": 7,   # 목
            "content": 8    # 일반 내용
        }
        
        # 조문 유형 분류
        self.article_types = {
            "목적": "purpose",
            "정의": "definition", 
            "적용범위": "scope",
            "원칙": "principle",
            "절차": "procedure",
            "기준": "standard",
            "벌칙": "penalty",
            "부칙": "supplementary"
        }
    
    def _init_legal_patterns(self) -> Dict[str, str]:
        """법령 패턴 초기화"""
        return {
            # 법령명
            "law_title": r"^(.+?법)(?:\s*\(.+?\))?\s*$",
            
            # 편/장/절
            "part": r"제(\d+)편\s+(.+?)$",
            "chapter": r"제(\d+)장\s+(.+?)$", 
            "section": r"제(\d+)절\s+(.+?)$",
            
            # 조문
            "article": r"제(\d+)조(?:의(\d+))?\s*(?:\((.+?)\))?\s*(.*)$",
            
            # 항
            "paragraph": r"^(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩|⑪|⑫|⑬|⑭|⑮|⑯|⑰|⑱|⑲|⑳)\s*(.+?)$",
            
            # 호
            "item": r"^(\d+)\.\s+(.+?)$",
            
            # 목
            "subitem": r"^([가-힣])\.\s+(.+?)$",
            
            # 다만서, 단서
            "proviso": r"(다만|단,)\s*(.+?)$",
            
            # 법령 번호
            "law_number": r"(법률|대통령령|총리령|부령)\s*제(\d+)호",
            
            # 날짜
            "date": r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",
            
            # 법령 제정일
            "enactment_date": r"제정\s*(\d{4})[.년]\s*(\d{1,2})[.월]\s*(\d{1,2})[.일]?",
            
            # 법령 시행일
            "enforcement_date": r"시행\s*(\d{4})[.년]\s*(\d{1,2})[.월]\s*(\d{1,2})[.일]?",
            
            # 개정일
            "amendment_date": r"개정\s*(\d{4})[.년]\s*(\d{1,2})[.월]\s*(\d{1,2})[.일]?",
            
            # 조문 참조
            "article_ref": r"제(\d+)조(?:의(\d+))?(?:\s*제(\d+)항)?(?:\s*제(\d+)호)?",
            
            # 법령 참조
            "law_ref": r"「(.+?)」",
        }
    
    def parse_legal_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        법령 문서를 위계형 구조로 파싱
        
        Args:
            document: 원본 법령 문서
            
        Returns:
            List[Dict]: 파싱된 위계형 청크들
        """
        try:
            self.logger.info(f"법령 문서 파싱 시작: {document.get('title', 'Unknown')}")
            
            text = document.get("text", "")
            if not text.strip():
                self.logger.warning("빈 텍스트 문서")
                return []
            
            # 기본 정보 추출
            doc_metadata = self._extract_document_metadata(document)
            
            # 텍스트를 라인별로 분할
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # 파싱 컨텍스트 초기화
            context = self._init_parsing_context(doc_metadata)
            
            # 라인별 파싱
            chunks = []
            for line_num, line in enumerate(lines):
                chunk_info = self._parse_line(line, context, line_num)
                if chunk_info:
                    chunks.append(chunk_info)
            
            # 후처리
            processed_chunks = self._post_process_chunks(chunks, doc_metadata)
            
            self.logger.info(f"법령 파싱 완료: {len(processed_chunks)}개 청크")
            return processed_chunks
            
        except Exception as e:
            self.logger.error(f"법령 파싱 중 오류: {e}")
            return []
    
    def _extract_document_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """문서 메타데이터 추출"""
        try:
            metadata = {
                "original_doc": document,
                "law_title": document.get("title", ""),
                "law_type": document.get("law_type", "법률"),
                "law_number": document.get("law_number", ""),
                "ministry": document.get("ministry", ""),
                "enactment_date": document.get("enactment_date", ""),
                "enforcement_date": document.get("enforcement_date", ""),
                "domain": document.get("domain", "legal"),
            }
            
            # 법령 번호 자동 추출 시도
            if not metadata["law_number"]:
                extracted_number = self._extract_law_number(document.get("text", ""))
                if extracted_number:
                    metadata["law_number"] = extracted_number
            
            # 법령 날짜 자동 추출
            text_content = document.get("text", "")
            metadata["enactment_date"] = self._extract_enactment_date(text_content)
            metadata["enforcement_date"] = self._extract_enforcement_date(text_content)
            metadata["amendment_date"] = self._extract_amendment_date(text_content)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"메타데이터 추출 중 오류: {e}")
            return {"original_doc": document}
    
    def _extract_law_number(self, text: str) -> Optional[str]:
        """텍스트에서 법령 번호 추출"""
        try:
            match = re.search(self.patterns["law_number"], text)
            if match:
                law_type = match.group(1)
                number = match.group(2)
                return f"{law_type} 제{number}호"
            return None
        except Exception:
            return None
    
    def _extract_enactment_date(self, text: str) -> str:
        """텍스트에서 제정일 추출 (JSON 설정 기반)"""
        try:
            return self._extract_date_by_priority(text, "제정")
        except Exception as e:
            self.logger.error(f"제정일 추출 중 오류: {e}")
            return ""
    
    def _extract_enforcement_date(self, text: str) -> str:
        """텍스트에서 시행일 추출 (JSON 설정 기반)"""
        try:
            return self._extract_date_by_priority(text, "시행")
        except Exception as e:
            self.logger.error(f"시행일 추출 중 오류: {e}")
            return ""
    
    def _extract_amendment_date(self, text: str) -> str:
        """텍스트에서 개정일 추출 (JSON 설정 기반)"""
        try:
            return self._extract_date_by_priority(text, "개정")
        except Exception as e:
            self.logger.error(f"개정일 추출 중 오류: {e}")
            return ""
    
    def _init_parsing_context(self, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """파싱 컨텍스트 초기화"""
        return {
            "doc_metadata": doc_metadata,
            "current_part": "",
            "current_chapter": "",
            "current_section": "",
            "current_article": "",
            "current_article_title": "",
            "current_paragraph": "",
            "current_item": "",
            "current_subitem": "",
            "provision_type": "본칙",  # 기본값
            "hierarchy_path": f"/{doc_metadata.get('law_title', '')}",
        }
    
    def _parse_line(self, line: str, context: Dict[str, Any], line_num: int) -> Optional[Dict[str, Any]]:
        """단일 라인 파싱"""
        try:
            # 공백 라인 스킵
            if not line.strip():
                return None
            
            # 편 패턴 확인
            part_match = re.match(self.patterns["part"], line)
            if part_match:
                return self._create_part_chunk(part_match, context, line_num)
            
            # 장 패턴 확인
            chapter_match = re.match(self.patterns["chapter"], line)
            if chapter_match:
                return self._create_chapter_chunk(chapter_match, context, line_num)
            
            # 절 패턴 확인
            section_match = re.match(self.patterns["section"], line)
            if section_match:
                return self._create_section_chunk(section_match, context, line_num)
            
            # 조 패턴 확인
            article_match = re.match(self.patterns["article"], line)
            if article_match:
                return self._create_article_chunk(article_match, context, line_num)
            
            # 항 패턴 확인
            paragraph_match = re.match(self.patterns["paragraph"], line)
            if paragraph_match:
                return self._create_paragraph_chunk(paragraph_match, context, line_num)
            
            # 호 패턴 확인
            item_match = re.match(self.patterns["item"], line)
            if item_match:
                return self._create_item_chunk(item_match, context, line_num)
            
            # 목 패턴 확인
            subitem_match = re.match(self.patterns["subitem"], line)
            if subitem_match:
                return self._create_subitem_chunk(subitem_match, context, line_num)
            
            # 일반 내용
            return self._create_content_chunk(line, context, line_num)
            
        except Exception as e:
            self.logger.error(f"라인 파싱 중 오류 (line {line_num}): {e}")
            return None
    
    def _create_part_chunk(self, match, context: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """편 청크 생성"""
        part_num = match.group(1)
        part_title = match.group(2)
        
        context["current_part"] = f"제{part_num}편"
        context["current_chapter"] = ""
        context["current_section"] = ""
        context["current_article"] = ""
        
        return {
            "text": f"제{part_num}편 {part_title}",
            "hierarchy_level": self.hierarchy_mapping["part"],
            "section_type": "part",
            "section_number": f"제{part_num}편",
            "hierarchy_path": f"{context['hierarchy_path']}/제{part_num}편",
            "metadata": {
                "line_number": line_num,
                "part_number": part_num,
                "part_title": part_title,
                "context": context.copy()
            }
        }
    
    def _create_chapter_chunk(self, match, context: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """장 청크 생성"""
        chapter_num = match.group(1)
        chapter_title = match.group(2)
        
        context["current_chapter"] = f"제{chapter_num}장"
        context["current_section"] = ""
        context["current_article"] = ""
        
        return {
            "text": f"제{chapter_num}장 {chapter_title}",
            "hierarchy_level": self.hierarchy_mapping["chapter"],
            "section_type": "chapter",
            "section_number": f"제{chapter_num}장",
            "hierarchy_path": f"{context['hierarchy_path']}/제{chapter_num}장",
            "metadata": {
                "line_number": line_num,
                "chapter_number": chapter_num,
                "chapter_title": chapter_title,
                "context": context.copy()
            }
        }
    
    def _create_section_chunk(self, match, context: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """절 청크 생성"""
        section_num = match.group(1)
        section_title = match.group(2)
        
        context["current_section"] = f"제{section_num}절"
        context["current_article"] = ""
        
        return {
            "text": f"제{section_num}절 {section_title}",
            "hierarchy_level": self.hierarchy_mapping["section"],
            "section_type": "section",
            "section_number": f"제{section_num}절",
            "hierarchy_path": f"{context['hierarchy_path']}/제{section_num}절",
            "metadata": {
                "line_number": line_num,
                "section_number": section_num,
                "section_title": section_title,
                "context": context.copy()
            }
        }
    
    def _create_article_chunk(self, match, context: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """조 청크 생성"""
        article_num = match.group(1)
        article_sub = match.group(2) or ""  # 조의2, 조의3 등
        article_title = match.group(3) or ""
        article_content = match.group(4) or ""
        
        # 조 번호 생성
        if article_sub:
            full_article_num = f"제{article_num}조의{article_sub}"
        else:
            full_article_num = f"제{article_num}조"
        
        context["current_article"] = full_article_num
        context["current_article_title"] = article_title
        context["current_paragraph"] = ""
        context["current_item"] = ""
        
        # 조문 유형 판단
        article_type = self._classify_article_type(article_title, article_content)
        
        # 전체 텍스트 구성
        full_text_parts = [full_article_num]
        if article_title:
            full_text_parts.append(f"({article_title})")
        if article_content:
            full_text_parts.append(article_content)
        
        full_text = " ".join(full_text_parts)
        
        return {
            "text": full_text,
            "hierarchy_level": self.hierarchy_mapping["article"],
            "section_type": "article",
            "section_number": full_article_num,
            "hierarchy_path": f"{context['hierarchy_path']}/{full_article_num}",
            "article_number": full_article_num,
            "article_title": article_title,
            "article_type": article_type,
            "provision_type": context.get("provision_type", "본칙"),
            "metadata": {
                "line_number": line_num,
                "article_num": article_num,
                "article_sub": article_sub,
                "article_content": article_content,
                "context": context.copy()
            }
        }
    
    def _create_paragraph_chunk(self, match, context: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """항 청크 생성"""
        paragraph_symbol = match.group(1)
        paragraph_content = match.group(2)
        
        context["current_paragraph"] = paragraph_symbol
        context["current_item"] = ""
        context["current_subitem"] = ""
        
        return {
            "text": f"{paragraph_symbol} {paragraph_content}",
            "hierarchy_level": self.hierarchy_mapping["paragraph"],
            "section_type": "paragraph",
            "section_number": paragraph_symbol,
            "hierarchy_path": f"{context['hierarchy_path']}/{paragraph_symbol}",
            "article_number": context.get("current_article", ""),
            "paragraph_number": paragraph_symbol,
            "provision_type": context.get("provision_type", "본칙"),
            "metadata": {
                "line_number": line_num,
                "paragraph_content": paragraph_content,
                "context": context.copy()
            }
        }
    
    def _create_item_chunk(self, match, context: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """호 청크 생성"""
        item_num = match.group(1)
        item_content = match.group(2)
        
        context["current_item"] = f"{item_num}."
        context["current_subitem"] = ""
        
        return {
            "text": f"{item_num}. {item_content}",
            "hierarchy_level": self.hierarchy_mapping["item"],
            "section_type": "item",
            "section_number": f"{item_num}.",
            "hierarchy_path": f"{context['hierarchy_path']}/{item_num}.",
            "article_number": context.get("current_article", ""),
            "paragraph_number": context.get("current_paragraph", ""),
            "item_number": f"{item_num}.",
            "provision_type": context.get("provision_type", "본칙"),
            "metadata": {
                "line_number": line_num,
                "item_content": item_content,
                "context": context.copy()
            }
        }
    
    def _create_subitem_chunk(self, match, context: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """목 청크 생성"""
        subitem_letter = match.group(1)
        subitem_content = match.group(2)
        
        context["current_subitem"] = f"{subitem_letter}."
        
        return {
            "text": f"{subitem_letter}. {subitem_content}",
            "hierarchy_level": self.hierarchy_mapping["subitem"],
            "section_type": "subitem",
            "section_number": f"{subitem_letter}.",
            "hierarchy_path": f"{context['hierarchy_path']}/{subitem_letter}.",
            "article_number": context.get("current_article", ""),
            "paragraph_number": context.get("current_paragraph", ""),
            "item_number": context.get("current_item", ""),
            "subitem_number": f"{subitem_letter}.",
            "provision_type": context.get("provision_type", "본칙"),
            "metadata": {
                "line_number": line_num,
                "subitem_content": subitem_content,
                "context": context.copy()
            }
        }
    
    def _create_content_chunk(self, line: str, context: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """일반 내용 청크 생성"""
        return {
            "text": line,
            "hierarchy_level": self.hierarchy_mapping["content"],
            "section_type": "content",
            "section_number": "",
            "hierarchy_path": context.get("hierarchy_path", "/"),
            "article_number": context.get("current_article", ""),
            "paragraph_number": context.get("current_paragraph", ""),
            "item_number": context.get("current_item", ""),
            "subitem_number": context.get("current_subitem", ""),
            "provision_type": context.get("provision_type", "본칙"),
            "metadata": {
                "line_number": line_num,
                "context": context.copy()
            }
        }
    
    def _classify_article_type(self, article_title: str, article_content: str) -> str:
        """조문 유형 분류"""
        try:
            # 제목 기반 분류
            if article_title:
                for keyword, article_type in self.article_types.items():
                    if keyword in article_title:
                        return article_type
            
            # 내용 기반 분류 (간단한 키워드 매칭)
            content = article_content.lower()
            if "목적" in content or "하기 위하여" in content:
                return "purpose"
            elif "정의" in content or "라 함은" in content:
                return "definition"
            elif "적용범위" in content or "적용한다" in content:
                return "scope"
            elif "벌금" in content or "징역" in content or "처벌" in content:
                return "penalty"
            else:
                return "general"
                
        except Exception as e:
            self.logger.error(f"조문 유형 분류 중 오류: {e}")
            return "general"
    
    def _post_process_chunks(self, chunks: List[Dict[str, Any]], 
                           doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """청크 후처리"""
        try:
            processed_chunks = []
            
            for chunk in chunks:
                # 공통 메타데이터 추가
                chunk.update({
                    "law_type": doc_metadata.get("law_type", ""),
                    "law_number": doc_metadata.get("law_number", ""),
                    "law_title": doc_metadata.get("law_title", ""),
                    "ministry": doc_metadata.get("ministry", ""),
                    "enactment_date": doc_metadata.get("enactment_date", ""),
                    "enforcement_date": doc_metadata.get("enforcement_date", ""),
                })
                
                # 법령 키워드 추출
                legal_keywords = self._extract_legal_keywords(chunk.get("text", ""))
                chunk["legal_keywords"] = legal_keywords
                
                # 참조 조문 추출
                referenced_articles = self._extract_article_references(chunk.get("text", ""))
                chunk["referenced_articles"] = referenced_articles
                
                processed_chunks.append(chunk)
            
            return processed_chunks
            
        except Exception as e:
            self.logger.error(f"청크 후처리 중 오류: {e}")
            return chunks
    
    def _extract_legal_keywords(self, text: str) -> List[str]:
        """법률 키워드 추출 (간단한 구현)"""
        try:
            legal_terms = [
                "권리", "의무", "책임", "절차", "기준", "원칙", "범위",
                "허가", "신고", "승인", "등록", "신청", "보고", "통지",
                "벌금", "과태료", "징역", "처벌", "제재", "금지", "허용"
            ]
            
            found_keywords = []
            for term in legal_terms:
                if term in text:
                    found_keywords.append(term)
            
            return found_keywords
            
        except Exception as e:
            self.logger.error(f"키워드 추출 중 오류: {e}")
            return []
    
    def _extract_article_references(self, text: str) -> List[str]:
        """조문 참조 추출"""
        try:
            references = []
            matches = re.finditer(self.patterns["article_ref"], text)
            
            for match in matches:
                article_num = match.group(1)
                article_sub = match.group(2)
                paragraph_num = match.group(3)
                item_num = match.group(4)
                
                ref = f"제{article_num}조"
                if article_sub:
                    ref += f"의{article_sub}"
                if paragraph_num:
                    ref += f" 제{paragraph_num}항"
                if item_num:
                    ref += f" 제{item_num}호"
                
                references.append(ref)
            
            return references
            
        except Exception as e:
            self.logger.error(f"조문 참조 추출 중 오류: {e}")
            return []
    
    def get_parsing_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """파싱 통계 생성"""
        try:
            stats = {
                "total_chunks": len(chunks),
                "by_section_type": {},
                "by_hierarchy_level": {},
                "article_count": 0,
                "paragraph_count": 0,
                "item_count": 0
            }
            
            for chunk in chunks:
                section_type = chunk.get("section_type", "unknown")
                hierarchy_level = chunk.get("hierarchy_level", 0)
                
                # 섹션 타입별 통계
                if section_type not in stats["by_section_type"]:
                    stats["by_section_type"][section_type] = 0
                stats["by_section_type"][section_type] += 1
                
                # 위계 레벨별 통계
                if hierarchy_level not in stats["by_hierarchy_level"]:
                    stats["by_hierarchy_level"][hierarchy_level] = 0
                stats["by_hierarchy_level"][hierarchy_level] += 1
                
                # 특정 요소 카운트
                if section_type == "article":
                    stats["article_count"] += 1
                elif section_type == "paragraph":
                    stats["paragraph_count"] += 1
                elif section_type == "item":
                    stats["item_count"] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"통계 생성 중 오류: {e}")
            return {"error": str(e)}
