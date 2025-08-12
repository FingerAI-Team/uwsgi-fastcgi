"""
텍스트 처리 유틸리티

위계형 문서 처리에 필요한 텍스트 관련 유틸리티 함수들을 제공합니다.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import logging


class TextProcessor:
    """텍스트 처리 유틸리티 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 한국어 법령 특화 패턴
        self.korean_legal_patterns = {
            "whitespace_normalize": r"\s+",
            "paragraph_split": r"\n\s*\n",
            "bullet_points": r"^[·•▪▫◦‣⁃]\s*",
            "numbering": r"^(\d+)\.\s*",
            "korean_numbering": r"^([가-힣])\.\s*",
            "parentheses": r"\([^)]*\)",
            "brackets": r"\[[^\]]*\]",
            "quotes": r"[「」『』""'']",
        }
        
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        텍스트 정리
        
        Args:
            text: 정리할 텍스트
            aggressive: 강력한 정리 여부
            
        Returns:
            str: 정리된 텍스트
        """
        try:
            if not text:
                return ""
            
            cleaned = text
            
            # 기본 정리
            cleaned = self._normalize_whitespace(cleaned)
            cleaned = self._remove_control_characters(cleaned)
            
            if aggressive:
                # 강력한 정리
                cleaned = self._remove_special_formatting(cleaned)
                cleaned = self._normalize_punctuation(cleaned)
            
            return cleaned.strip()
            
        except Exception as e:
            self.logger.error(f"텍스트 정리 중 오류: {e}")
            return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """공백 문자 정규화"""
        # 여러 공백을 하나로
        text = re.sub(self.korean_legal_patterns["whitespace_normalize"], " ", text)
        
        # 불필요한 줄바꿈 제거 (2개 이상의 연속 줄바꿈은 2개로)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text
    
    def _remove_control_characters(self, text: str) -> str:
        """제어 문자 제거"""
        # 출력 가능한 문자, 공백, 탭, 줄바꿈만 유지
        cleaned = ''.join(char for char in text 
                         if char.isprintable() or char in '\n\t\r ')
        return cleaned
    
    def _remove_special_formatting(self, text: str) -> str:
        """특수 서식 제거"""
        # 괄호 안 내용 제거 (옵션)
        # text = re.sub(self.korean_legal_patterns["parentheses"], "", text)
        
        # 불릿 포인트 제거
        text = re.sub(self.korean_legal_patterns["bullet_points"], "", text, flags=re.MULTILINE)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """문장부호 정규화"""
        # 한국어 따옴표 정규화
        text = re.sub(r"[「『]", '"', text)
        text = re.sub(r"[」』]", '"', text)
        
        # 연속된 문장부호 정리
        text = re.sub(r"[.]{2,}", "...", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        
        return text
    
    def extract_paragraphs(self, text: str, min_length: int = 10) -> List[str]:
        """문단 추출"""
        try:
            # 문단 분리
            paragraphs = re.split(self.korean_legal_patterns["paragraph_split"], text)
            
            # 정리 및 필터링
            cleaned_paragraphs = []
            for para in paragraphs:
                cleaned = self.clean_text(para)
                if len(cleaned) >= min_length:
                    cleaned_paragraphs.append(cleaned)
            
            return cleaned_paragraphs
            
        except Exception as e:
            self.logger.error(f"문단 추출 중 오류: {e}")
            return [text]
    
    def extract_sentences(self, text: str, min_length: int = 5) -> List[str]:
        """문장 추출"""
        try:
            # 한국어 문장 분리 패턴
            sentence_endings = r'[.!?](?=\s|$)|[。！？](?=\s|$)'
            sentences = re.split(sentence_endings, text)
            
            # 정리 및 필터링
            cleaned_sentences = []
            for sentence in sentences:
                cleaned = self.clean_text(sentence)
                if len(cleaned) >= min_length:
                    cleaned_sentences.append(cleaned)
            
            return cleaned_sentences
            
        except Exception as e:
            self.logger.error(f"문장 추출 중 오류: {e}")
            return [text]
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """키워드 추출 (간단한 구현)"""
        try:
            # 불용어 정의
            stopwords = {
                '이', '가', '을', '를', '은', '는', '에', '의', '로', '으로',
                '와', '과', '도', '만', '부터', '까지', '에서', '에게', '에게서',
                '으며', '이며', '그리고', '또한', '그러나', '하지만', '따라서'
            }
            
            # 단어 추출 (2글자 이상)
            words = re.findall(r'[가-힣]{2,}', text)
            
            # 빈도 계산
            word_freq = {}
            for word in words:
                if word not in stopwords:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # 빈도순 정렬하여 상위 키워드 반환
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, freq in sorted_words[:max_keywords]]
            
            return keywords
            
        except Exception as e:
            self.logger.error(f"키워드 추출 중 오류: {e}")
            return []
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """텍스트 요약 생성 (간단한 구현)"""
        try:
            if len(text) <= max_length:
                return text
            
            # 문장별로 분리
            sentences = self.extract_sentences(text)
            
            if not sentences:
                return text[:max_length] + "..."
            
            # 첫 번째 문장이 너무 길지 않으면 사용
            first_sentence = sentences[0]
            if len(first_sentence) <= max_length:
                return first_sentence
            
            # 아니면 단순 자르기
            return text[:max_length] + "..."
            
        except Exception as e:
            self.logger.error(f"요약 생성 중 오류: {e}")
            return text[:max_length] + "..."
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (간단한 Jaccard 유사도)"""
        try:
            # 단어 집합 생성
            words1 = set(re.findall(r'[가-힣a-zA-Z]+', text1.lower()))
            words2 = set(re.findall(r'[가-힣a-zA-Z]+', text2.lower()))
            
            if not words1 and not words2:
                return 1.0
            
            if not words1 or not words2:
                return 0.0
            
            # Jaccard 유사도 계산
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"유사도 계산 중 오류: {e}")
            return 0.0
    
    def generate_content_hash(self, text: str, algorithm: str = "sha256") -> str:
        """내용 해시 생성"""
        try:
            # 텍스트 정규화
            normalized = self.clean_text(text, aggressive=True)
            
            # 해시 생성
            if algorithm == "md5":
                hash_obj = hashlib.md5()
            elif algorithm == "sha1":
                hash_obj = hashlib.sha1()
            elif algorithm == "sha256":
                hash_obj = hashlib.sha256()
            else:
                hash_obj = hashlib.blake2b(digest_size=16)
            
            hash_obj.update(normalized.encode('utf-8'))
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.error(f"해시 생성 중 오류: {e}")
            return "error_hash"
    
    def chunk_text_by_length(self, text: str, max_length: int = 1000, 
                           overlap: int = 100) -> List[Dict[str, Any]]:
        """길이 기반 텍스트 청킹"""
        try:
            if len(text) <= max_length:
                return [{
                    "text": text,
                    "start_pos": 0,
                    "end_pos": len(text),
                    "chunk_id": 0,
                    "overlap_prev": False,
                    "overlap_next": False
                }]
            
            chunks = []
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = min(start + max_length, len(text))
                
                # 단어 경계에서 자르기 시도
                if end < len(text):
                    # 뒤에서부터 공백 찾기
                    for i in range(end, max(start, end - 100), -1):
                        if text[i].isspace():
                            end = i
                            break
                
                chunk_text = text[start:end].strip()
                
                chunks.append({
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_id": chunk_id,
                    "overlap_prev": chunk_id > 0 and start < overlap,
                    "overlap_next": end < len(text)
                })
                
                # 다음 청크 시작 위치 (오버랩 고려)
                start = max(0, end - overlap)
                chunk_id += 1
                
                # 무한 루프 방지
                if start >= end:
                    break
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"텍스트 청킹 중 오류: {e}")
            return [{"text": text, "start_pos": 0, "end_pos": len(text), "chunk_id": 0}]
    
    def validate_korean_text(self, text: str) -> Dict[str, Any]:
        """한국어 텍스트 검증"""
        try:
            stats = {
                "total_chars": len(text),
                "korean_chars": 0,
                "english_chars": 0,
                "number_chars": 0,
                "special_chars": 0,
                "whitespace_chars": 0,
                "korean_ratio": 0.0,
                "has_korean": False,
                "encoding_issues": False
            }
            
            for char in text:
                if '\uAC00' <= char <= '\uD7AF':  # 한글 완성형
                    stats["korean_chars"] += 1
                elif char.isalpha():
                    stats["english_chars"] += 1
                elif char.isdigit():
                    stats["number_chars"] += 1
                elif char.isspace():
                    stats["whitespace_chars"] += 1
                else:
                    stats["special_chars"] += 1
            
            if stats["total_chars"] > 0:
                stats["korean_ratio"] = stats["korean_chars"] / stats["total_chars"]
                stats["has_korean"] = stats["korean_chars"] > 0
            
            # 인코딩 문제 체크 (간단한 체크)
            try:
                text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                stats["encoding_issues"] = True
            
            return stats
            
        except Exception as e:
            self.logger.error(f"텍스트 검증 중 오류: {e}")
            return {"error": str(e)}
    
    def find_text_patterns(self, text: str, pattern_type: str = "legal") -> Dict[str, List[str]]:
        """텍스트 패턴 찾기"""
        try:
            patterns = {}
            
            if pattern_type == "legal":
                # 법령 패턴들
                patterns["articles"] = re.findall(r'제\d+조(?:의\d+)?', text)
                patterns["paragraphs"] = re.findall(r'[①-⑳]', text)
                patterns["items"] = re.findall(r'\d+\.', text)
                patterns["law_references"] = re.findall(r'「([^」]+)」', text)
                patterns["dates"] = re.findall(r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일', text)
                
            elif pattern_type == "general":
                # 일반 패턴들
                patterns["emails"] = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
                patterns["urls"] = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
                patterns["phone_numbers"] = re.findall(r'\d{2,3}-\d{3,4}-\d{4}', text)
                patterns["numbers"] = re.findall(r'\d+', text)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"패턴 검색 중 오류: {e}")
            return {"error": str(e)}
