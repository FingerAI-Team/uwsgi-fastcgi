"""
패턴 스캔 엔진

법령 텍스트에서 헤더 패턴을 스캔하고 분류하는 엔진
"""

import re
import logging
from typing import List, Dict, Any, Tuple

class PatternScanner:
    """법령 헤더 패턴 스캐너"""
    
    def __init__(self):
        self.logger = logging.getLogger('hierarchical')
        
        # 정규식 패턴 정의 (FLAGS 설정)
        self.FLAGS = re.UNICODE | re.MULTILINE
        
        # 헤더 패턴 정의
        self.header_patterns = [
            # 장/절/관
            (r"제(\d+)장", "chapter", "장"),
            (r"제(\d+)절", "section", "절"),
            (r"제(\d+)관", "division", "관"),
            
            # 조 (의조 포함)
            (r"제(\d+)조(?:의(\d+))?", "article", "조"),
            
            # 항
            (r"\((\d+)\)", "paragraph", "항"),
            (r"①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩", "paragraph", "항"),
            
            # 호
            (r"(\d+)\.", "subparagraph", "호"),
            
            # 목
            (r"([가-힣])\.", "item", "목"),
        ]
        
        # 컴파일된 정규식 패턴들
        self.compiled_patterns = []
        for pattern, pattern_type, description in self.header_patterns:
            try:
                compiled = re.compile(pattern, self.FLAGS)
                self.compiled_patterns.append((compiled, pattern_type, description))
                self.logger.debug(f"패턴 컴파일 성공: {pattern_type} - {pattern}")
            except Exception as e:
                self.logger.error(f"패턴 컴파일 실패: {pattern_type} - {pattern}, 오류: {e}")
        
        self.logger.info(f"✅ 패턴 스캐너 초기화 완료: {len(self.compiled_patterns)}개 패턴")
    
    def _detect_status_flags(self, text: str) -> Dict[str, Any]:
        """
        텍스트에서 상태 플래그를 감지
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            상태 플래그 딕셔너리
        """
        flags = {
            "is_omission": False,
            "is_deletion": False,
            "is_amendment": False,
            "is_appendix": False,
            "is_attachment": False,
            "appendix_type": "main"
        }
        
        try:
            # 생략 여부
            if re.search(r'생략|omission|\.\.\.|…|중략', text, re.IGNORECASE):
                flags["is_omission"] = True
            
            # 삭제 여부
            if re.search(r'삭제|deletion|취소선|~~|삭제됨', text, re.IGNORECASE):
                flags["is_deletion"] = True
            
            # 개정 여부
            if re.search(r'개정|amendment|수정|변경|개정됨', text, re.IGNORECASE):
                flags["is_amendment"] = True
            
            # 부칙/별지 여부
            if re.search(r'부칙|별지|appendix|부록|첨부서류', text, re.IGNORECASE):
                flags["is_appendix"] = True
                
                # 부칙/별지 유형 판별
                if re.search(r'부칙|부칙규정', text):
                    flags["appendix_type"] = "appendix"
                elif re.search(r'별지|별표|별첨', text):
                    flags["appendix_type"] = "attachment"
                else:
                    flags["appendix_type"] = "appendix"
            
            # 첨부 여부
            if re.search(r'첨부|attachment|첨부서류|첨부파일', text, re.IGNORECASE):
                flags["is_attachment"] = True
            
            self.logger.debug(f"상태 플래그 감지 완료: {flags}")
            return flags
            
        except Exception as e:
            self.logger.error(f"상태 플래그 감지 중 오류: {e}")
            return flags
    
    def scan_line(self, line: str) -> List[Dict[str, Any]]:
        """
        한 라인에서 모든 헤더 패턴을 스캔
        
        Args:
            line: 스캔할 라인
            
        Returns:
            발견된 패턴들의 리스트
        """
        patterns_found = []
        
        try:
            for compiled_pattern, pattern_type, description in self.compiled_patterns:
                matches = compiled_pattern.finditer(line)
                
                for match in matches:
                    # 상세 로그 추가
                    self.logger.debug(f"🔍 패턴 매칭 발견: {pattern_type} - '{match.group()}' (라인: {line[:50]}...)")
                    self.logger.debug(f"   📍 위치: {match.start()}-{match.end()}, 그룹: {match.groups()}")
                    
                    pattern_info = {
                        "type": pattern_type,
                        "description": description,
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "groups": match.groups(),
                        "full_match": match.group(0)
                    }
                    
                    # 특별한 패턴별 추가 정보 설정
                    if pattern_type == "article" and match.groups()[1]:
                        # 의조인 경우
                        pattern_info["sub_number"] = match.groups()[1]
                        pattern_info["is_sub_article"] = True
                        self.logger.debug(f"   ✅ 의조 패턴 감지: {match.groups()[1]}")
                    elif pattern_type == "paragraph" and "①" in match.group():
                        # 원형 숫자인 경우
                        pattern_info["circle_number"] = match.group()
                        self.logger.debug(f"   ✅ 원형 숫자 패턴 감지: {match.group()}")
                    
                    # 상태 플래그 감지 및 추가
                    status_flags = self._detect_status_flags(line)
                    pattern_info["status_flags"] = status_flags
                    if any(status_flags.values()):
                        self.logger.debug(f"   🏷️ 상태 플래그 감지: {status_flags}")
                    
                    patterns_found.append(pattern_info)
            
            # 위치 순으로 정렬
            patterns_found.sort(key=lambda x: x["start"])
            
            self.logger.debug(f"📊 라인 스캔 완료: {len(patterns_found)}개 패턴 발견")
            return patterns_found
            
        except Exception as e:
            self.logger.error(f"라인 스캔 중 오류: {e}")
            return []
    
    def scan_multiple_lines(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        여러 라인에서 패턴 스캔
        
        Args:
            lines: 스캔할 라인들의 리스트
            
        Returns:
            모든 라인의 패턴 정보를 포함한 리스트
        """
        all_patterns = []
        
        for line_num, line in enumerate(lines):
            line_patterns = self.scan_line(line)
            
            for pattern in line_patterns:
                pattern["line_number"] = line_num
                pattern["line_text"] = line
                all_patterns.append(pattern)
        
        self.logger.info(f"전체 라인 스캔 완료: {len(all_patterns)}개 패턴 발견")
        return all_patterns
    
    def get_pattern_summary(self, patterns: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        패턴 타입별 개수 요약
        
        Args:
            patterns: 발견된 패턴들의 리스트
            
        Returns:
            타입별 개수 요약
        """
        summary = {}
        
        for pattern in patterns:
            pattern_type = pattern["type"]
            summary[pattern_type] = summary.get(pattern_type, 0) + 1
        
        return summary
