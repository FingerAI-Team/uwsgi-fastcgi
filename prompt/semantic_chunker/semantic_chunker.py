import os
import re
import time
import json
import logging
import requests
from typing import Dict, List, Any

logger = logging.getLogger("semantic-chunker")

class SemanticChunker:
    """시멘틱 청커 - 현재 질의와 관련된 히스토리만 선별"""
    
    def __init__(self, 
                 vllm_endpoint: str = "http://vllm:8000",
                 model: str = "/app/models/mistralai/Mistral-7B-Instruct-v0.2",
                 template_path: str = None):
        
        self.vllm_endpoint = vllm_endpoint
        self.model = model
        
        # 템플릿 로드
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), "templates", "history_selection.txt")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
    
    def select_relevant_history(self, current_query: str, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """현재 질의와 관련된 히스토리만 선별"""
        
        start_time = time.time()
        logger.info(f"시멘틱 청커 시작: 질의='{current_query[:50]}...'")
        
        try:
            history = session_data.get("history", [])
            if not history:
                logger.info("히스토리가 없어 빈 리스트 반환")
                return []
            
            # 1. 히스토리 전처리 (최대 10턴으로 제한)
            processed_history = self._truncate_history(history, max_turns=10)
            
            # 2. 프롬프트 생성
            prompt = self._build_prompt(current_query, processed_history)
            
            # 프롬프트 로깅 추가
            logger.info("=== 시멘틱 청커 VLLM 프롬프트 ===")
            logger.info(f"전송할 프롬프트:\n{prompt}")
            logger.info("================================")
            
            # 3. LLM 호출
            llm_start = time.time()
            response = self._call_vllm(prompt)
            llm_time = time.time() - llm_start
            
            # 응답 로깅 추가
            logger.info("=== 시멘틱 청커 VLLM 응답 ===")
            logger.info(f"받은 응답: '{response}'")
            logger.info("=============================")
            
            # 4. 결과 파싱
            selected_history = self._parse_response(response, history)
            
            total_time = time.time() - start_time
            logger.info(f"시멘틱 청커 완료: {len(selected_history)}개 턴 선택, 총 시간: {total_time:.3f}초 (LLM: {llm_time:.3f}초)")
            
            return selected_history
            
        except Exception as e:
            logger.error(f"시멘틱 청커 오류: {e}")
            return history[-3:]  # 기본값: 최근 3턴
    
    def _truncate_history(self, history: List[Dict[str, Any]], max_turns: int = 10) -> List[Dict[str, Any]]:
        """히스토리 길이 제한"""
        if len(history) <= max_turns * 2:
            return history
        return history[-(max_turns * 2):]
    
    def _format_numbered_history(self, history: List[Dict[str, Any]]) -> str:
        """번호가 매겨진 대화 기록 포맷팅"""
        formatted = []
        turn_count = 0
        
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                turn_count += 1
                user_msg = history[i]['message']
                bot_msg = history[i + 1]['message']
                
                # 봇 답변의 번호 매기기를 구분하기 위해 들여쓰기와 다른 형식 사용
                formatted.append(f"[턴 {turn_count}] 사용자: {user_msg}")
                formatted.append(f"[턴 {turn_count}] 봇: {bot_msg}")
            else:
                turn_count += 1
                user_msg = history[i]['message']
                formatted.append(f"[턴 {turn_count}] 사용자: {user_msg}")
        
        return "\n".join(formatted)
    
    def _build_prompt(self, current_query: str, history: List[Dict[str, Any]]) -> str:
        """프롬프트 생성"""
        numbered_history = self._format_numbered_history(history)
        return self.template.format(
            current_query=current_query,
            numbered_history=numbered_history
        )
    
    def _call_vllm(self, prompt: str) -> str:
        """VLLM API 호출"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 100
        }
        
        logger.info(f"VLLM 요청 시작: endpoint={self.vllm_endpoint}, model={self.model}")
        logger.info(f"VLLM 요청 데이터: {payload}")
        
        response = requests.post(
            f"{self.vllm_endpoint}/v1/chat/completions",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info(f"VLLM 원본 응답: {result}")
            logger.info(f"VLLM 추출된 응답: '{content}'")
            return content
        else:
            logger.error(f"VLLM API 오류: status_code={response.status_code}, response={response.text}")
            raise Exception(f"VLLM API 오류: {response.status_code}")
    
    def _parse_response(self, response: str, original_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """응답 파싱"""
        try:
            numbers = re.findall(r'\d+', response)
            indices = [int(n) - 1 for n in numbers]  # 0-based 인덱스로 변환
            
            # 유효한 인덱스만 필터링
            valid_indices = [i for i in indices if 0 <= i < len(original_history)]
            
            if not valid_indices:
                return original_history[-3:]  # 기본값
            
            return [original_history[i] for i in valid_indices]
            
        except Exception as e:
            logger.warning(f"응답 파싱 실패: {e}")
            return original_history[-3:] 