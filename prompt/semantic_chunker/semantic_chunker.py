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
    
    def select_relevant_history(self, current_query: str, session_data: Dict[str, Any]) -> List[int]:
        """관련 히스토리 선택 (턴 번호만 반환)"""
        start_time = time.time()
        
        try:
            history = session_data.get("history", [])
            if not history:
                logger.info("히스토리가 비어있음")
                return []
            
            # 1. 히스토리 전처리 (최근 5턴으로 제한)
            # 마지막 질의는 무조건 관련이므로 시멘틱청커에서 제외
            if len(history) > 1 and history[-1]["role"] == "user":
                history_without_last_query = history[:-1]
            else:
                history_without_last_query = history
            
            # 최근 5턴(10개 메시지)으로 제한
            recent_history = self._truncate_history(history_without_last_query, max_turns=5)
            
            # 최근 5턴에서 사용자 질의만 추출 (성능 향상을 위해)
            user_queries = [msg for msg in recent_history if msg["role"] == "user"]
            
            # 2. 프롬프트 생성 (사용자 질의만)
            prompt = self._build_prompt_with_user_queries(current_query, user_queries)
            
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
            
            # 4. 결과 파싱 (턴 번호만 반환)
            selected_turn_numbers = self._parse_turn_numbers(response)
            
            total_time = time.time() - start_time
            logger.info(f"시멘틱 청커 완료: 턴 번호 {selected_turn_numbers}, 총 시간: {total_time:.3f}초 (LLM: {llm_time:.3f}초)")
            
            return selected_turn_numbers
            
        except Exception as e:
            logger.error(f"시멘틱 청커 오류: {e}")
            return [3, 4, 5]  # 기본값: 최근 3턴
    
    def _truncate_history(self, history: List[Dict[str, Any]], max_turns: int = 5) -> List[Dict[str, Any]]:
        """히스토리 길이 제한"""
        if len(history) <= max_turns * 2:
            return history
        return history[-(max_turns * 2):]
    
    def _build_prompt_with_user_queries(self, current_query: str, user_queries: List[Dict[str, Any]]) -> str:
        """사용자 질의만으로 프롬프트 생성"""
        formatted = []
        for i, msg in enumerate(user_queries, 1):
            formatted.append(f"[턴 {i}] 사용자: {msg['message']}")
        
        numbered_history = "\n".join(formatted)
        return self.template.format(
            current_query=current_query,
            numbered_history=numbered_history
        )

    def _parse_turn_numbers(self, response: str) -> List[int]:
        """턴 번호만 파싱하여 반환"""
        try:
            numbers = re.findall(r'\d+', response)
            # 중복 제거
            numbers = list(dict.fromkeys(numbers))
            
            # 유효한 턴 번호만 필터링 (1-5 범위)
            valid_turns = []
            for num in numbers:
                turn_num = int(num)
                if 1 <= turn_num <= 5:  # 5턴 제한
                    valid_turns.append(turn_num)
            
            logger.info(f"선택된 턴 번호: {valid_turns}")
            return valid_turns
            
        except Exception as e:
            logger.warning(f"턴 번호 파싱 실패: {e}")
            return [3, 4, 5]  # 기본값: 최근 3턴

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
    
    def _parse_response(self, response: str, processed_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """응답 파싱 (제한된 히스토리 기준)"""
        try:
            # 145번째 줄 수정
            numbers = re.findall(r'\d+', response)
            # 중복 제거 (VLLM이 같은 턴 번호를 여러 형태로 반환할 수 있음)
            numbers = list(dict.fromkeys(numbers))  # 순서 유지하면서 중복 제거
            
            # 턴 번호를 히스토리 인덱스로 변환
            # 턴 1 = 히스토리 인덱스 0, 1 (사용자 메시지, 봇 메시지)
            # 턴 2 = 히스토리 인덱스 2, 3 (사용자 메시지, 봇 메시지)
            # 턴 3 = 히스토리 인덱스 4, 5 (사용자 메시지, 봇 메시지)
            selected_messages = []
            
            for turn_num in numbers:
                turn_index = int(turn_num)
                # 턴 번호를 제한된 히스토리 인덱스로 변환
                user_index = (turn_index - 1) * 2
                bot_index = user_index + 1
                
                # 사용자 메시지 추가
                if 0 <= user_index < len(processed_history):
                    selected_messages.append(processed_history[user_index])
                
                # 봇 메시지 추가 (있는 경우)
                if 0 <= bot_index < len(processed_history):
                    selected_messages.append(processed_history[bot_index])
            
            if not selected_messages:
                logger.warning(f"유효한 턴 번호가 없음: {numbers}")
                return processed_history[-3:]  # 기본값: 제한된 히스토리에서 최근 3턴
            
            logger.info(f"턴 번호 {numbers} → 선택된 메시지 {len(selected_messages)}개 (제한된 히스토리 기준)")
            return selected_messages
            
        except Exception as e:
            logger.warning(f"응답 파싱 실패: {e}")
            return processed_history[-3:] 