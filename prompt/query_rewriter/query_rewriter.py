import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("query-rewriter")

class QueryRewriter:
    """Query Rewrite 시스템 - 대화 컨텍스트를 기반으로 사용자 질문을 개선"""
    
    def __init__(self, 
                 vllm_endpoint: str = "http://vllm:8000",
                 default_model: str = "/app/models/mistralai/Mistral-7B-Instruct-v0.2",
                 temperature: float = 0.3,
                 max_history_turns: int = 5):
        """
        Query Rewriter 초기화
        
        Args:
            ollama_endpoint: Ollama API 엔드포인트
            default_model: 기본 LLM 모델
            temperature: 생성 온도 (낮을수록 일관성 높음)
            max_history_turns: 참조할 최대 대화 턴 수
        """
        self.vllm_endpoint = vllm_endpoint
        self.default_model = default_model
        self.temperature = temperature
        self.max_history_turns = max_history_turns
        
        # 프롬프트 템플릿 로드
        self.prompt_template = self._load_prompt_template()
        
        logger.info(f"QueryRewriter 초기화 완료: endpoint={vllm_endpoint}, model={default_model}, max_history_turns={max_history_turns}")
    
    def _load_prompt_template(self) -> str:
        """Query Rewrite 프롬프트 템플릿을 로드합니다."""
        template_path = os.path.join(os.path.dirname(__file__), "templates", "query_rewrite.txt")
        
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
                logger.debug(f"Query Rewrite 템플릿 로드 완료: {len(template)} 문자")
                return template
        except FileNotFoundError:
            logger.warning(f"Query Rewrite 템플릿 파일을 찾을 수 없습니다: {template_path}")
            # 기본 템플릿 반환
            return self._get_default_template()
    
    def _get_default_template(self) -> str:
        """기본 Query Rewrite 프롬프트 템플릿을 반환합니다."""
        return """당신은 사용자의 질문을 개선하는 전문가입니다. 

주어진 대화 기록을 바탕으로 사용자의 최신 질문을 더 명확하고 구체적으로 재작성해주세요.

## 대화 기록 (최근 {max_history_turns}개 턴):
{conversation_history}

## 사용자의 최신 질문:
{current_query}

## 지침:
1. 대화 맥락을 고려하여 질문의 의도를 파악하세요
2. 대명사나 생략된 부분을 명확하게 표현하세요
3. 질문을 더 구체적이고 검색에 적합하게 만드세요
4. 원래 질문의 핵심 의도를 유지하세요
5. 불필요한 정보는 제거하고 핵심만 남기세요

## 개선된 질문:
"""
    
    def rewrite_query(self, 
                     current_query: str, 
                     session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 질문을 대화 컨텍스트를 기반으로 재작성합니다.
        
        Args:
            current_query: 현재 사용자 질문
            session_data: 세션 데이터 (대화 기록 포함)
            model: 사용할 모델 (기본값 사용 시 None)
            
        Returns:
            Dict containing:
            - original_query: 원본 질문
            - rewritten_query: 재작성된 질문
            - confidence: 개선 신뢰도 (0.0-1.0)
            - reasoning: 재작성 이유
            - used_history: 사용된 대화 기록 수
        """
        try:
            logger.info(f"Query Rewrite 시작: 원본 질문='{current_query[:50]}...'")
            
            # 대화 기록 추출
            conversation_history = self._extract_conversation_history(session_data)
            
            # 대화 기록이 없거나 질문이 이미 명확한 경우 원본 반환
            if not conversation_history or self._is_query_already_clear(current_query):
                logger.info("대화 기록이 없거나 질문이 이미 명확함 - 원본 반환")
                return {
                    "original_query": current_query,
                    "rewritten_query": current_query,
                    "confidence": 1.0,
                    "reasoning": "대화 기록이 없거나 질문이 이미 명확함",
                    "used_history": 0
                }
            
            # 프롬프트 구성
            prompt = self._build_prompt(current_query, conversation_history)
            
            # LLM 호출 (항상 VLLM용 모델 사용)
            rewritten_query = self._call_llm(prompt, self.default_model)
            
            # 결과 검증 및 후처리
            rewritten_query = self._post_process_query(rewritten_query, current_query)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(current_query, rewritten_query, conversation_history)
            
            result = {
                "original_query": current_query,
                "rewritten_query": rewritten_query,
                "confidence": confidence,
                "reasoning": f"최근 {len(conversation_history)}개 대화 턴을 기반으로 질문을 개선했습니다",
                "used_history": len(conversation_history)
            }
            
            logger.info(f"Query Rewrite 완료: '{current_query[:30]}...' → '{rewritten_query[:30]}...' (신뢰도: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Query Rewrite 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 반환
            return {
                "original_query": current_query,
                "rewritten_query": current_query,
                "confidence": 0.0,
                "reasoning": f"오류 발생으로 원본 사용: {str(e)}",
                "used_history": 0
            }
    
    def _extract_conversation_history(self, session_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """세션 데이터에서 최근 대화 기록을 추출합니다."""
        history = session_data.get("history", [])
        
        # 마지막 사용자 메시지 제외 (현재 질문이므로)
        if history and history[-1].get("role") == "user":
            history = history[:-1]
        
        # 최근 N개 턴만 사용
        recent_history = history[-self.max_history_turns * 2:]  # 사용자+봇 메시지 쌍
        
        conversation_history = []
        for i in range(0, len(recent_history), 2):
            if i + 1 < len(recent_history):
                user_msg = recent_history[i].get("message", "")  # "content" → "message"로 수정
                bot_msg = recent_history[i + 1].get("message", "")
                conversation_history.append({
                    "user": user_msg,
                    "bot": bot_msg
                })
        
        logger.debug(f"대화 기록 추출: {len(conversation_history)}개 턴 (현재 질문 제외)")
        return conversation_history
    
    def _is_query_already_clear(self, query: str) -> bool:
        """질문이 이미 명확한지 판단합니다."""
        # 명확한 질문의 특징들
        clear_indicators = [
            "무엇", "어떻게", "언제", "어디서", "누가", "왜",
            "what", "how", "when", "where", "who", "why",
            "설명", "방법", "과정", "단계", "예시", "사례"
        ]
        
        # 질문이 너무 짧거나 긴 경우
        if len(query.strip()) < 3 or len(query.strip()) > 200:
            return True
        
        # 명확한 질문어가 포함된 경우
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in clear_indicators):
            return True
        
        return False
    
    def _build_prompt(self, current_query: str, conversation_history: List[Dict[str, str]]) -> str:
        """Query Rewrite 프롬프트를 구성합니다."""
        # 대화 기록을 텍스트로 변환
        history_text = ""
        for i, turn in enumerate(conversation_history, 1):
            history_text += f"턴 {i}:\n"
            history_text += f"사용자: {turn['user']}\n"
            history_text += f"봇: {turn['bot']}\n\n"
        
        # 프롬프트 템플릿에 값 삽입
        prompt = self.prompt_template.format(
            max_history_turns=self.max_history_turns,
            conversation_history=history_text.strip(),
            current_query=current_query
        )
        
        return prompt
    
    def _call_llm(self, prompt: str, model: str) -> str:
        """LLM을 호출하여 질문을 재작성합니다."""
        try:
            # 디버깅: 실제 요청 정보 로깅
            request_data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": 200
            }
            logger.info(f"VLLM 요청 URL: {self.vllm_endpoint}/v1/chat/completions")
            logger.info(f"VLLM 요청 모델: {model}")
            logger.info(f"VLLM 요청 데이터: {request_data}")
            
            response = requests.post(
                f"{self.vllm_endpoint}/v1/chat/completions",
                json=request_data,
                timeout=30
            )

            
            if response.status_code != 200:
                raise Exception(f"LLM API 오류: {response.status_code}")
            
            result = response.json()
            rewritten_query = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # 응답에서 불필요한 부분 제거
            rewritten_query = self._clean_response(rewritten_query)
            
            return rewritten_query
            
        except Exception as e:
            logger.error(f"LLM 호출 중 오류: {str(e)}")
            raise
    
    def _clean_response(self, response: str) -> str:
        """LLM 응답을 정리합니다."""
        # 줄바꿈 제거
        response = response.replace('\n', ' ').replace('\r', ' ')
        
        # 여러 공백을 하나로
        response = ' '.join(response.split())
        
        # 따옴표 제거
        response = response.strip('"\'')
        
        return response
    
    def _post_process_query(self, rewritten_query: str, original_query: str) -> str:
        """재작성된 질문을 후처리합니다."""
        # 빈 응답이나 너무 짧은 응답 처리
        if not rewritten_query or len(rewritten_query.strip()) < 3:
            return original_query
        
        # 원본과 너무 다른 경우 원본 사용
        if len(rewritten_query) > len(original_query) * 3:
            logger.warning("재작성된 질문이 너무 김 - 원본 사용")
            return original_query
        
        return rewritten_query
    
    def _calculate_confidence(self, 
                            original_query: str, 
                            rewritten_query: str, 
                            conversation_history: List[Dict[str, str]]) -> float:
        """재작성 신뢰도를 계산합니다."""
        # 기본 신뢰도
        confidence = 0.5
        
        # 대화 기록이 많을수록 신뢰도 증가
        if conversation_history:
            confidence += min(len(conversation_history) * 0.1, 0.3)
        
        # 질문이 개선되었는지 확인
        if len(rewritten_query) > len(original_query) * 1.2:
            confidence += 0.2
        
        # 명확한 질문어가 포함되었는지 확인
        clear_indicators = ["무엇", "어떻게", "언제", "어디서", "누가", "왜"]
        if any(indicator in rewritten_query for indicator in clear_indicators):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_rewrite_stats(self) -> Dict[str, Any]:
        """Query Rewrite 통계를 반환합니다."""
        return {
            "model": self.default_model,
            "temperature": self.temperature,
            "max_history_turns": self.max_history_turns,
            "endpoint": self.vllm_endpoint
        } 