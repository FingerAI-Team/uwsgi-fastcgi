import os
import json
import logging
import requests
import time
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
            vllm_endpoint: vLLM API 엔드포인트 (OpenAI 호환)
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
        
        # 성능 통계 초기화
        self.performance_stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'llm_calls': 0,
            'llm_total_time': 0.0,
            'llm_avg_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
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
            - processing_time: 처리 소요 시간
            - timing_breakdown: 단계별 시간 분석
        """
        start_time = time.time()
        logger.info(f"Query Rewrite 시작: 원본 질문='{current_query[:50]}...'")
        
        try:
            # 1단계: 대화 기록 추출
            history_start = time.time()
            conversation_history = self._extract_conversation_history(session_data)
            history_time = time.time() - history_start
            logger.info(f"대화 기록 추출 완료: {history_time:.3f}초")
            
            # 대화 기록이 없거나 질문이 이미 명확한 경우 원본 반환
            if not conversation_history or self._is_query_already_clear(current_query):
                total_time = time.time() - start_time
                self._update_performance_stats(total_time, 0.0, cache_hit=True)
                logger.info(f"Query Rewrite 완료 (원본 반환): {total_time:.3f}초")
                return {
                    "original_query": current_query,
                    "rewritten_query": current_query,
                    "confidence": 1.0,
                    "reasoning": "대화 기록이 없거나 질문이 이미 명확함",
                    "used_history": 0
                }
            
            # 2단계: 프롬프트 구성
            prompt_start = time.time()
            prompt = self._build_prompt(current_query, conversation_history)
            prompt_time = time.time() - prompt_start
            logger.info(f"프롬프트 구성 완료: {prompt_time:.3f}초")
            
            # 3단계: LLM 호출
            llm_start = time.time()
            rewritten_query = self._call_llm(prompt, self.default_model)
            llm_time = time.time() - llm_start
            logger.info(f"LLM 호출 완료: {llm_time:.3f}초")
            
            # 4단계: 후처리
            post_start = time.time()
            rewritten_query = self._post_process_query(rewritten_query, current_query)
            confidence = self._calculate_confidence(current_query, rewritten_query, conversation_history)
            post_time = time.time() - post_start
            logger.info(f"후처리 완료: {post_time:.3f}초")
            
            # 총 소요 시간
            total_time = time.time() - start_time
            
            # 성능 통계 업데이트
            self._update_performance_stats(total_time, llm_time, cache_hit=False)
            
            result = {
                "original_query": current_query,
                "rewritten_query": rewritten_query,
                "confidence": confidence,
                "reasoning": f"최근 {len(conversation_history)}개 대화 턴을 기반으로 질문을 개선했습니다",
                "used_history": len(conversation_history)
            }
            
            logger.info(f"Query Rewrite 완료: '{current_query[:30]}...' → '{rewritten_query[:30]}...' (신뢰도: {confidence:.2f}, 총 시간: {total_time:.3f}초)")
            logger.info(f"시간 분석: 히스토리={history_time:.3f}s, 프롬프트={prompt_time:.3f}s, LLM={llm_time:.3f}s, 후처리={post_time:.3f}s")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Query Rewrite 중 오류 발생: {str(e)} (소요 시간: {total_time:.3f}초)")
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
        current_turn = {"user": "", "bot": ""}
        
        for msg in recent_history:
            role = msg.get("role", "")
            message = msg.get("message", "")
            
            if role == "user":
                # 새로운 턴 시작
                if current_turn["user"]:  # 이전 턴이 있으면 저장
                    conversation_history.append(current_turn)
                current_turn = {"user": message, "bot": ""}
            elif role == "bot":
                # 봇 메시지 추가
                current_turn["bot"] = message
        
        # 마지막 턴 저장
        if current_turn["user"]:
            conversation_history.append(current_turn)
        
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
                "max_tokens": 100
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
            
            # VLLM 응답 로깅
            logger.info(f"VLLM 원본 응답: {result}")
            logger.info(f"VLLM 추출된 응답: '{rewritten_query}'")
            
            # 응답에서 불필요한 부분 제거
            rewritten_query = self._clean_response(rewritten_query)
            
            logger.info(f"VLLM 정리된 응답: '{rewritten_query}'")
            
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
            logger.warning("재작성된 질문이 너무 짧음 - 원본 사용")
            return original_query
        
        # 100자 제한
        if len(rewritten_query) > 100:
            logger.warning(f"재작성된 질문이 너무 김 ({len(rewritten_query)}자) - 원본 사용")
            return original_query
        
        logger.info(f"재작성된 질문 사용: '{rewritten_query}' ({len(rewritten_query)}자)")
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
    
    def _update_performance_stats(self, total_time: float, llm_time: float, cache_hit: bool = False):
        """성능 통계 업데이트"""
        self.performance_stats['total_calls'] += 1
        self.performance_stats['total_time'] += total_time
        
        if llm_time > 0:
            self.performance_stats['llm_calls'] += 1
            self.performance_stats['llm_total_time'] += llm_time
        
        if cache_hit:
            self.performance_stats['cache_hits'] += 1
        else:
            self.performance_stats['cache_misses'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.performance_stats.copy()
        if stats['total_calls'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['total_calls']
        if stats['llm_calls'] > 0:
            stats['llm_avg_time'] = stats['llm_total_time'] / stats['llm_calls']
        return stats
    
    def get_rewrite_stats(self) -> Dict[str, Any]:
        """Query Rewrite 통계를 반환합니다."""
        return {
            "performance_stats": self.get_performance_stats(),
            "model_info": {
                "endpoint": self.vllm_endpoint,
                "model": self.default_model,
                "temperature": self.temperature,
                "max_history_turns": self.max_history_turns
            }
        } 