import threading
import logging
from typing import Dict, Any, Optional
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableLambda
from .session_manager import SessionManager

# 로깅 설정
logger = logging.getLogger("async-summarizer")

class AsyncSummarizer:
    """비동기 대화 요약 처리 클래스"""
    
    def __init__(self, 
                 session_manager: SessionManager,
                 ollama_endpoint: str = "http://ollama:11434",
                 default_model: str = "gemma3:12b"):
        """
        비동기 요약 처리기 초기화
        
        Args:
            session_manager: 세션 관리자 인스턴스
            ollama_endpoint: Ollama API 엔드포인트
            default_model: 기본 LLM 모델
        """
        self.session_manager = session_manager
        self.ollama_endpoint = ollama_endpoint
        self.default_model = default_model
        
        # 요약 프롬프트 템플릿
        self.summary_prompt = self._create_summary_prompt()
        
        # LangChain 체인 구성
        self._setup_chains()
        
        logger.info(f"AsyncSummarizer 초기화 완료: endpoint={ollama_endpoint}, model={default_model}")
    
    def _create_summary_prompt(self) -> PromptTemplate:
        """요약 프롬프트 템플릿을 생성합니다."""
        template = """아래 대화 내용을 1000자 이내로 간결하게 요약해주세요.
요약은 대화의 주요 주제, 질문, 해결책을 포함해야 합니다.
중요한 정보나 결정사항이 있다면 반드시 포함해주세요.

{conversation}

요약:"""
        
        return PromptTemplate(
            input_variables=["conversation"],
            template=template
        )
    
    def _setup_chains(self):
        """LangChain 체인을 설정합니다."""
        # 요약 함수
        def summarize(inputs):
            # Ollama LLM 초기화
            llm = Ollama(
                base_url=self.ollama_endpoint,
                model=inputs.get("model", self.default_model),
                temperature=0.1  # 요약은 낮은 온도로 사실적으로
            )
            
            # LLM 체인 생성
            chain = LLMChain(
                llm=llm,
                prompt=self.summary_prompt,
                verbose=False
            )
            
            # 요약 수행
            result = chain.invoke({"conversation": inputs["conversation"]})
            return {"summary": result["text"]}
        
        # 체인 구성
        self.summary_chain = RunnableLambda(summarize)
    
    def process_summarization_async(self, session_id: str, model: str = None) -> None:
        """세션의 대화 기록을 비동기적으로 요약 처리합니다."""
        try:
            # 세션 데이터 로드
            session_data = self.session_manager.load_session(session_id)
            
            # 요약 필요 여부 확인
            if not self.session_manager.needs_summarization(session_data):
                logger.info(f"세션 {session_id}는 요약이 필요하지 않습니다")
                return
            
            # 요약 처리
            self._process_summarization(session_data, model)
            
            logger.info(f"세션 {session_id} 요약 처리 완료")
        except Exception as e:
            logger.error(f"비동기 요약 처리 중 오류 발생: {str(e)}", exc_info=True)
    
    def _process_summarization(self, session_data: Dict[str, Any], model: str = None) -> None:
        """세션 데이터의 대화 기록을 요약합니다."""
        while self.session_manager.needs_summarization(session_data):
            # 요약할 청크 가져오기
            chunk = self.session_manager.get_chunk_for_summary(session_data)
            
            if not chunk:
                logger.info("요약할 대화가 충분하지 않습니다")
                break
            
            # 요약을 위한 대화 포맷팅
            conversation_text = self.session_manager.format_messages_for_summary(chunk)
            
            try:
                # LangChain을 사용하여 대화 요약
                logger.info(f"대화 요약 시작: {len(chunk)}턴, 모델: {model or self.default_model}")
                
                # 체인 실행 입력 준비
                chain_input = {
                    "conversation": conversation_text,
                    "model": model or self.default_model
                }
                
                # 체인 실행
                result = self.summary_chain.invoke(chain_input)
                summary = result.get("summary", "요약 실패")
                
                # 세션 업데이트
                session_data = self.session_manager.update_with_summary(session_data, summary)
                logger.info(f"대화 요약 완료: {len(summary)} 문자")
                
                # 업데이트된 세션 저장
                self.session_manager.save_session(session_data)
            except Exception as e:
                logger.error(f"대화 요약 중 오류 발생: {str(e)}")
                # 오류 발생 시 기본 요약으로 대체
                fallback_summary = f"이전 대화: {len(chunk)}턴의 대화가 있었습니다."
                session_data = self.session_manager.update_with_summary(session_data, fallback_summary)
                self.session_manager.save_session(session_data)
    
    def start_async_summarization(self, session_id: str, model: str = None) -> None:
        """비동기 요약 처리를 시작합니다."""
        thread = threading.Thread(
            target=self.process_summarization_async,
            args=(session_id, model),
            daemon=True
        )
        thread.start()
        logger.info(f"세션 {session_id}의 비동기 요약 처리 시작")
        return 