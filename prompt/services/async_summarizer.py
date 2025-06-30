import threading
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableLambda
from .session_manager import SessionManager
# 현재 미사용 코드 최근 5개 대화만 사용하기 때문
# 로깅 설정
logger = logging.getLogger("async-summarizer")

class AsyncSummarizer:
    """비동기 대화 요약 처리 클래스"""
    
    def __init__(self, 
                 session_manager: SessionManager,
                 ollama_endpoint: str = "http://ollama:11434",
                 default_model: str = "gemma3:12b",
                 max_workers: int = 3,  # 병렬 처리를 위한 최대 워커 수
                 batch_size: int = 3):  # 배치 처리를 위한 크기
        """
        비동기 요약 처리기 초기화
        
        Args:
            session_manager: 세션 관리자 인스턴스
            ollama_endpoint: Ollama API 엔드포인트
            default_model: 기본 LLM 모델
            max_workers: 병렬 처리를 위한 최대 워커 수
            batch_size: 배치 처리를 위한 크기
        """
        self.session_manager = session_manager
        self.ollama_endpoint = ollama_endpoint
        self.default_model = default_model
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # 요약 프롬프트 템플릿
        self.summary_prompt = self._create_summary_prompt()
        self.meta_summary_prompt = self._create_meta_summary_prompt()
        
        # LangChain 체인 구성
        self._setup_chains()
        
        logger.info(f"AsyncSummarizer 초기화 완료: endpoint={ollama_endpoint}, model={default_model}, max_workers={max_workers}, batch_size={batch_size}")
    
    def _create_summary_prompt(self) -> PromptTemplate:
        """요약 프롬프트 템플릿을 생성합니다."""
        template = """아래 대화 내용을 원본 길이의 약 절반 정도로 간결하게 요약해주세요.
요약은 대화의 주요 주제, 질문, 해결책을 포함해야 합니다.
중요한 정보나 결정사항이 있다면 반드시 포함해주세요.

{conversation}

요약:"""
        
        return PromptTemplate(
            input_variables=["conversation"],
            template=template
        )
    
    def _create_meta_summary_prompt(self) -> PromptTemplate:
        """메타 요약 프롬프트 템플릿을 생성합니다."""
        template = """다음은 이미 요약된 대화 내용들입니다. 이 요약들을 더 간결하게 통합해주세요.
요약은 원본 요약 길이의 약 절반 정도로 줄이되, 핵심 정보는 유지해주세요.
중요한 주제, 결정사항, 정보는 반드시 포함해야 합니다.

{summaries}

통합 요약:"""
        
        return PromptTemplate(
            input_variables=["summaries"],
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
        
        # 메타 요약 함수
        def meta_summarize(inputs):
            # Ollama LLM 초기화
            llm = Ollama(
                base_url=self.ollama_endpoint,
                model=inputs.get("model", self.default_model),
                temperature=0.1
            )
            
            # LLM 체인 생성
            chain = LLMChain(
                llm=llm,
                prompt=self.meta_summary_prompt,
                verbose=False
            )
            
            # 요약 수행
            result = chain.invoke({"summaries": inputs["summaries"]})
            return {"summary": result["text"]}
        
        # 체인 구성
        self.summary_chain = RunnableLambda(summarize)
        self.meta_summary_chain = RunnableLambda(meta_summarize)
    
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
            start_time = time.time()
            self._process_summarization(session_data, model)
            
            logger.info(f"세션 {session_id} 요약 처리 완료: {time.time() - start_time:.2f}초 소요")
        except Exception as e:
            logger.error(f"비동기 요약 처리 중 오류 발생: {str(e)}", exc_info=True)
    
    def _process_summarization(self, session_data: Dict[str, Any], model: str = None) -> None:
        """세션 데이터의 대화 기록을 요약합니다."""
        summarization_start = datetime.now()
        chunks_processed = 0
        
        # 1. 일반 요약 처리
        while self.session_manager.needs_summarization(session_data):
            # 요약할 청크 수집
            chunks_to_summarize = []
            for _ in range(self.batch_size):
                if self.session_manager.needs_summarization(session_data):
                    chunk = self.session_manager.get_chunk_for_summary(session_data)
                    if chunk:
                        chunks_to_summarize.append(chunk)
                    else:
                        break
                else:
                    break
            
            if not chunks_to_summarize:
                logger.info("요약할 대화가 충분하지 않습니다")
                break
            
            # 배치 처리 또는 병렬 처리 결정
            if len(chunks_to_summarize) == 1:
                # 단일 청크는 직접 처리
                self._summarize_single_chunk(session_data, chunks_to_summarize[0], model)
                chunks_processed += 1
            else:
                # 여러 청크는 병렬 처리
                self._summarize_chunks_parallel(session_data, chunks_to_summarize, model)
                chunks_processed += len(chunks_to_summarize)
        
        # 2. 메타 요약 필요 여부 확인 및 처리
        if hasattr(self.session_manager, 'needs_meta_summarization') and self.session_manager.needs_meta_summarization(session_data):
            logger.info(f"요약 청크가 많아 메타 요약 시작: {len(session_data['summarized_chunks'])}개 요약")
            session_data = self._create_meta_summary(session_data, model)
            self.session_manager.save_session(session_data)
            logger.info("메타 요약 완료 및 세션 저장됨")
        
        # 총 처리 시간 로깅
        total_time = (datetime.now() - summarization_start).total_seconds()
        logger.info(f"요약 처리 완료: {chunks_processed}개 청크, 총 {total_time:.2f}초 소요")
    
    def _summarize_single_chunk(self, session_data: Dict[str, Any], chunk: List[Dict[str, Any]], model: str = None) -> None:
        """단일 대화 청크를 요약합니다."""
        # 대화 텍스트 길이 계산
        conversation_text = self.session_manager.format_messages_for_summary(chunk)
        original_length = len(conversation_text)
        
        try:
            # LangChain을 사용하여 대화 요약
            logger.info(f"대화 요약 시작: {len(chunk)}턴, {original_length}자")
            
            # 체인 실행 입력 준비
            chain_input = {
                "conversation": conversation_text,
                "model": model or self.default_model
            }
            
            # 체인 실행
            result = self.summary_chain.invoke(chain_input)
            summary = result.get("summary", "요약 실패")
            
            # 요약 결과 길이 확인
            summary_length = len(summary)
            logger.info(f"대화 요약 완료: 원본 {original_length}자 → 요약 {summary_length}자 (비율: {summary_length/original_length:.2f})")
            
            # 세션 업데이트
            session_data = self.session_manager.update_with_summary(session_data, summary)
            
            # 업데이트된 세션 저장
            self.session_manager.save_session(session_data)
            
        except Exception as e:
            logger.error(f"대화 요약 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 요약으로 대체
            fallback_summary = f"이전 대화: {len(chunk)}턴의 대화가 있었습니다."
            session_data = self.session_manager.update_with_summary(session_data, fallback_summary)
            self.session_manager.save_session(session_data)
    
    def _summarize_chunks_parallel(self, session_data: Dict[str, Any], chunks: List[List[Dict[str, Any]]], model: str = None) -> None:
        """여러 대화 청크를 병렬로 요약합니다."""
        logger.info(f"병렬 요약 시작: {len(chunks)}개 청크")
        
        # 각 청크의 대화 텍스트 준비
        conversation_texts = [self.session_manager.format_messages_for_summary(chunk) for chunk in chunks]
        
        # 병렬 처리를 위한 함수
        def process_chunk(idx, text):
            try:
                original_length = len(text)
                logger.info(f"청크 {idx+1} 요약 시작: {len(chunks[idx])}턴, {original_length}자")
                
                # 체인 실행 입력 준비
                chain_input = {
                    "conversation": text,
                    "model": model or self.default_model
                }
                
                # 체인 실행
                result = self.summary_chain.invoke(chain_input)
                summary = result.get("summary", f"요약 실패 (청크 {idx+1})")
                
                # 요약 결과 길이 확인
                summary_length = len(summary)
                logger.info(f"청크 {idx+1} 요약 완료: 원본 {original_length}자 → 요약 {summary_length}자 (비율: {summary_length/original_length:.2f})")
                
                return summary
            except Exception as e:
                logger.error(f"청크 {idx+1} 요약 중 오류 발생: {str(e)}")
                return f"이전 대화: {len(chunks[idx])}턴의 대화가 있었습니다."
        
        # 병렬 처리 실행
        summaries = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
            futures = [executor.submit(process_chunk, i, text) for i, text in enumerate(conversation_texts)]
            for future in futures:
                summaries.append(future.result())
        
        # 모든 요약 결과를 세션에 적용
        for i, summary in enumerate(summaries):
            session_data = self.session_manager.update_with_summary(session_data, summary)
            # 중간 저장 (마지막 청크 제외)
            if i < len(summaries) - 1:
                self.session_manager.save_session(session_data)
        
        # 최종 세션 저장
        self.session_manager.save_session(session_data)
        logger.info(f"병렬 요약 완료: {len(chunks)}개 청크")
    
    def _create_meta_summary(self, session_data: Dict[str, Any], model: str = None) -> Dict[str, Any]:
        """여러 요약을 하나의 메타 요약으로 통합합니다."""
        # 기존 요약 청크들을 하나의 텍스트로 결합
        summaries_text = ""
        total_chars = 0
        
        for idx, summary in enumerate(session_data["summarized_chunks"], 1):
            summaries_text += f"요약 {idx}: {summary}\n\n"
            total_chars += len(summary)
        
        try:
            # LLM을 사용하여 메타 요약 생성
            logger.info(f"메타 요약 시작: {len(session_data['summarized_chunks'])}개 요약, 총 {total_chars}자")
            
            # 체인 실행 입력 준비
            chain_input = {
                "summaries": summaries_text,
                "model": model or self.default_model
            }
            
            # 체인 실행
            result = self.meta_summary_chain.invoke(chain_input)
            meta_summary = result.get("summary", "메타 요약 실패")
            
            # 메타 요약 길이 확인
            meta_summary_length = len(meta_summary)
            logger.info(f"메타 요약 완료: 원본 {total_chars}자 → 메타 요약 {meta_summary_length}자 (비율: {meta_summary_length/total_chars:.2f})")
            
            # 메타 요약으로 세션 업데이트
            session_data["summarized_chunks"] = [f"통합 요약: {meta_summary}"]
            
            return session_data
            
        except Exception as e:
            logger.error(f"메타 요약 생성 중 오류 발생: {str(e)}")
            # 오류 발생 시 가장 최근 요약만 유지
            session_data["summarized_chunks"] = [session_data["summarized_chunks"][-1]]
            return session_data
    
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