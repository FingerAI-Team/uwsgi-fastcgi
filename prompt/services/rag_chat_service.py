import os
import json
import logging
import requests
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Generator
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from .session_manager import SessionManager

# 로깅 설정
logger = logging.getLogger("rag-chat-service")

class RagChatService:
    """RAG 챗봇 서비스 클래스"""
    
    def __init__(self, 
                 memory_dir: str = "./memory",
                 ollama_endpoint: str = "http://ollama:11434",
                 rag_endpoint: str = "http://nginx/rag",
                 reranker_endpoint: str = "http://nginx/reranker",
                 default_model: str = "gemma3:12b",
                 search_top: int = 100,
                 rerank_top: int = 10,
                 rerank_threshold: float = 0.1,
                 max_total_tokens: int = 10000,
                 max_context_tokens: int = 7500):
        """
        RAG 챗봇 서비스 초기화
        
        Args:
            memory_dir: 세션 메모리 저장 디렉토리
            ollama_endpoint: Ollama API 엔드포인트
            rag_endpoint: RAG 검색 엔드포인트
            reranker_endpoint: Reranker 엔드포인트
            default_model: 기본 LLM 모델
            search_top: RAG 검색 결과 수
            rerank_top: 재랭킹 결과 수
            rerank_threshold: 재랭킹 점수 임계치
            max_total_tokens: 최대 총 토큰 수
            max_context_tokens: 최대 컨텍스트 토큰 수
        """
        self.ollama_endpoint = ollama_endpoint
        self.rag_endpoint = rag_endpoint
        self.reranker_endpoint = reranker_endpoint
        self.default_model = default_model
        self.search_top = search_top
        self.rerank_top = rerank_top
        self.rerank_threshold = rerank_threshold
        
        # 세션 관리자 초기화
        self.session_manager = SessionManager(
            memory_dir=memory_dir,
            max_total_tokens=max_total_tokens,
            max_context_tokens=max_context_tokens
        )
        
        # 시스템 프롬프트 로드
        self.system_prompt = self._load_system_prompt()
        
        # LangChain LLM 초기화
        self.llm = Ollama(
            base_url=ollama_endpoint,
            model=default_model
        )
        
        # 체인 구성
        self._setup_chains()
        
        logger.info(f"RagChatService 초기화 완료: model={default_model}, ollama_endpoint={ollama_endpoint}, search_top={search_top}, rerank_top={rerank_top}")
    
    def _setup_chains(self):
        """LangChain 체인을 설정합니다."""
        # 세션 로드 및 사용자 메시지 추가 함수
        def load_session(inputs):
            session_id = inputs["session_id"]
            query = inputs["query"]
            logger.debug(f"세션 로드 및 사용자 메시지 추가: 세션={session_id}")
            return {
                "session_data": self.session_manager.add_user_message(session_id, query),
                **inputs  # 원래 입력 유지
            }
        
        # RAG 검색 수행 함수
        def perform_search(inputs):
            query = inputs["query"]
            kwargs = inputs.get("kwargs", {})
            logger.debug(f"RAG 검색 수행: 쿼리='{query[:30]}...'")
            search_results = self._perform_enhanced_search(query, **kwargs)
            return {
                "search_results": search_results,
                **inputs  # 원래 입력 유지
            }
        
        # 컨텍스트 포맷팅 함수
        def format_search_results(inputs):
            search_results = inputs["search_results"]
            logger.debug(f"검색 결과 포맷팅: {len(search_results)}개 문서")
            return {
                "rag_context": self.format_context(search_results),
                **inputs  # 원래 입력 유지
            }
        
        # 세션 저장 및 프롬프트 구성 함수
        def build_prompt(inputs):
            session_data = inputs["session_data"]
            query = inputs["query"]
            rag_context = inputs["rag_context"]
            
            # 세션 저장
            self.session_manager.save_session(session_data)
            
            # 프롬프트 구성
            prompt = self.session_manager.build_prompt_context(
                session_data,
                self.system_prompt,
                rag_context=rag_context,
                current_query=query
            )
            
            logger.debug(f"프롬프트 구성 완료: {len(prompt)} 문자")
            return {
                "prompt": prompt,
                **inputs  # 원래 입력 유지
            }
        
        # 체인 구성 (최신 LangChain 방식)
        self.rag_chain = (
            RunnableLambda(load_session)
            | RunnableLambda(perform_search)
            | RunnableLambda(format_search_results)
            | RunnableLambda(build_prompt)
        )
    
    def _load_system_prompt(self) -> str:
        """시스템 프롬프트를 로드합니다."""
        template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "rag_chat.txt")
        logger.debug(f"[시스템 프롬프트] 템플릿 파일 경로: {template_path}")
        
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
                logger.debug(f"[시스템 프롬프트] 템플릿 파일 로드 성공: {len(system_prompt)} 문자")
                return system_prompt
        except FileNotFoundError:
            logger.error(f"템플릿 파일을 찾을 수 없습니다: {template_path}")
            # 기본 시스템 프롬프트
            default_prompt = """당신은 도움이 되는 AI 어시스턴트입니다.
사용자의 질문에 친절하고 정확하게 답변해 주세요.
아래 제공된 문서 정보와 대화 기록을 참고하여 답변하되, 문서에 없는 내용은 솔직히 모른다고 말하세요.

질문에 답변할 때 다음 지침을 따르세요:
1. 문서 내용을 기반으로 정확하게 답변하세요.
2. 문서에 없는 내용은 추측하지 말고 솔직히 모른다고 말하세요.
3. 답변은 간결하고 명확하게 작성하세요.
4. 문서의 출처를 인용할 필요는 없습니다.
5. 사용자가 이해하기 쉽게 설명하세요.

문서에 제공된 메타데이터(작성자, 날짜, 링크 등)는 답변에 활용할 수 있습니다."""
            logger.debug(f"[시스템 프롬프트] 기본 프롬프트 사용: {len(default_prompt)} 문자")
            return default_prompt
    
    def _perform_enhanced_search(self, query: str, **kwargs) -> List[Dict]:
        """enhanced_search API를 통해 문서 검색 및 재랭킹을 수행합니다."""
        start_time = datetime.now()
        logger.info(f"[성능] Enhanced Search 시작: 쿼리='{query[:30]}...'")
        
        search_params = {
            "query": query,
            "top_m": self.search_top,  # RAG 검색 결과 수
            "top_n": self.rerank_top,  # Reranker 결과 수
            "threshold": self.rerank_threshold  # Reranker 점수 임계치
        }
        
        # 추가 검색 매개변수
        for param in ["domains", "domain", "author", "start_date", "end_date", "title", "info_filter", "tags_filter"]:
            if param in kwargs and kwargs[param]:
                search_params[param] = kwargs[param]
        
        try:
            logger.info(f"Enhanced Search 요청: {json.dumps(search_params, ensure_ascii=False)}")
            search_response = requests.post(
                f"{self.rag_endpoint.replace('/rag', '/prompt')}/enhanced_search", 
                json=search_params,
                timeout=30
            )
            
            if search_response.status_code != 200:
                logger.error(f"Enhanced Search 오류: {search_response.text}")
                return []
            
            search_results = search_response.json()
            results = search_results.get("results", [])
            
            logger.info(f"[성능] Enhanced Search 완료: {(datetime.now() - start_time).total_seconds():.3f}초, 결과 수: {len(results)}")
            return results
        except Exception as e:
            logger.error(f"Enhanced Search 중 오류 발생: {str(e)}")
            return []
    
    def format_context(self, documents: List[Dict]) -> str:
        """검색 결과를 컨텍스트 형식으로 포맷팅합니다."""
        logger.debug(f"[RAG 컨텍스트] 시작: {len(documents)}개 문서")
        
        if not documents:
            logger.info("검색 결과가 없어 일반 지식 기반 응답 모드로 전환합니다.")
            no_docs_context = """관련 문서를 찾을 수 없어 제가 학습해둔 일반 지식을 기반으로 답변하겠습니다.

참고사항:
1. 이 답변은 문서 검색 결과가 아닌 LLM의 일반 지식을 기반으로 합니다.
2. 최신 정보나 특정 통계 데이터가 필요한 질문은 정확성이 떨어질 수 있습니다.
3. 가능한 범위 내에서 도움이 되는 정보를 제공하겠습니다.
4. 정확한 정보가 필요하시면 질문을 더 구체적으로 해주시거나 다른 키워드로 시도해보세요.

중요: 응답 형식은 다음과 같이 작성해야 합니다:

<answer>
제공된 문서에는 이 질문에 대한 관련 정보가 없습니다. 하지만 일반적인 지식을 기반으로 답변해 드리겠습니다.

(여기에 일반 지식 기반 답변 작성)
</answer>

<references>
관련 문서 없음
</references>"""
            logger.debug(f"[RAG 컨텍스트] 문서 없음 - 기본 컨텍스트 생성: {len(no_docs_context)} 문자")
            return no_docs_context
            
        try:
            context = ""
            logger.info(f"검색 결과 {len(documents)}개 문서를 컨텍스트로 포맷팅합니다.")
            
            for idx, doc in enumerate(documents, 1):
                logger.debug(f"[RAG 컨텍스트] 문서 {idx} 처리: {doc.get('title', '제목 없음')[:30]}...")
                
                doc_context = f"[문서 {idx}]\n"
                doc_context += f"제목: {doc.get('title', '제목 없음')}\n"
                
                # 작성자 정보 추가
                if doc.get('author'):
                    doc_context += f"작성자: {doc.get('author')}\n"
                
                # 날짜 정보 추가
                if 'tags' in doc and 'date' in doc['tags']:
                    date = doc['tags']['date']
                    if len(date) == 8:  # YYYYMMDD 형식인 경우
                        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                        doc_context += f"날짜: {formatted_date}\n"
                
                # 도메인 정보 추가
                if doc.get('domain'):
                    doc_context += f"출처: {doc.get('domain')}\n"
                
                # MRC 정보 추가 (질문에 대한 직접 답변)
                if doc.get('mrc_answer'):
                    doc_context += f"핵심 답변: {doc.get('mrc_answer')}\n"
                    doc_context += f"답변 신뢰도: {doc.get('mrc_score', 0):.2f}\n"
                
                # 재랭킹 점수 정보 추가
                if doc.get('score'):
                    doc_context += f"관련도: {doc.get('score', 0):.2f}\n"
                
                # 링크 정보 추가 - 참고 문헌에서 활용하기 위해 별도 필드로 추가
                link = ""
                if 'info' in doc and 'url' in doc['info']:
                    link = doc['info']['url']
                    doc_context += f"원문 링크: {link}\n"
                elif 'info' in doc and 'link' in doc['info']:
                    link = doc['info']['link']
                    doc_context += f"원문 링크: {link}\n"
                
                # 문서 ID 정보 추가 (디버깅용)
                if doc.get('doc_id'):
                    doc_context += f"문서 ID: {doc.get('doc_id')}\n"
                
                # 본문 내용
                doc_context += f"내용: {doc.get('text', '')}\n\n"
                
                context += doc_context
                logger.debug(f"[RAG 컨텍스트] 문서 {idx} 완료: {len(doc_context)} 문자")
            
            # 문서가 있는 경우 응답 형식 안내 추가
            format_instruction = """
중요: 응답 형식은 다음과 같이 작성하세요:


여기에 사용자 질문에 대한 답변을 작성하세요. 문서에서 정보를 인용할 때는 반드시 "📚[숫자]" 형식으로 출처를 표시하세요.
예: "메타버스는 가상 세계입니다 📚[1]"
"""
            
            context += format_instruction
            logger.debug(f"[RAG 컨텍스트] 응답 형식 지침 추가: {len(format_instruction)} 문자")
            logger.debug(f"[RAG 컨텍스트] 완료: 총 {len(context)} 문자")
            
            return context
        except Exception as e:
            logger.error(f"컨텍스트 포맷팅 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시 기본 메시지 반환
            error_context = "검색 결과 처리 중 오류가 발생했습니다. 일반 지식을 기반으로 답변하겠습니다."
            logger.debug(f"[RAG 컨텍스트] 오류 - 기본 컨텍스트 반환: {len(error_context)} 문자")
            return error_context
    
    def generate_response(self, session_id: str, query: str, model: str = None, stream: bool = False, **kwargs) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """RAG 검색 결과를 기반으로 챗봇 응답을 생성합니다."""
        try:
            start_time = datetime.now()
            logger.info(f"[성능] 응답 생성 시작: 세션={session_id}, 쿼리='{query[:30]}...'")
            
            # 체인 실행을 위한 입력 데이터 준비
            chain_input = {
                "session_id": session_id,
                "query": query,
                "model": model or self.default_model,
                "stream": stream,
                "kwargs": kwargs,
                "start_time": start_time
            }
            
            # 체인 실행
            if stream:
                return self._run_streaming_chain(chain_input)
            else:
                return self._run_normal_chain(chain_input)
                
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}", exc_info=True)
            error_response = {"error": f"응답 생성 중 오류가 발생했습니다: {str(e)}"}
            if not stream:
                return error_response
            else:
                def error_generator():
                    yield json.dumps(error_response, ensure_ascii=False)
                return error_generator()
    
    def _run_normal_chain(self, chain_input: Dict[str, Any]) -> Dict[str, Any]:
        """비스트리밍 모드에서 체인을 실행합니다."""
        session_id = chain_input["session_id"]
        query = chain_input["query"]
        model = chain_input["model"]
        start_time = chain_input["start_time"]
        
        try:
            # 1. RAG 체인 실행
            session_load_start = datetime.now()
            chain_result = self.rag_chain.invoke(chain_input)
            logger.info(f"[성능] RAG 체인 실행 완료: {(datetime.now() - session_load_start).total_seconds():.3f}초")
            
            # 최종 프롬프트 로깅 (디버깅용)
            logger.debug(f"[디버깅] LLM에 전달되는 최종 프롬프트 (처음 500자):\n{chain_result['prompt'][:500]}...")
            logger.debug(f"[디버깅] LLM에 전달되는 최종 프롬프트 (마지막 500자):\n{chain_result['prompt'][-500:] if len(chain_result['prompt']) > 500 else chain_result['prompt']}")
            
            # 전체 프롬프트 로깅 (INFO 레벨)
            logger.info(f"[최종 프롬프트] 길이: {len(chain_result['prompt'])} 문자")
            logger.info("=== 최종 프롬프트 전체 내용 ===")
            logger.info(f"{chain_result['prompt']}")
            logger.info("================================")
            
            # 2. LLM 응답 생성
            llm_start = datetime.now()
            model_to_use = model or self.default_model
            logger.info(f"[성능] LLM 요청 시작: 모델={model_to_use}, 세션={session_id}")
            
            # Ollama API 호출
            ollama_response = requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": chain_result["prompt"],
                    "stream": False
                },
                timeout=120
            )
            
            if ollama_response.status_code != 200:
                logger.error(f"Ollama API 오류: {ollama_response.text}")
                return {"error": "LLM 요청 중 오류가 발생했습니다", "details": ollama_response.text}
            
            llm_end = datetime.now()
            logger.info(f"[성능] LLM 응답 완료: {(llm_end - llm_start).total_seconds():.3f}초")
            
            response_text = ollama_response.json().get("response", "")
            
            # 응답 로깅 (디버깅용)
            logger.debug(f"[디버깅] LLM 응답 (처음 500자):\n{response_text[:500]}...")
            logger.debug(f"[디버깅] LLM 응답 (마지막 500자):\n{response_text[-500:] if len(response_text) > 500 else response_text}")
            
            # 응답 파싱 및 구조화
            structured_response = self.parse_structured_response(response_text, chain_result["search_results"])
            
            # 3. 봇 응답 저장 (원본 텍스트)
            self.session_manager.add_bot_message(session_id, response_text)
            
            logger.info(f"[성능] 총 응답 생성 시간: {(datetime.now() - start_time).total_seconds():.3f}초")
            
            return {
                "response": structured_response["answer"],
                "model": model_to_use,
                "references": structured_response["references"]
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 서비스 연결 오류: {str(e)}")
            return {"error": "Ollama 서비스에 연결할 수 없습니다", "details": str(e)}
    
    def parse_structured_response(self, response_text: str, search_results: List[Dict]) -> Dict[str, Any]:
        """LLM 응답을 구조화된 형식으로 파싱합니다."""
        import re
        
        result = {
            "answer": response_text.strip(),
            "references": []
        }
        
        # 📚[숫자] 패턴에서 숫자만 추출
        citation_pattern = r'📚\[(\d+)\]'
        citation_matches = re.findall(citation_pattern, response_text)
        
        if citation_matches:
            logger.info(f"응답에서 {len(set(citation_matches))}개의 인용 패턴을 찾았습니다.")
            
            # 중복 제거 및 정렬
            unique_citations = sorted(set([int(num) for num in citation_matches]))
            
            # 실제 검색된 문서와 매핑
            for citation_num in unique_citations:
                doc_index = citation_num - 1  # 1-based to 0-based
                if 0 <= doc_index < len(search_results):
                    doc = search_results[doc_index]
                    result["references"].append({
                        "number": str(citation_num),
                        "title": doc.get('title', '제목 없음'),
                        "link": doc.get('info', {}).get('url', '')
                    })
                    logger.debug(f"인용 패턴에서 참고 문헌 생성: 번호={citation_num}, 제목={doc.get('title', '제목 없음')}")
                else:
                    logger.warning(f"인용된 문서 번호 {citation_num}이 검색 결과 범위를 벗어남 (최대: {len(search_results)})")
        else:
            logger.warning("응답에서 인용 패턴(📚[숫자])을 찾을 수 없습니다.")
            
            # 문서가 있는 경우 기본 참고 문헌 생성
            if "관련 문서를 찾을 수 없어" not in result["answer"] and search_results:
                result["references"].append({
                    "number": "1",
                    "title": "참고 문헌",
                    "link": ""
                })
                logger.debug("기본 참고 문헌 생성")
        
        return result
    
    def _run_streaming_chain(self, chain_input: Dict[str, Any]) -> Generator[str, None, None]:
        """스트리밍 모드에서 체인을 실행합니다."""
        session_id = chain_input["session_id"]
        query = chain_input["query"]
        model = chain_input["model"]
        start_time = chain_input["start_time"]
        
        try:
            # 1. RAG 체인 실행
            session_load_start = datetime.now()
            chain_result = self.rag_chain.invoke(chain_input)
            logger.info(f"[성능] RAG 체인 실행 완료: {(datetime.now() - session_load_start).total_seconds():.3f}초")
            
            # 최종 프롬프트 로깅 (디버깅용)
            logger.debug(f"[디버깅] LLM에 전달되는 최종 프롬프트 (처음 500자):\n{chain_result['prompt'][:500]}...")
            logger.debug(f"[디버깅] LLM에 전달되는 최종 프롬프트 (마지막 500자):\n{chain_result['prompt'][-500:] if len(chain_result['prompt']) > 500 else chain_result['prompt']}")
            
            # 전체 프롬프트 로깅 (INFO 레벨)
            logger.info(f"[최종 프롬프트] 길이: {len(chain_result['prompt'])} 문자")
            logger.info("=== 최종 프롬프트 전체 내용 ===")
            logger.info(f"{chain_result['prompt']}")
            logger.info("================================")
            
            # 2. LLM 스트리밍 응답 생성
            model_to_use = model or self.default_model
            logger.info(f"[성능] 스트리밍 LLM 요청 시작: 모델={model_to_use}, 세션={session_id}")
            llm_start = datetime.now()
            
            # Ollama API 스트리밍 호출
            with requests.post(
                f"{self.ollama_endpoint}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": chain_result["prompt"],
                    "stream": True
                },
                timeout=120,
                stream=True
            ) as ollama_response:
                if ollama_response.status_code != 200:
                    logger.error(f"Ollama API 오류: {ollama_response.text}")
                    yield json.dumps({
                        "error": "LLM 요청 중 오류가 발생했습니다",
                        "details": ollama_response.text
                    }, ensure_ascii=False)
                    return
                
                # 응답 누적을 위한 변수
                accumulated_response = ""
                
                for line in ollama_response.iter_lines():
                    if line:
                        try:
                            response_chunk = json.loads(line)
                            chunk_text = response_chunk.get("response", "")
                            if chunk_text:
                                accumulated_response += chunk_text
                                
                                # 스트리밍 응답 형식
                                stream_response = {
                                    "response": accumulated_response,
                                    "model": model_to_use,
                                    "streaming": True
                                }
                                
                                yield json.dumps(stream_response, ensure_ascii=False)
                        except json.JSONDecodeError:
                            logger.error(f"JSON 디코딩 오류: {line}")
                            continue
                
                # 스트림 종료 후 메모리에 저장
                self.session_manager.add_bot_message(session_id, accumulated_response)
                
                # 응답 로깅 (디버깅용)
                logger.debug(f"[디버깅] 최종 누적 LLM 응답 (처음 500자):\n{accumulated_response[:500]}...")
                logger.debug(f"[디버깅] 최종 누적 LLM 응답 (마지막 500자):\n{accumulated_response[-500:] if len(accumulated_response) > 500 else accumulated_response}")
                
                # 성능 로깅
                logger.info(f"[성능] 스트리밍 LLM 응답 완료: {(datetime.now() - llm_start).total_seconds():.3f}초, 응답 길이: {len(accumulated_response)}")
                logger.info(f"[성능] 총 응답 생성 시간: {(datetime.now() - start_time).total_seconds():.3f}초")
                
                # 스트림 종료 시 구조화된 응답 파싱
                structured_response = self.parse_structured_response(accumulated_response, chain_result["search_results"])
                
                # 스트림 종료 응답
                final_response = {
                    "response": structured_response["answer"],
                    "model": model_to_use,
                    "streaming": False,
                    "done": True,
                    "references": structured_response["references"]
                }
                yield json.dumps(final_response, ensure_ascii=False)
        except Exception as e:
            logger.error(f"스트리밍 응답 생성 중 오류 발생: {str(e)}")
            yield json.dumps({
                "error": f"스트리밍 응답 생성 중 오류가 발생했습니다: {str(e)}",
                "streaming": False,
                "done": True
            }, ensure_ascii=False)
    
    def clear_session(self, session_id: str) -> Dict[str, Any]:
        """세션을 초기화합니다."""
        self.session_manager.clear_session(session_id)
        return {"status": "success", "message": f"세션 {session_id}가 초기화되었습니다"}
    
    def cleanup_expired_sessions(self) -> Dict[str, Any]:
        """만료된 세션을 정리합니다."""
        count = self.session_manager.cleanup_expired_sessions()
        return {"status": "success", "message": f"{count}개의 만료된 세션이 정리되었습니다"} 