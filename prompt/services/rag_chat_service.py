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
from domain_selector.domain_service import DomainService
from query_rewriter.query_rewriter import QueryRewriter

# 로깅 설정
logger = logging.getLogger("rag-chat-service")

class RagChatService:
    """RAG 챗봇 서비스 클래스"""
    
    def __init__(self, 
                 memory_dir: str = "./memory",
                 ollama_endpoint: str = "http://ollama-gpu:11434",
                 rag_endpoint: str = "http://nginx/rag",
                 reranker_endpoint: str = "http://nginx/reranker",
                 default_model: str = "gemma3:12b",
                 search_top: int = 100,
                 rerank_top: int = 10,
                 rerank_threshold: float = 0.1,
                 temperature: float = 0.7,
                 max_total_tokens: int = 10000,
                 max_context_tokens: int = 7500,
                 vllm_endpoint: str = "http://vllm:8000"):
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
        self.temperature = temperature
        
        # 세션 관리자 초기화
        self.session_manager = SessionManager(
            memory_dir=memory_dir,
            max_total_tokens=max_total_tokens,
            max_context_tokens=max_context_tokens
        )
        
        # 도메인 셀렉터 서비스 초기화
        self.domain_service = DomainService()
        
        # Query Rewriter 서비스 초기화 (vLLM 사용)
        self.query_rewriter = QueryRewriter(
            ollama_endpoint=vllm_endpoint,  # vLLM 엔드포인트 사용
            default_model=default_model,
            temperature=temperature
        )
        
        # 시스템 프롬프트는 검색 결과에 따라 동적으로 로드됨
        self.system_prompt = None
        
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
            logger.info(f"[체인 실행] 1단계 - 세션 로드 및 사용자 메시지 추가: 세션={session_id}")
            return {
                "session_data": self.session_manager.add_user_message(session_id, query),
                **inputs  # 원래 입력 유지
            }
        
        # Query Rewrite 수행 함수
        def perform_query_rewrite(inputs):
            session_data = inputs["session_data"]
            original_query = inputs["query"]
            model = inputs.get("model")
            
            logger.info(f"[체인 실행] 2단계 - Query Rewrite 수행: 원본 쿼리='{original_query[:30]}...'")
            
            # Query Rewrite 수행
            rewrite_result = self.query_rewriter.rewrite_query(
                current_query=original_query,
                session_data=session_data,
                model=model
            )
            
            rewritten_query = rewrite_result["rewritten_query"]
            confidence = rewrite_result["confidence"]
            
            logger.info(f"[Query Rewrite] 결과: '{original_query[:20]}...' → '{rewritten_query[:20]}...' (신뢰도: {confidence:.2f})")
            
            return {
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "rewrite_confidence": confidence,
                "rewrite_reasoning": rewrite_result["reasoning"],
                **inputs  # 원래 입력 유지
            }
        
        # RAG 검색 수행 함수 (도메인 셀렉터 통합)
        def perform_search(inputs):
            # Query Rewrite 결과 사용
            query = inputs.get("rewritten_query", inputs["query"])
            original_query = inputs.get("original_query", query)
            kwargs = inputs.get("kwargs", {})
            logger.info(f"[체인 실행] 3단계 - 도메인 셀렉터 및 RAG 검색 수행: 쿼리='{query[:30]}...' (원본: '{original_query[:30]}...')")
            
            # 도메인 셀렉터로 검색 범위 결정
            domain_result = self.domain_service.process_query(query)
            logger.info(f"[도메인 셀렉터] 결과: {json.dumps(domain_result, ensure_ascii=False)}")
            
            # 도메인 선택 우선순위: 사용자 지정 > 도메인 셀렉터 > 기본 도메인
            search_kwargs = kwargs.copy()
            
            # 1. 사용자가 직접 지정한 도메인이 있으면 우선 사용
            if "domains" in kwargs and kwargs["domains"]:
                logger.info(f"[도메인 선택] 사용자 지정 도메인 우선 사용: {kwargs['domains']}")
            elif "domain" in kwargs and kwargs["domain"]:
                logger.info(f"[도메인 선택] 사용자 지정 단일 도메인 우선 사용: {kwargs['domain']}")
            # 2. 사용자 지정이 없고 도메인 셀렉터가 도메인을 찾았으면 사용
            elif domain_result["domain_candidates"]:
                search_kwargs["domains"] = domain_result["domain_candidates"]
                logger.info(f"[도메인 선택] 도메인 셀렉터 결과 사용: {domain_result['domain_candidates']}")
            # 3. 둘 다 없으면 전체 도메인에서 검색
            else:
                logger.info(f"[도메인 선택] 전체 도메인에서 검색")
            
            # 검색 결과만 반환 (도메인 정보는 검색 파라미터로만 사용)
            search_results = self._perform_enhanced_search(query, **search_kwargs)
            return {
                "search_results": search_results,
                **inputs  # 원래 입력 유지
            }
        
        # 컨텍스트 포맷팅 함수
        def format_search_results(inputs):
            search_results = inputs["search_results"]
            logger.info(f"[체인 실행] 5단계 - 검색 결과 포맷팅: {len(search_results)}개 문서")
            return {
                "rag_context": self.format_context(search_results),
                **inputs  # 원래 입력 유지
            }
        
        # 세션 저장 및 프롬프트 구성 함수
        def build_prompt(inputs):
            session_data = inputs["session_data"]
            # Query Rewrite 결과가 있으면 재작성된 질문 사용, 없으면 원본 사용
            query = inputs.get("rewritten_query", inputs["query"])
            rag_context = inputs["rag_context"]
            system_prompt = inputs["system_prompt"]
            
            # 세션 저장
            self.session_manager.save_session(session_data)
            
            # 프롬프트 구성
            prompt = self.session_manager.build_prompt_context(
                session_data,
                system_prompt,
                rag_context=rag_context,
                current_query=query
            )
            
            logger.info(f"[체인 실행] 6단계 - 프롬프트 구성 완료: {len(prompt)} 문자")
            return {
                "prompt": prompt,
                **inputs  # 원래 입력 유지
            }
        
        # 시스템 프롬프트 로드 함수
        def load_system_prompt(inputs):
            search_results = inputs.get("search_results", [])
            has_documents = len(search_results) > 0
            system_prompt = self._load_system_prompt(has_documents)
            logger.info(f"[체인 실행] 4단계 - 시스템 프롬프트 로드: 문서 유무={has_documents}, 템플릿={'rag_chat_with_docs' if has_documents else 'rag_chat_no_docs'}")
            return {
                "system_prompt": system_prompt,
                **inputs
            }
        
        # 체인 구성 (최신 LangChain 방식)
        self.rag_chain = (
            RunnableLambda(load_session)
            | RunnableLambda(perform_query_rewrite)  # Query Rewrite 추가
            | RunnableLambda(perform_search)
            | RunnableLambda(load_system_prompt)  # 검색 결과에 따라 시스템 프롬프트 로드
            | RunnableLambda(format_search_results)
            | RunnableLambda(build_prompt)
        )
    
    def _load_system_prompt(self, has_documents: bool = True) -> str:
        """검색 결과 유무에 따라 적절한 시스템 프롬프트를 로드합니다."""
        template_name = "rag_chat_with_docs" if has_documents else "rag_chat_no_docs"
        template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", f"{template_name}.txt")
        logger.debug(f"[시스템 프롬프트] 템플릿 파일 경로: {template_path}")
        
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
                logger.debug(f"[시스템 프롬프트] 템플릿 파일 로드 성공: {len(system_prompt)} 문자")
                return system_prompt
        except FileNotFoundError:
            logger.error(f"템플릿 파일을 찾을 수 없습니다: {template_path}")
            raise FileNotFoundError(f"필수 템플릿 파일이 없습니다: {template_name}.txt")
    
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
            logger.info("검색 결과가 없어 빈 컨텍스트를 반환합니다.")
            return ""  # 빈 값 반환 (시스템 프롬프트에서 처리)
            
        try:
            context = ""
            # 상위 5개 문서만 포함 (프롬프트 길이 단축)
            top_documents = documents[:5]
            logger.info(f"검색 결과 {len(documents)}개 중 상위 {len(top_documents)}개 문서를 컨텍스트로 포맷팅합니다.")
            
            for idx, doc in enumerate(top_documents, 1):
                logger.debug(f"[RAG 컨텍스트] 문서 {idx} 처리: {doc.get('title', '제목 없음')[:30]}...")
                
                doc_context = f"[문서 {idx}]\n"
                # 제목에서 개행 문자 제거하고 정리
                title = doc.get('title', '제목 없음').strip().replace('\n', ' ').replace('\r', '')
                doc_context += f"제목: {title}\n"
                
                # 작성자 정보 추가
                if doc.get('author'):
                    author = doc.get('author').strip().replace('\n', ' ').replace('\r', '')
                    doc_context += f"작성자: {author}\n"
                
                # 날짜 정보 추가
                if 'tags' in doc and 'date' in doc['tags']:
                    date = doc['tags']['date']
                    if len(date) == 8:  # YYYYMMDD 형식인 경우
                        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                        doc_context += f"날짜: {formatted_date}\n"
                
                # 도메인 정보 추가
                if doc.get('domain'):
                    domain = doc.get('domain').strip().replace('\n', ' ').replace('\r', '')
                    doc_context += f"출처: {domain}\n"
                
                # MRC 정보 추가 (질문에 대한 직접 답변)
                if doc.get('mrc_answer'):
                    mrc_answer = doc.get('mrc_answer').strip().replace('\n', ' ').replace('\r', '')
                    doc_context += f"핵심 답변: {mrc_answer}\n"
                    doc_context += f"답변 신뢰도: {doc.get('mrc_score', 0):.2f}\n"
                
                # 재랭킹 점수 정보 추가
                if doc.get('score'):
                    doc_context += f"관련도: {doc.get('score', 0):.2f}\n"
                
                # 링크 정보 추가 - 참고 문헌에서 활용하기 위해 별도 필드로 추가
                link = ""
                
                # 디버깅을 위해 문서 구조 로깅
                logger.debug(f"문서 {idx} 구조: {json.dumps(doc, ensure_ascii=False, default=str)}")
                
                # 여러 가능한 링크 필드 확인
                if 'info' in doc and 'url' in doc['info']:
                    link = doc['info']['url']
                elif 'info' in doc and 'link' in doc['info']:
                    link = doc['info']['link']
                elif 'url' in doc:
                    link = doc['url']
                elif 'link' in doc:
                    link = doc['link']
                elif 'raw_doc_id' in doc and doc['raw_doc_id']:
                    # raw_doc_id에서 링크 정보 추출 시도
                    raw_id = doc['raw_doc_id']
                    if 'http' in raw_id:
                        link = raw_id
                elif 'doc_id' in doc and doc['doc_id']:
                    # doc_id에서 링크 정보 추출 시도
                    doc_id = doc['doc_id']
                    if 'http' in doc_id:
                        link = doc_id
                
                # 링크에서 개행 문자 제거
                link = link.strip().replace('\n', ' ').replace('\r', '')
                doc_context += f"원문 링크: {link}\n"
                
                # 문서 ID 정보 추가 (디버깅용)
                if doc.get('doc_id'):
                    doc_id = doc.get('doc_id').strip().replace('\n', ' ').replace('\r', '')
                    doc_context += f"문서 ID: {doc_id}\n"
                
                # 본문 내용 (개행 문자는 유지하되 앞뒤 공백 제거)
                text = doc.get('text', '').strip()
                doc_context += f"내용: {text}\n\n"
                
                context += doc_context
                logger.debug(f"[RAG 컨텍스트] 문서 {idx} 완료: {len(doc_context)} 문자")
            
            # 문서가 있는 경우 응답 형식 안내 추가 (시스템 프롬프트에서 처리하므로 제거)
            logger.debug(f"[RAG 컨텍스트] 문서 정보만 포함, 응답 형식은 시스템 프롬프트에서 처리")
            logger.debug(f"[RAG 컨텍스트] 완료: 총 {len(context)} 문자")
            
            return context
        except Exception as e:
            logger.error(f"컨텍스트 포맷팅 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시 빈 값 반환 (시스템 프롬프트에서 처리)
            logger.debug(f"[RAG 컨텍스트] 오류 - 빈 컨텍스트 반환")
            return ""
    
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
                    "stream": False,
                    "temperature": self.temperature
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
            
            # Query Rewrite 정보 추가
            query_rewrite_info = {}
            if "rewritten_query" in chain_result and "original_query" in chain_result:
                query_rewrite_info = {
                    "original_query": chain_result["original_query"],
                    "rewritten_query": chain_result["rewritten_query"],
                    "confidence": chain_result.get("rewrite_confidence", 0.0),
                    "reasoning": chain_result.get("rewrite_reasoning", "")
                }
            
            return {
                "response": structured_response["answer"],
                "model": model_to_use,
                "references": structured_response["references"],
                "query_rewrite": query_rewrite_info
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
        
        # 📚[숫자] 패턴에서 숫자만 추출 (단일 및 복수 모두 처리)
        citation_pattern = r'📚\[([\d,\s]+)\]'
        citation_matches = re.findall(citation_pattern, response_text)
        
        if citation_matches:
            logger.info(f"응답에서 {len(citation_matches)}개의 인용 패턴을 찾았습니다.")
            
            # 모든 인용된 숫자들을 추출
            all_citations = []
            for match in citation_matches:
                # 쉼표로 구분된 숫자들을 분리
                numbers = [int(num.strip()) for num in match.split(',') if num.strip().isdigit()]
                all_citations.extend(numbers)
            
            # 중복 제거 및 정렬
            unique_citations = sorted(set(all_citations))
            logger.info(f"추출된 고유 인용 번호: {unique_citations}")
            
            # 실제 검색된 문서와 매핑
            for citation_num in unique_citations:
                doc_index = citation_num - 1  # 1-based to 0-based
                if 0 <= doc_index < len(search_results):
                    doc = search_results[doc_index]
                    
                    # 링크 추출 로직 개선
                    link = ""
                    
                    # 디버깅을 위해 문서 구조 로깅
                    logger.debug(f"문서 {citation_num} 구조: {json.dumps(doc, ensure_ascii=False, default=str)}")
                    
                    # 여러 가능한 링크 필드 확인
                    if 'info' in doc and 'url' in doc['info']:
                        link = doc['info']['url']
                    elif 'info' in doc and 'link' in doc['info']:
                        link = doc['info']['link']
                    elif 'url' in doc:
                        link = doc['url']
                    elif 'link' in doc:
                        link = doc['link']
                    elif 'raw_doc_id' in doc and doc['raw_doc_id']:
                        # raw_doc_id에서 링크 정보 추출 시도
                        raw_id = doc['raw_doc_id']
                        if 'http' in raw_id:
                            link = raw_id
                    elif 'doc_id' in doc and doc['doc_id']:
                        # doc_id에서 링크 정보 추출 시도
                        doc_id = doc['doc_id']
                        if 'http' in doc_id:
                            link = doc_id
                    
                    # 제목에서 개행 문자 제거
                    title = doc.get('title', '제목 없음').strip()
                    
                    result["references"].append({
                        "number": str(citation_num),
                        "title": title,
                        "link": link
                    })
                    logger.debug(f"인용 패턴에서 참고 문헌 생성: 번호={citation_num}, 제목={title}, 링크={link}")
                else:
                    logger.warning(f"인용된 문서 번호 {citation_num}이 검색 결과 범위를 벗어남 (최대: {len(search_results)})")
        else:
            logger.info("응답에서 인용 패턴(📚[숫자])을 찾을 수 없습니다.")
            # 인용 패턴이 없으면 참고문헌 없음 (빈 배열)
        
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
                    "stream": True,
                    "temperature": self.temperature
                },
                timeout=120,
                stream=True
            ) as ollama_response:
                if ollama_response.status_code != 200:
                    yield json.dumps({"error": ollama_response.text}, ensure_ascii=False)
                    return
                
                accumulated_response = ""

                for line in ollama_response.iter_lines():
                    if not line:
                        continue

                    decoded = line.decode().strip()

                    # SSE 주석/빈줄 무시
                    if not decoded or decoded.startswith(":"):
                        continue

                    # SSE라면 data: 접두어 제거
                    if decoded.startswith("data:"):
                        decoded = decoded[5:].strip()

                    try:
                        response_chunk = json.loads(decoded)
                        accumulated_response = response_chunk.get("response", "")

                        # 클라이언트에 그대로 전달
                        yield json.dumps(response_chunk, ensure_ascii=False) + "\n\n"
                    except json.JSONDecodeError:
                        logger.error(f"JSON 디코딩 오류: {decoded}")
                        continue

                # 끝난 후
                self.session_manager.add_bot_message(session_id, accumulated_response)
                structured_response = self.parse_structured_response(accumulated_response, chain_result["search_results"])
                
                # Query Rewrite 정보 추가
                query_rewrite_info = {}
                if "rewritten_query" in chain_result and "original_query" in chain_result:
                    query_rewrite_info = {
                        "original_query": chain_result["original_query"],
                        "rewritten_query": chain_result["rewritten_query"],
                        "confidence": chain_result.get("rewrite_confidence", 0.0),
                        "reasoning": chain_result.get("rewrite_reasoning", "")
                    }
                
                final_response = {
                    "response": structured_response["answer"],
                    "model": model_to_use,
                    "streaming": False,
                    "done": True,
                    "references": structured_response["references"],
                    "query_rewrite": query_rewrite_info
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