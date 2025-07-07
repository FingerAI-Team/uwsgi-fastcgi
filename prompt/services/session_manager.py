import os
import json
import time
import logging
import re
import fcntl  # 파일 잠금을 위한 모듈
import errno
from datetime import datetime
import tiktoken
from typing import List, Dict, Any, Optional, Tuple

# 로깅 설정
logger = logging.getLogger("session-manager")

# 토큰 카운터 초기화
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT 모델용 토크나이저 (대부분의 모델과 호환)

def count_tokens(text: str) -> int:
    """텍스트의 토큰 수를 계산합니다."""
    if not text:
        return 0
    return len(tokenizer.encode(text))

class FileLock:
    """파일 잠금 클래스"""
    
    def __init__(self, file_path, timeout=30, delay=0.1):
        self.file_path = file_path
        self.timeout = timeout
        self.delay = delay
        self.lock_file = f"{file_path}.lock"
        self.fd = None
        
    def acquire(self):
        """잠금을 획득합니다."""
        start_time = time.time()
        
        while True:
            try:
                # 잠금 파일 생성 시도
                self.fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                
                # 타임아웃 확인
                if (time.time() - start_time) >= self.timeout:
                    raise TimeoutError(f"파일 잠금 획득 시간 초과: {self.file_path}")
                
                # 잠시 대기 후 재시도
                time.sleep(self.delay)
                
        # 잠금 파일에 PID 기록
        os.write(self.fd, str(os.getpid()).encode())
        
    def release(self):
        """잠금을 해제합니다."""
        if self.fd:
            os.close(self.fd)
            try:
                os.unlink(self.lock_file)
            except OSError:
                pass
            self.fd = None
            
    def __enter__(self):
        self.acquire()
        return self
        
    def __exit__(self, type, value, traceback):
        self.release()

class SessionManager:
    """파일 기반 세션 관리 클래스"""
    
    def __init__(self, 
                 memory_dir: str = "./memory", 
                 max_total_tokens: int = 10000,
                 max_context_tokens: int = 7500,
                 session_ttl: int = 86400):  # 24시간
        """
        세션 관리자 초기화
        
        Args:
            memory_dir: 메모리 파일 저장 디렉토리
            max_total_tokens: 전체 입력 최대 토큰 수
            max_context_tokens: 문맥 유지 최대 토큰 수
            session_ttl: 세션 유효 시간(초)
        """
        self.memory_dir = memory_dir
        self.max_total_tokens = max_total_tokens
        self.max_context_tokens = max_context_tokens
        self.session_ttl = session_ttl
        
        # 메모리 디렉토리 생성
        os.makedirs(memory_dir, exist_ok=True)
        
        logger.info(f"SessionManager 초기화 완료: memory_dir={memory_dir}, max_context_tokens={max_context_tokens}")
    
    def _validate_session_id(self, session_id: str) -> str:
        """세션 ID를 검증하고 안전한 형식으로 변환합니다."""
        # 세션 ID가 없는 경우 기본값 생성
        if not session_id:
            session_id = f"session_{int(time.time())}"
            logger.warning(f"세션 ID가 없어 기본값 생성: {session_id}")
        
        # 안전한 파일명 형식으로 변환 (알파벳, 숫자, 하이픈, 언더스코어만 허용)
        safe_id = re.sub(r'[^a-zA-Z0-9\-_]', '_', session_id)
        
        # 변환 전후가 다르면 로깅
        if safe_id != session_id:
            logger.warning(f"세션 ID 변환: '{session_id}' → '{safe_id}'")
        
        return safe_id
    
    def _get_session_path(self, session_id: str) -> str:
        """세션 파일 경로를 반환합니다."""
        safe_id = self._validate_session_id(session_id)
        return os.path.join(self.memory_dir, f"{safe_id}.json")
    
    def session_exists(self, session_id: str) -> bool:
        """세션이 존재하는지 확인합니다."""
        session_path = self._get_session_path(session_id)
        return os.path.exists(session_path)
    
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """세션 데이터를 로드합니다."""
        session_path = self._get_session_path(session_id)
        
        if os.path.exists(session_path):
            try:
                # 파일 잠금 사용
                with FileLock(session_path):
                    with open(session_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # 세션 유효성 검사
                    if self._is_session_expired(session_data):
                        logger.info(f"세션 {session_id} 만료됨, 새 세션 생성")
                        return self._create_new_session(session_id)
                    
                    logger.debug(f"세션 {session_id} 로드 완료: 대화 턴 수={len(session_data.get('history', []))}")
                    return session_data
            except Exception as e:
                logger.error(f"세션 {session_id} 로드 중 오류: {str(e)}")
                return self._create_new_session(session_id)
        else:
            logger.info(f"새 세션 생성: {session_id}")
            return self._create_new_session(session_id)
    
    def _is_session_expired(self, session_data: Dict[str, Any]) -> bool:
        """세션이 만료되었는지 확인합니다."""
        try:
            last_updated = datetime.fromisoformat(session_data.get("last_updated", "2000-01-01T00:00:00"))
            now = datetime.now()
            time_diff = (now - last_updated).total_seconds()
            return time_diff > self.session_ttl
        except (ValueError, TypeError) as e:
            logger.error(f"세션 만료 확인 중 오류: {str(e)}")
            return True  # 오류 발생 시 만료된 것으로 간주
    
    def _create_new_session(self, session_id: str) -> Dict[str, Any]:
        """새 세션 데이터를 생성합니다."""
        return {
            "session_id": session_id,
            "history": [],
            "last_updated": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }
    
    def save_session(self, session_data: Dict[str, Any]) -> None:
        """세션 데이터를 저장합니다."""
        session_id = session_data["session_id"]
        safe_id = self._validate_session_id(session_id)
        session_path = self._get_session_path(safe_id)
        
        # 마지막 업데이트 시간 갱신
        session_data["last_updated"] = datetime.now().isoformat()
        
        try:
            # 디렉토리 존재 확인
            os.makedirs(os.path.dirname(session_path), exist_ok=True)
            
            # 파일 잠금 사용
            with FileLock(session_path):
                # 임시 파일에 먼저 저장 후 이름 변경 (파일 손상 방지)
                temp_path = f"{session_path}.tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, ensure_ascii=False, indent=2)
                
                # 임시 파일을 실제 파일로 이름 변경
                if os.path.exists(session_path):
                    os.replace(temp_path, session_path)
                else:
                    os.rename(temp_path, session_path)
                    
                logger.debug(f"세션 {session_id} 저장 완료")
        except Exception as e:
            logger.error(f"세션 {session_id} 저장 중 오류: {str(e)}")
    
    def add_user_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """사용자 메시지를 세션에 추가합니다."""
        session_data = self.load_session(session_id)
        
        # 사용자 메시지 추가
        session_data["history"].append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        self.save_session(session_data)
        return session_data
    
    def add_bot_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """봇 메시지를 세션에 추가합니다."""
        session_data = self.load_session(session_id)
        
        # 봇 메시지 추가
        session_data["history"].append({
            "role": "bot",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        self.save_session(session_data)
        return session_data
    
    def clear_session(self, session_id: str) -> None:
        """세션을 초기화합니다."""
        session_path = self._get_session_path(session_id)
        
        if os.path.exists(session_path):
            try:
                # 파일 잠금 사용
                with FileLock(session_path):
                    os.remove(session_path)
                    logger.info(f"세션 {session_id} 삭제됨")
            except Exception as e:
                logger.error(f"세션 {session_id} 삭제 중 오류: {str(e)}")
    
    def get_all_sessions(self) -> List[str]:
        """모든 세션 ID 목록을 반환합니다."""
        session_ids = []
        try:
            for filename in os.listdir(self.memory_dir):
                if filename.endswith('.json'):
                    session_ids.append(filename[:-5])  # .json 확장자 제거
            return session_ids
        except Exception as e:
            logger.error(f"세션 목록 조회 중 오류: {str(e)}")
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """만료된 세션 파일을 정리합니다."""
        count = 0
        now = time.time()
        
        for filename in os.listdir(self.memory_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.memory_dir, filename)
                
                try:
                    # 파일 수정 시간 확인
                    mtime = os.path.getmtime(file_path)
                    if now - mtime > self.session_ttl:
                        # 파일 잠금 사용
                        with FileLock(file_path):
                            os.remove(file_path)
                            count += 1
                            logger.info(f"만료된 세션 파일 삭제: {filename}")
                except Exception as e:
                    logger.error(f"세션 파일 {filename} 정리 중 오류: {str(e)}")
        
        return count
    
    def build_prompt_context(self, session_data: Dict[str, Any], system_prompt: str, 
                           rag_context: str = "", current_query: str = "") -> str:
        """프롬프트 컨텍스트를 구성합니다."""
        # 시스템 프롬프트
        prompt = system_prompt + "\n\n"
        
        # RAG 컨텍스트 (있는 경우)
        if rag_context:
            prompt += "관련 문서 정보:\n" + rag_context + "\n\n"
        
        # 대화 기록 - 최근 5턴만 포함
        if session_data["history"]:
            prompt += "최근 대화:\n"
            # 최대 5턴(10개 메시지)만 포함
            recent_history = session_data["history"][-10:] if len(session_data["history"]) > 10 else session_data["history"]
            for msg in recent_history:
                role = "사용자" if msg["role"] == "user" else "AI"
                prompt += f"{role}: {msg['message']}\n\n"
        
        return prompt 