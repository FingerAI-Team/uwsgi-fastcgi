#!/usr/bin/env python3
"""
로깅 시스템 테스트 스크립트
"""

import logging
import os
import time
from datetime import datetime

def test_logging_system():
    """로깅 시스템을 테스트합니다."""
    
    # 로그 디렉토리 생성
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    print(f"로그 디렉토리: {log_dir}")
    
    # 메인 로거 설정
    logger = logging.getLogger("rag-backend")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # 스트림 핸들러
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # API 요청 로거
    api_logger = logging.getLogger('api-requests')
    api_logger.setLevel(logging.INFO)
    api_logger.handlers = []
    api_handler = logging.FileHandler(os.path.join(log_dir, 'api-requests.log'))
    api_formatter = logging.Formatter('%(asctime)s - %(message)s')
    api_handler.setFormatter(api_formatter)
    api_logger.addHandler(api_handler)
    
    # 위계형 시스템 로거
    hierarchical_logger = logging.getLogger('hierarchical')
    hierarchical_logger.setLevel(logging.INFO)
    hierarchical_logger.handlers = []
    hierarchical_handler = logging.FileHandler(os.path.join(log_dir, 'hierarchical.log'))
    hierarchical_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    hierarchical_handler.setFormatter(hierarchical_formatter)
    hierarchical_logger.addHandler(hierarchical_handler)
    
    print("=== 로깅 시스템 테스트 시작 ===")
    
    # 테스트 로그 메시지들
    test_messages = [
        ("INFO", "🔍 컬렉션 정보 조회 시작: legal_documents"),
        ("INFO", "📚 Milvus Collection 객체 생성: legal_documents"),
        ("INFO", "📥 컬렉션 로드 시작: legal_documents"),
        ("INFO", "✅ 컬렉션 로드 완료: legal_documents"),
        ("INFO", "📋 컬렉션 스키마 정보 수집: legal_documents"),
        ("INFO", "🔍 컬렉션 인덱스 정보 수집: legal_documents"),
        ("INFO", "📦 컬렉션 파티션 정보 수집: legal_documents"),
        ("INFO", "📊 컬렉션 엔티티 수 조회: legal_documents"),
        ("INFO", "✅ 컬렉션 정보 조회 성공: legal_documents"),
        ("INFO", "📊 컬렉션 정보 - 엔티티 수: 1500, 필드 수: 15, 파티션 수: 1"),
        ("INFO", "⏱️ 처리 시간: 0.234초"),
    ]
    
    # 각 로거에 테스트 메시지 전송
    for level, message in test_messages:
        if level == "INFO":
            logger.info(message)
            api_logger.info(message)
            hierarchical_logger.info(message)
        elif level == "ERROR":
            logger.error(message)
            api_logger.error(message)
            hierarchical_logger.error(message)
    
    # 에러 로그 테스트
    error_message = "❌ 컬렉션 정보 조회 실패: legal_documents"
    logger.error(error_message)
    api_logger.error(error_message)
    hierarchical_logger.error(error_message)
    
    print("=== 로깅 시스템 테스트 완료 ===")
    print(f"로그 파일들이 {log_dir} 디렉토리에 생성되었습니다.")
    
    # 생성된 로그 파일들 확인
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    print(f"생성된 로그 파일들: {log_files}")
    
    for log_file in log_files:
        file_path = os.path.join(log_dir, log_file)
        file_size = os.path.getsize(file_path)
        print(f"  {log_file}: {file_size} bytes")

if __name__ == "__main__":
    test_logging_system()
