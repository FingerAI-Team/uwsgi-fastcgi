-- 통계 데이터베이스 초기화 스크립트

-- API 호출 통계 테이블
CREATE TABLE IF NOT EXISTS api_calls (
    id INT AUTO_INCREMENT PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL COMMENT 'API 엔드포인트',
    domain VARCHAR(255) COMMENT '도메인 정보(단독)',
--    method VARCHAR(10) NOT NULL COMMENT 'HTTP 메서드',
    status_code INT NOT NULL COMMENT 'HTTP 상태 코드',
    response_time FLOAT NOT NULL COMMENT '응답 시간(초)',
--    request_size INT COMMENT '요청 크기(바이트)',
--    response_size INT COMMENT '응답 크기(바이트)',
--    user_agent VARCHAR(255) COMMENT '사용자 에이전트',
    ip_address VARCHAR(45) COMMENT 'IP 주소',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성 시간',
    INDEX idx_endpoint (endpoint),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
-- 추가 필드 domain 단독으로 넣고 domains 여러 개로 들어올 시 잘라서 넣기

-- 일별 통계 집계 테이블
CREATE TABLE IF NOT EXISTS daily_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL COMMENT 'API 엔드포인트',
    date DATE NOT NULL COMMENT '날짜',
    total_calls INT NOT NULL DEFAULT 0 COMMENT '총 호출 수',
    success_calls INT NOT NULL DEFAULT 0 COMMENT '성공 호출 수',
    error_calls INT NOT NULL DEFAULT 0 COMMENT '오류 호출 수',
    avg_response_time FLOAT NOT NULL DEFAULT 0 COMMENT '평균 응답 시간(초)',
    max_response_time FLOAT NOT NULL DEFAULT 0 COMMENT '최대 응답 시간(초)',
    min_response_time FLOAT NOT NULL DEFAULT 0 COMMENT '최소 응답 시간(초)',
    total_request_size BIGINT NOT NULL DEFAULT 0 COMMENT '총 요청 크기(바이트)',
    total_response_size BIGINT NOT NULL DEFAULT 0 COMMENT '총 응답 크기(바이트)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성 시간',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정 시간',
    UNIQUE KEY idx_endpoint_date (endpoint, date),
    INDEX idx_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 이벤트 스케줄러 제거 - 로그 파일 로테이션 방식으로 변경
-- DB는 초기화하지 않고, 로그 파일만 주기적으로 로테이션 