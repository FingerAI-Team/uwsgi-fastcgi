FROM vllm/vllm-openai:latest

# 환경 변수 설정
ENV MODEL_PATH=/app/models/mistralai/Mistral-7B-Instruct-v0.2

# 모델 저장 디렉토리 생성
RUN mkdir -p /app/models

# 포트 노출
EXPOSE 8000

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# vLLM 서버 시작 (로컬 모델 사용)
CMD ["--model", "/app/models/mistralai/Mistral-7B-Instruct-v0.2", "--host", "0.0.0.0", "--port", "8000", "--quantization", "awq", "--gpu-memory-utilization", "0.5", "--max-model-len", "1024", "--tensor-parallel-size", "1", "--disable-log-requests", "--disable-log-stats", "--max-num-batched-tokens", "1024", "--max-num-seqs", "64", "--swap-space", "2"] 