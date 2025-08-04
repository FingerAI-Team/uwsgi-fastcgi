FROM vllm/vllm-openai:latest

# 환경 변수 설정
ENV MODEL_PATH=/app/models/gemma3-12b

# 모델 저장 디렉토리 생성
RUN mkdir -p /app/models

# 포트 노출
EXPOSE 8000

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# vLLM 서버 시작 (L40S 최적화, 원본 모델 사용)
CMD ["--model", "/app/models/gemma3-12b", "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", "0.9", "--max-model-len", "8192", "--max-num-batched-tokens", "8192", "--max-num-seqs", "24", "--enforce-eager", "--trust-remote-code", "--model-name", "gemma3:12b"] 