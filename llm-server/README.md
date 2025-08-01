# LLM Server

L40S GPU 최적화된 vLLM 기반 LLM 서버입니다. Gemma3 12B 모델을 서빙합니다.

## 특징

- **GPU 최적화**: L40S 48GB VRAM에 최적화된 설정
- **모델**: Gemma3 12B (AWQ 양자화)
- **프레임워크**: vLLM
- **포트**: 11436

## L40S 최적화 설정

- `GPU_MEMORY_UTILIZATION=0.9`: 48GB 중 90% 사용
- `MAX_GPU_MEMORY=40GB`: 최대 GPU 메모리 사용량
- `MAX_NUM_BATCHED_TOKENS=8192`: 배치 토큰 수
- `MAX_NUM_SEQS=24`: 최대 동시 요청 수
- `MAX_MODEL_LEN=8192`: 최대 모델 길이

## 모델 요구사항

모델 파일은 다음 경로에 배치해야 합니다:
```
./models/gemma3-12b/
```

필수 파일:
- `config.json`
- `model.safetensors` (또는 양자화된 모델 파일)
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

## 사용법

```bash
# llm-server만 실행
docker compose --profile llm-server-only up -d

# 전체 시스템과 함께 실행 (all-llm 옵션)
./scripts/setup.sh all-llm
```

## API 엔드포인트

- **Health Check**: `http://localhost:11436/health`
- **Chat Completion**: `http://localhost:11436/v1/chat/completions`
- **Text Generation**: `http://localhost:11436/v1/completions`

## GPU 설정

GPU 1번을 사용하도록 설정되어 있습니다 (GPU 0번은 다른 서비스가 사용). 