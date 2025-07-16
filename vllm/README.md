# vLLM 서비스

OpenAI 호환 API를 제공하는 vLLM 기반 LLM 서비스입니다.

## 특징

- **OpenAI 호환 API**: OpenAI API와 동일한 인터페이스 제공
- **GPU 가속**: NVIDIA GPU를 활용한 고속 추론
- **양자화 지원**: AWQ, GPTQ, SqueezeLLM 양자화 지원
- **Mistral-7B 모델**: Mistral-7B-Instruct-v0.2 모델 기본 제공

## 설치 및 실행

### 1. 모델 다운로드

Mistral-7B-Instruct-v0.2 모델을 다음 경로에 설치하세요:

```bash
mkdir -p volumes/vllm/mistralai/Mistral-7B-Instruct-v0.2
# 모델 파일들을 위 경로에 복사
```

필요한 파일들:
- `config.json`
- `pytorch_model.bin` (또는 양자화된 모델 파일)
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

### 2. 서비스 실행

#### vLLM 서비스만 실행
```bash
./scripts/setup.sh vllm
```

#### Prompt + vLLM 조합 실행
```bash
./scripts/setup.sh prompt_vllm
```

### 3. 환경 변수 설정

GPU 모드에서 다음 환경 변수를 설정할 수 있습니다:

```bash
# 양자화 사용 여부 (기본값: true)
USE_QUANTIZATION=true

# 양자화 방법 (awq, gptq, sq)
QUANTIZATION_METHOD=awq

# 사용할 GPU 번호 (기본값: 1, 두 번째 GPU)
CUDA_VISIBLE_DEVICES=1

# 최대 모델 길이 (기본값: 4096)
MAX_MODEL_LEN=4096
```

## API 사용법

### 헬스체크
```bash
curl http://localhost:8000/health
```

### 채팅 완성 (OpenAI 호환)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## 성능 최적화

### 양자화 옵션

1. **AWQ (Activation-aware Weight Quantization)**
   - 메모리 사용량: ~4GB
   - 추론 속도: 빠름
   - 품질: 양호

2. **GPTQ (Gradient-based Post-training Quantization)**
   - 메모리 사용량: ~4GB
   - 추론 속도: 빠름
   - 품질: 양호

3. **SqueezeLLM**
   - 메모리 사용량: ~3GB
   - 추론 속도: 매우 빠름
   - 품질: 양호

### GPU 메모리 설정

- `gpu_memory_utilization`: GPU 메모리 사용률 (기본값: 0.9)
- `tensor_parallel_size`: 텐서 병렬 크기 (기본값: 1)

## 문제 해결

### 모델 로드 실패
- 모델 파일이 올바른 경로에 있는지 확인
- GPU 메모리가 충분한지 확인
- CUDA 드라이버 버전 확인

### 성능 문제
- 양자화 사용 여부 확인
- GPU 메모리 사용률 조정
- 배치 크기 조정

## 로그 확인

```bash
# vLLM 서비스 로그
docker logs milvus-vllm

# 실시간 로그 모니터링
docker logs -f milvus-vllm
``` 