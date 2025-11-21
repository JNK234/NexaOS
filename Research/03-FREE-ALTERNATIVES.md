# Free & Open Source Voice Agent Alternatives

**Research Date:** November 2025

---

## Table of Contents

1. [Free Speech-to-Text](#free-speech-to-text)
2. [Free Text-to-Speech](#free-text-to-speech)
3. [Complete Open Source Stacks](#complete-open-source-stacks)
4. [Free Infrastructure](#free-infrastructure)
5. [Cost Comparison](#cost-comparison)
6. [Quick Start Guides](#quick-start-guides)

---

## Free Speech-to-Text

### 1. faster-whisper ⭐ WINNER

**Technology:** Optimized OpenAI Whisper with CTranslate2

**Cost:** 100% free (Apache 2.0 license)

**Accuracy:**
- 2.7% WER (best-in-class for multilingual)
- Same model as Whisper API
- Word-level timestamps
- Excellent noise resilience

**Latency:**
- <1 second for medium model
- **4x faster than base Whisper**
- Sub-second achievable with optimization
- Streaming: Yes, via WhisperLive

**Languages:** 99 languages (same as Whisper)

**Hardware Requirements:**
| Model | VRAM | Quality |
|-------|------|---------|
| Tiny | 1GB | Basic |
| Base | 1GB | Good |
| Small | 2GB | Better |
| Medium | 5GB (~400MB int8) | Excellent |
| Large-v2 | 8-11GB | Best |

**Installation:**
```bash
pip install faster-whisper

# With CUDA acceleration
pip install faster-whisper[cuda]
```

**Code Example:**
```python
from faster_whisper import WhisperModel

# Load model (auto-downloads)
model = WhisperModel("medium", device="cuda", compute_type="float16")

# Transcribe
segments, info = model.transcribe("audio.mp3", beam_size=5)

print(f"Language: {info.language} ({info.language_probability:.2f})")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**Streaming Setup:**
```python
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda")

audio_buffer = []
CHUNK_DURATION = 3.0  # seconds

def process_audio_chunk(chunk):
    audio_buffer.append(chunk)

    if len(audio_buffer) >= CHUNK_DURATION * SAMPLE_RATE:
        audio_data = np.concatenate(audio_buffer)

        # Transcribe
        segments, _ = model.transcribe(audio_data)

        for segment in segments:
            yield segment.text

        # Keep overlap for context
        audio_buffer.clear()
        audio_buffer.extend(audio_data[-SAMPLE_RATE * 2:])  # 2s overlap
```

**Cost Analysis:**
- Self-hosting (RTX 4090): $0.35/hour = **$0.006/min**
- Commercial Whisper API: $0.36/hour = **$0.006/min**
- **Savings:** At scale, own hardware pays off in 6-12 months

**Best For:**
- Production quality needed
- Budget constraints
- Multilingual (99 languages)
- High volume (>1000 hours/month)

---

### 2. Vosk

**Technology:** Lightweight offline STT based on Kaldi

**Cost:** 100% free (Apache 2.0)

**Accuracy:** Good (not as accurate as Whisper, but acceptable)

**Latency:**
- <100ms typical
- **Extremely fast on CPU**
- Real-time on Raspberry Pi
- Best latency among open-source

**Languages:** 20+ with pre-trained models
- English, Spanish, French, German, Russian, Chinese, Japanese, etc.

**Hardware Requirements:**
- **Tiny:** 50MB models, 512MB RAM
- **CPU-only operation**
- Runs on Raspberry Pi
- Perfect for edge devices

**Installation:**
```bash
pip install vosk

# Download models from https://alphacephei.com/vosk/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

**Code Example:**
```python
from vosk import Model, KaldiRecognizer
import pyaudio
import json

# Load model
model = Model("vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

# Capture audio
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1,
                  rate=16000, input=True, frames_per_buffer=8000)

print("Listening...")

while True:
    data = stream.read(4000, exception_on_overflow=False)

    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        print("Final:", result['text'])
    else:
        partial = json.loads(recognizer.PartialResult())
        print("Partial:", partial['partial'])
```

**Cost:** ~$0.01 per hour (minimal CPU usage)

**Best For:**
- Edge devices (Raspberry Pi, mobile)
- Offline required
- CPU-only environments
- Minimal resource usage
- When speed > accuracy

---

### 3. Wav2Vec 2.0 (Meta)

**Technology:** Transformer-based ASR model

**Cost:** 100% free (open source)

**Accuracy:**
- 1.77% WER on clean audio (LibriSpeech)
- Excellent performance
- Competitive with commercial

**Latency:**
- Low latency
- Native streaming support
- Frame-by-frame predictions

**Languages:**
- English best supported
- Multilingual: XLSR-53 (53 languages)
- Fine-tunable for specific languages

**Hardware Requirements:**
- GPU recommended for real-time
- Base: 4-6GB VRAM
- Large: 8-12GB VRAM

**Installation:**
```bash
pip install transformers torch torchaudio
```

**Code Example:**
```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio
audio, rate = torchaudio.load("audio.wav")

# Resample to 16kHz
if rate != 16000:
    resampler = torchaudio.transforms.Resample(rate, 16000)
    audio = resampler(audio)

# Process
input_values = processor(audio.squeeze(), sampling_rate=16000,
                         return_tensors="pt").input_values

# Transcribe
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print(transcription)
```

**Best For:**
- Research applications
- Fine-tuning for domains
- Streaming required
- GPU available

---

## Free Text-to-Speech

### 1. Kokoro-82M ⭐ NEW 2025 WINNER

**Technology:** Indie TTS model (StyleTTS 2 architecture)

**Cost:** 100% free (Apache 2.0)

**Quality:**
- **#1 ranked on HuggingFace TTS Arena** (blind tests)
- Beats commercial services in quality tests
- Natural, expressive voices
- Multiple speaker options

**Latency:**
- Ultra-fast
- Real-time capable **on CPU**
- Sub-second synthesis
- No GPU required

**Size:** Only 82M parameters (incredibly efficient)

**Hardware Requirements:**
- **CPU-only operation**
- Minimal resources
- <$0.06 per hour of audio output

**Installation:**
```bash
pip install kokoro-onnx

# Or from source
git clone https://github.com/thewh1teagle/kokoro-onnx
cd kokoro-onnx
pip install -e .
```

**Code Example:**
```python
from kokoro import Kokoro

# Initialize
kokoro = Kokoro()

# Generate speech
audio = kokoro.generate(
    text="Hello! This is Kokoro, the highest-rated open-source TTS model.",
    voice="af_sky"  # Multiple voices available
)

# Save or stream
kokoro.save(audio, "output.wav")
```

**Available Voices:**
- af (American Female)
- af_bella
- af_sarah
- af_nicole
- af_sky
- am (American Male)
- am_adam
- am_michael
- bf (British Female)
- bm (British Male)

**Cost:** <$0.06/hour of audio

**Best For:**
- Production quality needed
- CPU-only environments
- Budget priority
- 2025 cutting-edge solution

---

### 2. Piper TTS

**Technology:** Fast neural TTS

**Cost:** 100% free (MIT license)

**Quality:**
- **Most natural-sounding among fast models**
- Good prosody
- Multiple voices per language
- Acceptable for production

**Latency:**
- **10x faster than cloud solutions**
- Sub-second synthesis
- Real-time capable
- **CPU-only**

**Languages:** 20+ languages supported

**Hardware Requirements:**
- **CPU-only operation**
- Runs on Raspberry Pi
- Minimal resources
- ~100MB per voice model

**Installation:**
```bash
pip install piper-tts

# Or download binary
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz
tar -xvf piper_linux_x86_64.tar.gz
```

**Code Example:**
```python
from piper import PiperVoice

# Load voice
voice = PiperVoice.load(
    "en_US-lessac-medium.onnx",
    config_path="en_US-lessac-medium.onnx.json",
    use_cuda=False
)

# Synthesize
audio = voice.synthesize("Hello! This is Piper text-to-speech.")

# Save
with open("output.wav", "wb") as f:
    voice.synthesize_stream_raw("Hello!", f)
```

**Streaming Example:**
```python
import wave

# Stream synthesis
with wave.open("output.wav", "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(22050)

    for audio_chunk in voice.synthesize_stream("Long text here..."):
        wav_file.writeframes(audio_chunk)
```

**Cost:** ~$0.01/hour of audio

**Best For:**
- **Overall best for production voice agents**
- Fast synthesis required
- CPU-only environments
- Edge devices
- Multiple languages needed

---

### 3. XTTS-v2 (Coqui/Idiap)

**Technology:** Advanced voice cloning TTS

**Cost:** 100% free (MPL 2.0 license)

**Quality:**
- Excellent, near-human
- **Voice cloning from 6-second sample** (85-95% similarity)
- Emotional tone replication
- Production-quality

**Latency:**
- <150ms streaming latency
- Real-time capable
- GPU required for best performance

**Languages:** 17 languages supported

**Voice Cloning:**
- **6 seconds of audio** needed
- Cross-language cloning
- Emotional characteristics preserved

**Hardware Requirements:**
- **GPU required:** 8GB+ VRAM recommended
- Can run on CPU (slower)
- 2GB+ model size

**Installation:**
```bash
pip install TTS

# Or from source
git clone https://github.com/idiap/coqui-ai-TTS
cd coqui-ai-TTS
pip install -e .
```

**Code Example:**
```python
from TTS.api import TTS

# Initialize XTTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Generate with default voice
tts.tts_to_file(
    text="Hello! This is XTTS voice synthesis.",
    file_path="output.wav",
    language="en"
)

# Voice cloning from sample
tts.tts_to_file(
    text="This will sound like the reference speaker.",
    file_path="cloned_output.wav",
    speaker_wav="reference_voice.wav",  # 6+ second sample
    language="en"
)
```

**Streaming Example:**
```python
import torch

# Streaming synthesis
for i, audio_chunk in enumerate(tts.tts_stream(
    text="Long text for streaming...",
    speaker_wav="reference.wav",
    language="en"
)):
    # Process chunk (send to client, play, etc.)
    print(f"Chunk {i}: {len(audio_chunk)} samples")
```

**Cost:** ~$0.10/hour (GPU rental)

**Note:** Coqui AI shut down in Dec 2024, but Idiap Research Institute forked and maintains XTTS.

**Best For:**
- Voice cloning needed
- High quality required
- Multilingual (17 languages)
- Emotional expression important
- GPU available

---

### 4. Browser SpeechSynthesis API

**Technology:** Browser-native TTS

**Cost:** 100% free

**Quality:** Fair (robotic, but acceptable for prototypes)

**Latency:** Very low (<50ms)

**Languages:** Browser-dependent (usually 20+)

**Hardware Requirements:** None (browser-based)

**Code Example:**
```javascript
// Simple usage
const utterance = new SpeechSynthesisUtterance("Hello! This is browser TTS.");
utterance.rate = 1.0;
utterance.pitch = 1.0;
utterance.volume = 1.0;

speechSynthesis.speak(utterance);

// Get available voices
const voices = speechSynthesis.getVoices();
console.log(voices);

// Use specific voice
utterance.voice = voices.find(v => v.lang === 'en-US');
speechSynthesis.speak(utterance);

// Events
utterance.onstart = () => console.log('Speaking started');
utterance.onend = () => console.log('Speaking ended');
utterance.onerror = (e) => console.error('Error:', e);
```

**Best For:**
- Quick prototypes
- Demos
- Zero setup needed
- When quality isn't critical

---

## Complete Open Source Stacks

### Stack 1: Optimal 2025 Production Stack ⭐

**Components:**
- STT: faster-whisper (medium)
- LLM: Llama 3.1 8B (via Ollama)
- TTS: Kokoro-82M
- Framework: Pipecat

**Performance:**
- Sub-second total latency
- Production quality
- Cost: $158/month for 1000 hours

**Setup Time:** 1-2 hours

**Installation:**
```bash
# 1. Install components
pip install faster-whisper pipecat-ai

# 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1

# 3. Install Kokoro
pip install kokoro-onnx

# 4. Clone Pipecat examples
git clone https://github.com/pipecat-ai/pipecat
cd pipecat/examples
```

**Code Example:**
```python
from pipecat.pipeline import Pipeline
from pipecat.services.whisper import WhisperSTT
from pipecat.services.ollama import OllamaLLM
from pipecat.services.kokoro import KokoroTTS
from pipecat.transports.websocket import WebsocketTransport

async def main():
    pipeline = Pipeline([
        WebsocketTransport(),
        WhisperSTT(model_size="medium"),
        OllamaLLM(model="llama3.1"),
        KokoroTTS(voice="af_sky")
    ])

    await pipeline.run()

# Run it
import asyncio
asyncio.run(main())
```

**Cost Breakdown (1000 hours/month):**
- RunPod RTX 4090: $0.35/hour × 450 hours = **$158/month**
- vs Commercial APIs: $2,400-4,600/month
- **Savings: 93-97%**

---

### Stack 2: Budget CPU-Only Stack

**Components:**
- STT: Vosk
- LLM: Llama 3.1 8B (quantized)
- TTS: Piper
- Framework: Custom Python

**Performance:**
- 1-2 second latency
- Acceptable quality
- Cost: $20-40/month

**Hardware:** CPU-only cloud instance

**Installation:**
```bash
pip install vosk piper-tts

# Download models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.onnx

ollama pull llama3.1
```

**Best For:**
- Minimal budget
- No GPU available
- Edge deployment

---

### Stack 3: High-Quality Voice Cloning Stack

**Components:**
- STT: faster-whisper (large)
- LLM: Llama 3.1 70B
- TTS: XTTS-v2 (voice cloning)
- Framework: Vocode

**Performance:**
- Best quality
- Voice cloning support
- Cost: $0.50-0.80/hour

**Hardware:** High-end GPU (A100 or RTX 4090)

**Best For:**
- Custom voices needed
- Premium quality
- Multilingual with cloning

---

## Free Infrastructure

### 1. Kaggle ⭐ BEST FREE GPU

**Offer:**
- **30 GPU-hours per week** (T4 or P100)
- **100% free**, no compute unit limits
- 9-hour sessions with background execution
- **Most generous in 2025**

**Specs:**
- T4: 16GB VRAM
- P100: 16GB VRAM
- 2x CPU cores
- 30GB RAM
- 100GB storage

**Limitations:**
- Not for production deployment
- Session-based (9 hours max)
- Internet access during execution only

**Use Cases:**
- Development
- Testing
- Model fine-tuning
- Experimentation

**Setup:**
```bash
# In Kaggle notebook
!pip install faster-whisper piper-tts

# Your code here
```

---

### 2. Google Colab

**Free Tier:**
- Limited GPU hours (varies)
- T4 GPUs when available
- 12-hour sessions
- Notebook-based

**Colab Pro ($10/month):**
- More GPU hours
- Better GPUs (V100, A100)
- Background execution
- Longer sessions

**Best For:**
- Development
- One-off tasks
- Not for production

---

### 3. RunPod (Cheapest Paid GPU)

**Pricing (2025):**
- RTX 3090: $0.18-0.22/hour
- RTX 4090: $0.34-0.39/hour
- A40: $0.49-0.54/hour
- A100: $1.09-1.19/hour

**Best Value:** RTX 4090 at $0.35/hour
- 24GB VRAM
- Excellent performance
- Good for production

**Break-even Analysis:**
- Own RTX 4090: $1,500-2,000
- Break-even: 4,300-5,700 hours (6-8 months at 1000 hrs/month)

---

### 4. TensorDock

**Pricing:**
- RTX 3090: $0.18/hour
- RTX 4090: $0.35/hour
- Similar to RunPod

**Features:**
- Per-second billing
- API access
- Docker support

---

### 5. Vast.ai

**Model:** Peer-to-peer GPU marketplace

**Pricing:**
- RTX 3090: $0.15-0.30/hour
- RTX 4090: $0.30-0.50/hour
- Cheapest but less reliable

**Considerations:**
- Variable reliability
- Individual hosts
- Best for development, not production

---

## Cost Comparison

### STT Cost Comparison (per 1000 hours)

| Solution | Infrastructure | Total Cost | vs Deepgram ($462) |
|----------|----------------|------------|---------------------|
| **faster-whisper** | RunPod RTX 4090 | **$158** | 66% savings |
| **faster-whisper** | Own GPU | **$0** (amortized) | 100% savings |
| **Vosk** | CPU ($20/mo) | **$20** | 96% savings |
| **Deepgram** | None | **$462** | Baseline |
| **AssemblyAI** | None | **$150-282** | 35-39% savings |

### TTS Cost Comparison (per 1000 hours)

| Solution | Infrastructure | Total Cost | vs ElevenLabs ($1,800) |
|----------|----------------|------------|------------------------|
| **Kokoro** | CPU ($20/mo) | **$20** | 99% savings |
| **Piper** | CPU ($20/mo) | **$20** | 99% savings |
| **XTTS-v2** | RunPod RTX 4090 | **$158** | 91% savings |
| **ElevenLabs** | None | **$1,800** | Baseline |
| **OpenAI TTS** | None | **$900** | Baseline |

### Complete Stack Cost (1000 hours/month)

| Stack | STT | LLM | TTS | Total | Savings |
|-------|-----|-----|-----|-------|---------|
| **Open Source (GPU)** | $158 | $0 | $0 | **$158** | 93-97% |
| **Open Source (CPU)** | $20 | $0 | $0 | **$20** | 99% |
| **Deepgram + OpenAI + ElevenLabs** | $462 | $360 | $1,800 | **$2,622** | Baseline |
| **Retell AI** | Inc | $360 | Inc | **$8,400** | -220% |

**Key Insight:** Self-hosted open source reduces costs by 90-99%

---

## Quick Start Guides

### Quick Start 1: Fastest Setup (30 minutes)

```bash
# 1. Install everything
pip install faster-whisper kokoro-onnx pipecat-ai
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1

# 2. Test each component
python -c "from faster_whisper import WhisperModel; m = WhisperModel('tiny'); print('Whisper OK')"
python -c "from kokoro import Kokoro; k = Kokoro(); print('Kokoro OK')"
ollama run llama3.1 "Hello"  # Test LLM

# 3. Clone Pipecat examples
git clone https://github.com/pipecat-ai/pipecat
cd pipecat/examples

# 4. Run simple example
python foundational/01-say-one-thing.py
```

---

### Quick Start 2: Voice Cloning Setup (1 hour)

```bash
# 1. Install XTTS
pip install TTS

# 2. Record 6-second voice sample
# Save as reference_voice.wav

# 3. Test voice cloning
python test_voice_clone.py
```

```python
# test_voice_clone.py
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Clone voice
tts.tts_to_file(
    text="Hello! This is my cloned voice speaking.",
    file_path="cloned_output.wav",
    speaker_wav="reference_voice.wav",
    language="en"
)

print("Voice cloned! Check cloned_output.wav")
```

---

### Quick Start 3: Production Deployment (2-3 hours)

```bash
# 1. Rent RunPod RTX 4090 GPU
# Go to runpod.io, create pod

# 2. SSH into pod and install
pip install faster-whisper kokoro-onnx
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1

# 3. Clone your agent code
git clone https://your-repo.git
cd your-repo

# 4. Set up WebSocket server
python server.py

# 5. Expose with ngrok or set up proper domain
ngrok http 8080

# 6. Test from frontend
```

---

## Best Practices

### 1. Model Selection

**Development:**
- STT: Whisper tiny/base
- LLM: Llama 3.1 8B
- TTS: Kokoro or Piper

**Production:**
- STT: faster-whisper medium
- LLM: Llama 3.1 8B or 70B
- TTS: Kokoro (CPU) or XTTS-v2 (GPU, if voice cloning needed)

### 2. Infrastructure

**<100 hours/month:** Use free Kaggle GPU
**100-500 hours/month:** RunPod on-demand
**>500 hours/month:** Consider buying GPU

### 3. Optimization

**STT:**
- Use int8 quantization for Whisper (3x faster)
- Implement VAD to avoid processing silence
- Use smaller models when possible

**LLM:**
- Use quantized models (4-bit or 8-bit)
- Implement prompt caching
- Use smaller models (8B) for most cases

**TTS:**
- Cache common phrases
- Use streaming synthesis
- Batch requests when possible

### 4. Monitoring

- Track latency at each stage
- Monitor GPU usage
- Log errors and failures
- Measure quality periodically

---

## 2024-2025 Updates

**Major Developments:**
1. **Kokoro-82M released (2025)** - tops HuggingFace TTS Arena
2. **Coqui AI shutdown (Dec 2024)** - Idiap Research forked XTTS
3. **Quality parity reached** - open source now matches commercial
4. **Mistral Voxtral** - new open ASR model (July 2025)
5. **DeepSpeech archived** - Mozilla officially discontinued (June 2025)
6. **22% of YC batch** building with voice AI (2025)

**Key Insights:**
- Open source TTS now **beats commercial** in blind tests (Kokoro)
- faster-whisper 4x speed improvement makes real-time practical
- GPU prices dropped 20-30% in 2024-2025
- Free tier options (Kaggle) most generous ever

---

## Conclusion

**You can build production-quality voice agents entirely free/open-source in 2025.**

**Recommended Free Stack:**
- STT: faster-whisper
- LLM: Llama 3.1 via Ollama
- TTS: Kokoro-82M
- Framework: Pipecat
- Infrastructure: Kaggle (dev) → RunPod (production)

**Cost:** $0 (dev) or $158/month (production at 1000 hours)
**Quality:** Competitive with commercial services
**Setup:** 1-2 hours

The open-source voice AI ecosystem is now mature, production-ready, and rapidly improving.

---

**Last Updated:** November 21, 2025
