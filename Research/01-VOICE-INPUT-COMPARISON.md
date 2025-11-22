# Voice Input Methods - Detailed Comparison

**Research Date:** November 2025

---

## Table of Contents

1. [Commercial Solutions](#commercial-solutions)
2. [Open Source Solutions](#open-source-solutions)
3. [Comparison Matrix](#comparison-matrix)
4. [Code Examples](#code-examples)
5. [Decision Guide](#decision-guide)

---

## Commercial Solutions

### 1. Deepgram Nova-3

**Type:** Cloud API - Real-time streaming transcription

**Latency:**
- Sub-300ms median latency (P50)
- Production target: <300ms keeps UI synchronous
- >500ms feels laggy to users
- **Best for:** Interactive voice agents

**Cost (2025 Pricing):**
- Batch (Nova-3): $0.0043/minute ($4.30 per 1,000 minutes)
- Streaming (Real-time): $0.0077/minute ($7.70 per 1,000 minutes)
- High-volume: $3.60 per 1,000 minutes
- Free tier: $200 credits (up to 45,000 free minutes)

**Accuracy:**
- Nova-3 WER: 5.26% - 6.84% (production median)
- Independent benchmarks: 88-92% on clear English
- Voice agent benchmark: 18.3% WER with <300ms latency
- Non-English: 23% relative WER improvement vs competitors

**Streaming:**
- Native WebSocket-based streaming
- Live results as speech happens
- Both interim and final results
- Built from ground up for real-time

**Languages:**
- 36+ languages (Nova-2 and Nova-3)
- Includes: English, Spanish, French, German, Italian, Portuguese, Japanese, Korean, Mandarin, and 27+ more
- Multilingual code-switching support
- 90%+ accuracy across all languages

**Integration:**
- Official JavaScript SDK: `@deepgram/sdk`
- WebSocket-based, simple connection
- Compatible with LiveKit, Pipecat, Vapi
- Custom vocabulary and keyword boosting

**Code Example:**
```javascript
import { createClient, LiveTranscriptionEvents } from '@deepgram/sdk';

const deepgram = createClient(process.env.DEEPGRAM_API_KEY);

const connection = deepgram.listen.live({
  model: 'nova-3',
  language: 'en-US',
  smart_format: true,
  interim_results: true,
  endpointing: 300, // ms of silence to detect end of speech
});

connection.on(LiveTranscriptionEvents.Open, () => {
  console.log('Connection opened');

  microphone.on('data', (chunk) => {
    connection.send(chunk);
  });
});

connection.on(LiveTranscriptionEvents.Transcript, (data) => {
  const transcript = data.channel.alternatives[0].transcript;
  const isFinal = data.is_final;

  if (transcript !== '') {
    console.log(isFinal ? 'Final:' : 'Interim:', transcript);
  }
});

connection.on(LiveTranscriptionEvents.Close, () => {
  console.log('Connection closed');
});
```

**Best For:**
- Real-time voice agents
- Low-latency applications (<300ms required)
- Noisy environments
- Multilingual streaming

---

### 2. AssemblyAI Universal-Streaming

**Type:** Cloud API - Real-time transcription with AI features

**Latency:**
- Median: ~300ms (P50)
- Claims 41% faster than competitors
- Immutable transcripts (won't change mid-conversation)
- Range: 300-600ms in voice agent benchmarks

**Cost (2025 Pricing):**
- Universal-Streaming: $0.15/hour (session-based, not audio length)
- Legacy Streaming: $0.47/hour
- Charged for connection time, not audio duration
- Rate limits: Free accounts = 5 sessions/min, Paid = 100+ sessions/min
- **Most expensive** among streaming options

**Accuracy:**
- Universal-2 WER: 14.5% (best in benchmarks)
- TELUS benchmark: Best speech-to-text model tested
- Independent Ionio (2025): 9.4% WER clean, 14.1% noisy
- 30% lower hallucination vs Whisper Large-v3
- Strong in medical and sales domains

**Streaming:**
- WebSocket-based real-time streaming
- Session-based pricing model
- Universal-Streaming is latest (2025)
- Intelligent endpointing for conversation turn detection

**Languages:**
- Streaming (Beta): 6 languages (English, Spanish, French, German, Italian, Portuguese)
- Batch/Pre-recorded: Much broader support
- Most limited streaming language support
- 2025 expansion planned

**Integration:**
- Multiple SDKs: Python, JavaScript/TypeScript
- Platform integrations: LiveKit, Pipecat, Vapi
- 99.95% uptime SLA
- Word boosting available

**Code Example:**
```javascript
import { RealtimeTranscriber } from 'assemblyai';

const transcriber = new RealtimeTranscriber({
  apiKey: process.env.ASSEMBLY_API_KEY,
  sampleRate: 16000,
  wordBoost: ['NexaOS', 'LangGraph', 'agentic'],
  encoding: 'pcm_s16le',
});

transcriber.on('open', ({ sessionId }) => {
  console.log(`Session opened: ${sessionId}`);
});

transcriber.on('transcript', (transcript) => {
  if (transcript.message_type === 'FinalTranscript') {
    console.log('Final:', transcript.text);
  } else if (transcript.message_type === 'PartialTranscript') {
    console.log('Partial:', transcript.text);
  }
});

transcriber.on('error', (error) => {
  console.error('Error:', error);
});

// Connect and start
await transcriber.connect();

// Send audio data (PCM16 at 16kHz)
microphone.on('data', (audioChunk) => {
  transcriber.sendAudio(audioChunk);
});

// Close when done
await transcriber.close();
```

**Best For:**
- Applications requiring highest accuracy
- Medical/sales transcription
- When accuracy > cost
- Structured environments
- AI-powered features (sentiment, topics)

---

### 3. OpenAI Whisper API

**Type:** Cloud API - Batch transcription (not streaming)

**Latency:**
- **Not designed for real-time**: Processes complete audio files only
- Processing speed: ~35-40 audio seconds per second of processing
- Chunking workarounds introduce seconds of lag
- Best for: Batch processing, not conversations

**Cost (2025 Pricing):**
- Whisper (legacy): $0.006/minute ($0.36/hour, $6 per 1,000 minutes)
- GPT-4o Transcribe: $0.006/minute (same price, better accuracy)
- GPT-4o Mini Transcribe: $0.003/minute (budget option)
- Free tier: $5 in credits (expires in 3 months)
- **Cheapest** for batch transcription

**Accuracy:**
- VoiceWriter benchmark (2025): Best for formatted transcriptions
- Strong noise resilience (tied with AssemblyAI)
- 30% higher hallucination rate vs AssemblyAI Universal-2
- Multilingual: 65% training data was English

**Streaming:**
- **No native streaming support**
- Community workarounds: 30-second segments with chunking
- Boundary errors when stitching chunks
- WhisperX: Third-party 70x realtime batch processing

**Languages:**
- **99 languages officially supported** (most comprehensive)
- Includes all major and many minor languages
- Translation capability: Any language → English
- Best multilingual coverage

**Integration:**
- REST API (not WebSocket)
- File size limit: 25 MB
- Simple HTTP POST request
- No official SDK needed (standard HTTP)

**Code Example:**
```javascript
// Browser-side audio capture
const mediaRecorder = new MediaRecorder(stream, {
  mimeType: 'audio/webm'
});
const audioChunks = [];

mediaRecorder.ondataavailable = (event) => {
  audioChunks.push(event.data);
};

mediaRecorder.onstop = async () => {
  const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

  const formData = new FormData();
  formData.append('file', audioBlob, 'audio.webm');
  formData.append('model', 'whisper-1');
  formData.append('language', 'en'); // optional
  formData.append('response_format', 'json');

  const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
    },
    body: formData
  });

  const result = await response.json();
  console.log('Transcript:', result.text);
};

// Record for 5 seconds
mediaRecorder.start();
setTimeout(() => mediaRecorder.stop(), 5000);
```

**Best For:**
- Podcast transcription
- Video subtitles
- Meeting recordings
- Multilingual content (99 languages)
- Non-time-sensitive applications
- Batch processing

---

### 4. Web Speech API (Browser Native)

**Type:** Browser-native API (free)

**Latency:**
- Near-instant: Typically <100ms
- Very low latency (Google's servers in Chrome)
- Network dependent (requires internet in Chrome)
- **Best raw latency** when working

**Cost:**
- **Completely free** - no API costs
- Undocumented per-user limits (desktop Chrome)
- Can't scale or buy more capacity
- Hidden cost: Privacy (data sent to Google)

**Accuracy:**
- Highly variable by browser
- Chrome: Uses Google Speech (generally good)
- Safari: Different algorithm, different results
- No published benchmarks
- Inconsistent across browsers

**Streaming:**
- Real-time only (not for pre-recorded files)
- Continuous mode: `continuous: true`
- Interim results supported
- Chrome stops after inactivity (auto-restart needed)

**Languages:**
- Browser-dependent support
- BCP 47 language tags (e.g., 'en-US')
- Chrome: Extensive via Google
- Safari: More limited
- No documented list

**Integration:**
- Extremely simple: 10-20 lines of JavaScript
- No API keys or authentication
- No server needed
- Limited configuration options

**Code Example:**
```javascript
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (typeof SpeechRecognition !== 'undefined') {
  const recognition = new SpeechRecognition();

  // Configuration
  recognition.continuous = true;       // Keep listening
  recognition.interimResults = true;   // Get partial results
  recognition.lang = 'en-US';          // Language
  recognition.maxAlternatives = 1;     // Number of alternatives

  // Event handlers
  recognition.onstart = () => {
    console.log('Speech recognition started');
  };

  recognition.onresult = (event) => {
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      const confidence = event.results[i][0].confidence;

      if (event.results[i].isFinal) {
        console.log('Final:', transcript, `(${confidence})`);
      } else {
        console.log('Interim:', transcript);
      }
    }
  };

  recognition.onerror = (event) => {
    console.error('Error:', event.error);
    // Common errors: no-speech, audio-capture, not-allowed
  };

  recognition.onend = () => {
    console.log('Recognition ended');
    // Auto-restart for continuous recognition
    recognition.start();
  };

  // Start recognition
  recognition.start();

  // Stop later
  // recognition.stop();
} else {
  console.log('Web Speech API not supported');
}
```

**Browser Support (2024-2025):**
- Chrome/Chromium 25-136: Partial support
- Safari 14.1-18.4: Partial support
- Firefox: No support
- Edge: No support (despite Chromium base)
- Compatibility: 50/100

**Privacy Concerns:**
- Chrome sends audio to Google servers
- Includes website domain, language settings
- Unclear data retention policies
- No offline mode (requires internet)
- 2024 proposals for on-device recognition

**Best For:**
- Rapid prototyping
- Internal tools
- Chrome-only applications
- Low-volume personal projects
- Simple voice commands
- When cost is primary concern

---

## Open Source Solutions

### 1. faster-whisper (OpenAI Whisper Optimized)

**Type:** Self-hosted - Optimized Whisper implementation

**Latency:**
- <1 second for medium model
- 4x faster than base Whisper
- Sub-second achievable with optimizations
- Streaming: Yes, via WhisperLive or custom

**Cost:**
- 100% free (Apache 2.0 license)
- Self-hosting: ~$0.35/hour (RunPod RTX 4090)
- Or $158/month flat for 1000 hours
- **92-95% savings** vs commercial

**Accuracy:**
- 2.7% WER (best-in-class for multilingual)
- Same model as Whisper API
- Word-level timestamps
- Excellent noise resilience

**Streaming:**
- Native via WhisperLive
- Custom implementations possible
- Frame-by-frame processing
- Sliding window approach

**Languages:**
- 99 languages (same as Whisper)
- Best multilingual open-source option
- Translation supported

**Hardware Requirements:**
- Tiny model: 1GB VRAM
- Base model: 1GB VRAM
- Small model: 2GB VRAM
- Medium model: 5GB VRAM (~400 MB with int8)
- Large models: 8-11GB VRAM

**Installation:**
```bash
# Using pip
pip install faster-whisper

# Or with CUDA acceleration
pip install faster-whisper[cuda]
```

**Code Example:**
```python
from faster_whisper import WhisperModel

# Load model (runs on GPU if available)
model = WhisperModel("medium", device="cuda", compute_type="float16")

# Transcribe audio file
segments, info = model.transcribe("audio.mp3", beam_size=5)

print(f"Detected language '{info.language}' with probability {info.language_probability}")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

**Streaming Example:**
```python
from faster_whisper import WhisperModel
import numpy as np

model = WhisperModel("base", device="cuda")

# Process audio chunks
audio_buffer = []
chunk_duration = 3.0  # seconds

def process_audio_chunk(audio_chunk):
    audio_buffer.append(audio_chunk)

    # When buffer reaches chunk_duration
    if len(audio_buffer) * 0.02 >= chunk_duration:  # 20ms chunks
        audio_data = np.concatenate(audio_buffer)
        segments, _ = model.transcribe(audio_data)

        for segment in segments:
            print(segment.text)

        # Overlap for context
        audio_buffer.clear()
        audio_buffer.extend(audio_data[-100:])  # Keep last 2 seconds
```

**Best For:**
- Budget-conscious production
- Multilingual applications
- Self-hosted requirements
- High-volume processing
- Privacy-sensitive applications

---

### 2. Vosk

**Type:** Self-hosted - Lightweight offline STT

**Latency:**
- <100ms typical
- Extremely fast on CPU
- Real-time capable on Raspberry Pi
- Best latency among open-source

**Cost:**
- 100% free (Apache 2.0)
- Runs on CPU only
- Minimal infrastructure cost
- ~$0.01 per hour estimated

**Accuracy:**
- Good (not as accurate as Whisper)
- Acceptable for most applications
- Language-specific models vary
- Trade accuracy for speed/size

**Streaming:**
- Native streaming support
- Frame-by-frame processing
- Real-time design

**Languages:**
- 20+ languages with pre-trained models
- Includes: English, Spanish, French, German, Russian, Chinese, etc.
- Download language-specific models

**Hardware Requirements:**
- **Tiny:** 50MB models, 512MB RAM
- CPU-only operation
- Runs on Raspberry Pi
- Perfect for edge devices

**Installation:**
```bash
pip install vosk

# Download models
# From https://alphacephei.com/vosk/models
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
stream.start_stream()

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

**Best For:**
- Edge devices (Raspberry Pi, mobile)
- Offline applications
- CPU-only environments
- Real-time processing on minimal hardware
- When size/speed > accuracy

---

### 3. Wav2Vec 2.0 (Meta)

**Type:** Self-hosted - Transformer-based ASR

**Latency:**
- Low latency
- Native streaming support
- Best for real-time applications
- Frame-by-frame capable

**Cost:**
- 100% free (open source)
- Requires GPU for real-time
- ~$0.35/hour cloud GPU

**Accuracy:**
- 1.77% WER on clean audio
- Excellent on LibriSpeech benchmark
- Competitive with commercial solutions
- Fine-tunable for domains

**Streaming:**
- Native streaming support
- Designed for real-time
- Frame-level predictions

**Languages:**
- English models best supported
- Multilingual models available
- 53 languages in XLSR-53
- Fine-tune for specific languages

**Hardware Requirements:**
- GPU recommended for real-time
- Base model: 4-6GB VRAM
- Large model: 8-12GB VRAM

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

# Resample if necessary
if rate != 16000:
    resampler = torchaudio.transforms.Resample(rate, 16000)
    audio = resampler(audio)

# Process
input_values = processor(audio.squeeze(), sampling_rate=16000, return_tensors="pt").input_values

# Transcribe
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("Transcription:", transcription)
```

**Best For:**
- Research applications
- Fine-tuning for specific domains
- Real-time streaming
- When GPU available

---

## Comparison Matrix

### Performance Comparison

| Solution | Latency | Accuracy (WER) | Streaming | Real-time | GPU Required |
|----------|---------|----------------|-----------|-----------|--------------|
| **Deepgram Nova-3** | <300ms | 5.3% | ✅ Native | ✅ Yes | ❌ No (cloud) |
| **AssemblyAI Universal** | ~300ms | 14.5% | ✅ Native | ✅ Yes | ❌ No (cloud) |
| **Whisper API** | 35-40s/file | Best formatted | ❌ Workarounds | ❌ No | ❌ No (cloud) |
| **Web Speech API** | <100ms | Variable | ✅ Native | ✅ Yes | ❌ No (browser) |
| **faster-whisper** | <1s | 2.7% | ✅ Custom | ⚠️ Possible | ✅ Recommended |
| **Vosk** | <100ms | Good | ✅ Native | ✅ Yes | ❌ No (CPU) |
| **Wav2Vec 2.0** | Low | 1.77% | ✅ Native | ✅ Yes | ✅ Recommended |

### Cost Comparison (per 1000 hours)

| Solution | Cost | Infrastructure | Setup Complexity |
|----------|------|----------------|------------------|
| **Deepgram** | $462 | None (cloud) | Easy |
| **AssemblyAI** | $150-282 | None (cloud) | Easy |
| **Whisper API** | $360 | None (cloud) | Easy |
| **Web Speech API** | $0 | None (browser) | Trivial |
| **faster-whisper** | $158 | GPU rental | Medium |
| **Vosk** | ~$10 | CPU only | Easy |
| **Wav2Vec 2.0** | $158 | GPU rental | Hard |

### Language Support

| Solution | Languages | Multilingual | Translation |
|----------|-----------|--------------|-------------|
| **Deepgram** | 36+ | ✅ Yes | ❌ No |
| **AssemblyAI** | 6 (streaming) | ⚠️ Limited | ❌ No |
| **Whisper API** | 99 | ✅ Excellent | ✅ Yes |
| **Web Speech API** | Browser-dependent | ⚠️ Varies | ❌ No |
| **faster-whisper** | 99 | ✅ Excellent | ✅ Yes |
| **Vosk** | 20+ | ✅ Yes | ❌ No |
| **Wav2Vec 2.0** | 53 | ✅ Yes | ❌ No |

---

## Decision Guide

### Choose Deepgram if:
- ✅ You need real-time voice agents
- ✅ Sub-300ms latency is critical
- ✅ You want easy integration
- ✅ Budget allows $0.46/hour
- ✅ Production reliability matters

### Choose AssemblyAI if:
- ✅ Accuracy is top priority
- ✅ You need AI features (sentiment, topics)
- ✅ Medical/sales domain
- ✅ Session-based usage pattern works
- ✅ Budget allows $0.15-0.47/hour

### Choose Whisper API if:
- ✅ Batch processing (not real-time)
- ✅ Need 99 language support
- ✅ Translation required
- ✅ Best formatting needed
- ✅ Budget-conscious ($0.36/hour)

### Choose Web Speech API if:
- ✅ Quick prototype/demo
- ✅ Chrome-only is acceptable
- ✅ Zero budget
- ✅ Internal tool (not customer-facing)
- ❌ Don't need production quality

### Choose faster-whisper if:
- ✅ High volume (>1000 hours/month)
- ✅ 92% cost savings needed
- ✅ Multilingual (99 languages)
- ✅ Self-hosting acceptable
- ✅ GPU available

### Choose Vosk if:
- ✅ Edge devices (Raspberry Pi)
- ✅ Offline required
- ✅ CPU-only environment
- ✅ Minimal resource usage
- ⚠️ Can accept lower accuracy

### Choose Wav2Vec 2.0 if:
- ✅ Research/fine-tuning project
- ✅ Streaming required
- ✅ GPU available
- ✅ Domain-specific optimization needed

---

## Code Examples

### Complete Voice Input Pipeline (Commercial)

```javascript
// Full implementation with Deepgram + React
import { useEffect, useRef, useState } from 'react';
import { createClient, LiveTranscriptionEvents } from '@deepgram/sdk';

function VoiceInput({ onTranscript }) {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorder = useRef(null);
  const deepgram = useRef(null);
  const connection = useRef(null);

  useEffect(() => {
    deepgram.current = createClient(process.env.NEXT_PUBLIC_DEEPGRAM_API_KEY);
  }, []);

  const startRecording = async () => {
    // Get microphone access
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder.current = new MediaRecorder(stream, {
      mimeType: 'audio/webm'
    });

    // Connect to Deepgram
    connection.current = deepgram.current.listen.live({
      model: 'nova-3',
      language: 'en-US',
      smart_format: true,
      interim_results: true,
      endpointing: 300,
    });

    connection.current.on(LiveTranscriptionEvents.Open, () => {
      console.log('Deepgram connection opened');

      // Send audio data to Deepgram
      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          connection.current.send(event.data);
        }
      };

      mediaRecorder.current.start(250); // Send chunks every 250ms
    });

    connection.current.on(LiveTranscriptionEvents.Transcript, (data) => {
      const transcript = data.channel.alternatives[0].transcript;
      const isFinal = data.is_final;

      if (transcript && transcript !== '') {
        onTranscript({ text: transcript, isFinal });
      }
    });

    connection.current.on(LiveTranscriptionEvents.Error, (error) => {
      console.error('Deepgram error:', error);
    });

    setIsRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
      mediaRecorder.current.stop();
      mediaRecorder.current.stream.getTracks().forEach(track => track.stop());
    }

    if (connection.current) {
      connection.current.finish();
    }

    setIsRecording(false);
  };

  return (
    <div>
      <button onClick={isRecording ? stopRecording : startRecording}>
        {isRecording ? '⏹ Stop' : '🎤 Start'} Recording
      </button>
    </div>
  );
}

export default VoiceInput;
```

### Complete Voice Input Pipeline (Open Source)

```python
# Full implementation with faster-whisper + Python
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue

class VoiceInput:
    def __init__(self, model_size="medium", device="cuda"):
        self.model = WhisperModel(model_size, device=device, compute_type="float16")
        self.audio_queue = queue.Queue()
        self.is_recording = False

        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        self.pyaudio = pyaudio.PyAudio()

    def start_recording(self, callback):
        """Start recording and transcribing in real-time"""
        self.is_recording = True

        # Start audio capture thread
        capture_thread = threading.Thread(target=self._capture_audio)
        capture_thread.start()

        # Start transcription thread
        transcribe_thread = threading.Thread(target=self._transcribe_audio, args=(callback,))
        transcribe_thread.start()

    def _capture_audio(self):
        """Capture audio from microphone"""
        stream = self.pyaudio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        print("Recording...")

        while self.is_recording:
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_queue.put(audio_data)

        stream.stop_stream()
        stream.close()

    def _transcribe_audio(self, callback):
        """Transcribe audio chunks"""
        audio_buffer = []
        buffer_duration = 3.0  # seconds
        samples_per_chunk = int(buffer_duration * self.RATE)

        while self.is_recording or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)

                # When buffer is large enough
                if len(audio_buffer) * self.CHUNK >= samples_per_chunk:
                    audio_data = np.concatenate(audio_buffer)

                    # Transcribe
                    segments, info = self.model.transcribe(audio_data, beam_size=5)

                    for segment in segments:
                        if segment.text.strip():
                            callback(segment.text.strip())

                    # Keep last second for context
                    overlap_samples = self.RATE  # 1 second
                    audio_buffer = [audio_data[-overlap_samples:]]

            except queue.Empty:
                continue

    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        print("Stopped recording")

# Usage
def on_transcript(text):
    print(f"Transcript: {text}")

voice_input = VoiceInput(model_size="medium", device="cuda")
voice_input.start_recording(on_transcript)

# Record for 30 seconds
import time
time.sleep(30)

voice_input.stop_recording()
```

---

## Additional Resources

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [Deepgram Documentation](https://developers.deepgram.com/)
- [AssemblyAI Documentation](https://www.assemblyai.com/docs)
- [Vosk Documentation](https://alphacephei.com/vosk/)
- [Wav2Vec 2.0 Paper](https://arxiv.org/abs/2006.11477)
- [Web Speech API Spec](https://wicg.github.io/speech-api/)

---

**Last Updated:** November 21, 2025
