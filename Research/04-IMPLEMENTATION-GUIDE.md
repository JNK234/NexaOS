# Voice Agent Implementation Guide

**Research Date:** November 2025

---

## Table of Contents

1. [Frontend Implementation](#frontend-implementation)
2. [Backend Implementation](#backend-implementation)
3. [WebSocket vs REST](#websocket-vs-rest)
4. [Voice Activity Detection](#voice-activity-detection)
5. [Interruption Handling](#interruption-handling)
6. [State Management](#state-management)
7. [Complete Examples](#complete-examples)

---

## Frontend Implementation

### React/Next.js Voice Recording Setup

#### Option 1: Using @ricky0123/vad-react (Recommended)

**Features:**
- Built-in Voice Activity Detection
- Automatic speech detection
- React hooks for easy integration

**Installation:**
```bash
npm install @ricky0123/vad-react
```

**Code Example:**
```typescript
// components/VoiceChat.tsx
import { useMicVAD } from '@ricky0123/vad-react';
import { useState, useEffect } from 'react';

export function VoiceChat() {
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);

  const vad = useMicVAD({
    startOnLoad: false,

    onSpeechStart: () => {
      console.log("User started speaking");
      if (isAgentSpeaking) {
        // Interrupt agent
        stopAgentSpeech();
      }
    },

    onSpeechEnd: async (audio) => {
      console.log("User stopped speaking");

      // Convert Float32Array to Blob
      const audioBlob = await float32ArrayToBlob(audio);

      // Send to backend
      await sendAudioToBackend(audioBlob);
    },

    onVADMisfire: () => {
      console.log("False alarm - no speech detected");
    },

    // Configuration
    positiveSpeechThreshold: 0.8,  // Higher = less sensitive
    negativeSpeechThreshold: 0.3,  // Lower = quicker end detection
    minSpeechFrames: 3,            // Min frames to trigger start
    redemptionFrames: 8            // Frames to wait before ending
  });

  async function float32ArrayToBlob(audio: Float32Array): Promise<Blob> {
    // Convert Float32Array to Int16Array (PCM)
    const pcm = new Int16Array(audio.length);
    for (let i = 0; i < audio.length; i++) {
      pcm[i] = Math.max(-32768, Math.min(32767, audio[i] * 32768));
    }

    return new Blob([pcm.buffer], { type: 'audio/pcm' });
  }

  async function sendAudioToBackend(audioBlob: Blob) {
    const formData = new FormData();
    formData.append('audio', audioBlob);

    const response = await fetch('/api/voice', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    // Add to messages
    setMessages(prev => [
      ...prev,
      { role: 'user', content: data.transcript },
      { role: 'assistant', content: data.response }
    ]);

    // Play agent response
    await playAudio(data.audioUrl);
  }

  async function playAudio(url: string) {
    setIsAgentSpeaking(true);

    const audio = new Audio(url);
    audio.onended = () => setIsAgentSpeaking(false);
    await audio.play();
  }

  function stopAgentSpeech() {
    // Interrupt currently playing audio
    const audioElements = document.querySelectorAll('audio');
    audioElements.forEach(audio => {
      audio.pause();
      audio.currentTime = 0;
    });
    setIsAgentSpeaking(false);
  }

  return (
    <div className="voice-chat">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>

      <button
        onClick={() => vad.listening ? vad.pause() : vad.start()}
        disabled={vad.loading}
      >
        {vad.loading ? 'Loading...' : vad.listening ? '⏹ Stop' : '🎤 Start'}
      </button>

      {vad.userSpeaking && <div className="indicator">🔴 Speaking...</div>}
      {isAgentSpeaking && <div className="indicator">🤖 Agent speaking...</div>}
    </div>
  );
}
```

---

#### Option 2: Custom MediaRecorder Implementation

**For more control over audio format and chunking:**

```typescript
// hooks/useVoiceRecorder.ts
import { useState, useRef, useCallback } from 'react';

export function useVoiceRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const stream = useRef<MediaStream | null>(null);

  const startRecording = useCallback(async () => {
    try {
      // Request microphone access
      stream.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        }
      });

      // Create MediaRecorder with Opus codec
      mediaRecorder.current = new MediaRecorder(stream.current, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000
      });

      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      mediaRecorder.current.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, {
          type: 'audio/webm;codecs=opus'
        });

        // Callback with recorded audio
        onRecordingComplete?.(audioBlob);

        // Cleanup
        audioChunks.current = [];
      };

      // Start recording
      mediaRecorder.current.start(100); // Collect data every 100ms
      setIsRecording(true);

    } catch (error) {
      console.error('Failed to start recording:', error);
      throw error;
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
      mediaRecorder.current.stop();

      // Stop all tracks
      stream.current?.getTracks().forEach(track => track.stop());

      setIsRecording(false);
    }
  }, []);

  return {
    isRecording,
    startRecording,
    stopRecording
  };
}
```

---

#### Option 3: WebSocket Streaming

**For real-time streaming to backend:**

```typescript
// hooks/useVoiceStream.ts
import { useState, useRef, useEffect } from 'react';

export function useVoiceStream(wsUrl: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioContext = useRef<AudioContext | null>(null);

  useEffect(() => {
    // Connect to WebSocket
    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'transcript':
          console.log('Transcript:', data.text, data.isFinal);
          break;
        case 'response':
          console.log('Agent response:', data.text);
          break;
        case 'audio':
          // Play audio response
          playAudioResponse(data.audio);
          break;
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };

    return () => {
      ws.current?.close();
    };
  }, [wsUrl]);

  async function startStreaming() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 16000,
        channelCount: 1
      }
    });

    // Create AudioContext for processing
    audioContext.current = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.current.createMediaStreamSource(stream);

    // Use AudioWorklet for low-latency processing
    await audioContext.current.audioWorklet.addModule('/audio-processor.js');
    const processor = new AudioWorkletNode(audioContext.current, 'audio-processor');

    processor.port.onmessage = (event) => {
      // Send audio chunks via WebSocket
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send(event.data);
      }
    };

    source.connect(processor);
    processor.connect(audioContext.current.destination);

    setIsStreaming(true);
  }

  function stopStreaming() {
    audioContext.current?.close();
    setIsStreaming(false);
  }

  async function playAudioResponse(base64Audio: string) {
    // Decode base64 audio
    const audioData = atob(base64Audio);
    const arrayBuffer = new ArrayBuffer(audioData.length);
    const uint8Array = new Uint8Array(arrayBuffer);

    for (let i = 0; i < audioData.length; i++) {
      uint8Array[i] = audioData.charCodeAt(i);
    }

    // Create audio context and play
    const ctx = new AudioContext();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    source.start();
  }

  return {
    isConnected,
    isStreaming,
    startStreaming,
    stopStreaming
  };
}
```

**Audio Processor (public/audio-processor.js):**
```javascript
class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];

    if (input.length > 0) {
      const samples = input[0];

      for (let i = 0; i < samples.length; i++) {
        this.buffer[this.bufferIndex++] = samples[i];

        if (this.bufferIndex >= this.bufferSize) {
          // Send buffer to main thread
          this.port.postMessage(this.buffer.slice());
          this.bufferIndex = 0;
        }
      }
    }

    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
```

---

### Voice Visualization

```typescript
// components/VoiceVisualizer.tsx
import { useEffect, useRef } from 'react';

interface VoiceVisualizerProps {
  stream: MediaStream | null;
  isActive: boolean;
}

export function VoiceVisualizer({ stream, isActive }: VoiceVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const analyserRef = useRef<AnalyserNode>();

  useEffect(() => {
    if (!stream || !isActive) return;

    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();

    analyser.fftSize = 256;
    source.connect(analyser);
    analyserRef.current = analyser;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      if (!canvasRef.current || !analyserRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      analyserRef.current.getByteFrequencyData(dataArray);

      ctx.fillStyle = 'rgb(20, 20, 20)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;

        ctx.fillStyle = `rgb(${barHeight + 100}, 50, 150)`;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
      }

      animationRef.current = requestAnimationFrame(draw);
    }

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      audioContext.close();
    };
  }, [stream, isActive]);

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={200}
      className="voice-visualizer"
    />
  );
}
```

---

## Backend Implementation

### Node.js/Express Backend

#### Option 1: REST API Endpoint

```typescript
// server/routes/voice.ts
import express from 'express';
import multer from 'multer';
import { OpenAI } from 'openai';

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });
const openai = new OpenAI();

router.post('/voice', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No audio file provided' });
    }

    // 1. Transcribe audio (STT)
    const transcript = await transcribeAudio(req.file.buffer);

    // 2. Generate response (LLM)
    const response = await generateResponse(transcript);

    // 3. Synthesize speech (TTS)
    const audioBuffer = await synthesizeSpeech(response);

    // 4. Return results
    res.json({
      transcript,
      response,
      audioUrl: `/audio/${Date.now()}.mp3` // Or return base64
    });

  } catch (error) {
    console.error('Voice processing error:', error);
    res.status(500).json({ error: 'Failed to process voice' });
  }
});

async function transcribeAudio(audioBuffer: Buffer): Promise<string> {
  // Using OpenAI Whisper API
  const file = new File([audioBuffer], 'audio.webm', { type: 'audio/webm' });

  const transcription = await openai.audio.transcriptions.create({
    file,
    model: 'whisper-1'
  });

  return transcription.text;
}

async function generateResponse(userInput: string): Promise<string> {
  const completion = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [
      { role: 'system', content: 'You are a helpful voice assistant.' },
      { role: 'user', content: userInput }
    ]
  });

  return completion.choices[0].message.content || '';
}

async function synthesizeSpeech(text: string): Promise<Buffer> {
  const mp3 = await openai.audio.speech.create({
    model: 'tts-1',
    voice: 'alloy',
    input: text
  });

  const buffer = Buffer.from(await mp3.arrayBuffer());
  return buffer;
}

export default router;
```

---

#### Option 2: WebSocket Server

```typescript
// server/websocket.ts
import WebSocket, { WebSocketServer } from 'ws';
import { createClient, LiveTranscriptionEvents } from '@deepgram/sdk';

interface VoiceSession {
  ws: WebSocket;
  deepgramConnection: any;
  conversationHistory: Array<{ role: string; content: string }>;
}

const sessions = new Map<string, VoiceSession>();

export function setupWebSocketServer(server: any) {
  const wss = new WebSocketServer({ server, path: '/voice' });

  wss.on('connection', async (ws: WebSocket) => {
    const sessionId = generateSessionId();
    console.log(`New voice session: ${sessionId}`);

    // Initialize Deepgram connection
    const deepgram = createClient(process.env.DEEPGRAM_API_KEY!);
    const deepgramConnection = deepgram.listen.live({
      model: 'nova-3',
      language: 'en-US',
      smart_format: true,
      interim_results: true,
      endpointing: 300
    });

    // Store session
    const session: VoiceSession = {
      ws,
      deepgramConnection,
      conversationHistory: []
    };
    sessions.set(sessionId, session);

    // Handle Deepgram events
    deepgramConnection.on(LiveTranscriptionEvents.Open, () => {
      console.log(`Deepgram connected for session ${sessionId}`);
    });

    deepgramConnection.on(LiveTranscriptionEvents.Transcript, async (data: any) => {
      const transcript = data.channel.alternatives[0].transcript;
      const isFinal = data.is_final;

      if (transcript && transcript !== '') {
        // Send transcript to client
        ws.send(JSON.stringify({
          type: 'transcript',
          text: transcript,
          isFinal
        }));

        // If final, generate response
        if (isFinal) {
          await handleUserInput(sessionId, transcript);
        }
      }
    });

    deepgramConnection.on(LiveTranscriptionEvents.Error, (error: any) => {
      console.error('Deepgram error:', error);
    });

    // Handle WebSocket messages from client
    ws.on('message', async (message: Buffer) => {
      // Audio data from client
      if (deepgramConnection) {
        deepgramConnection.send(message);
      }
    });

    ws.on('close', () => {
      console.log(`Session closed: ${sessionId}`);
      deepgramConnection?.finish();
      sessions.delete(sessionId);
    });
  });
}

async function handleUserInput(sessionId: string, userInput: string) {
  const session = sessions.get(sessionId);
  if (!session) return;

  // Add to conversation history
  session.conversationHistory.push({
    role: 'user',
    content: userInput
  });

  // Generate response with LLM
  const response = await generateLLMResponse(session.conversationHistory);

  session.conversationHistory.push({
    role: 'assistant',
    content: response
  });

  // Send text response
  session.ws.send(JSON.stringify({
    type: 'response',
    text: response
  }));

  // Generate and send audio
  const audioBuffer = await synthesizeSpeech(response);
  session.ws.send(JSON.stringify({
    type: 'audio',
    audio: audioBuffer.toString('base64')
  }));
}

async function generateLLMResponse(history: Array<{ role: string; content: string }>): Promise<string> {
  // Your LLM logic here (OpenAI, Anthropic, local model, LangGraph, etc.)
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'gpt-4',
      messages: history
    })
  });

  const data = await response.json();
  return data.choices[0].message.content;
}

function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
```

---

### Python/FastAPI Backend

```python
# server/main.py
from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import JSONResponse
import asyncio
from typing import List, Dict
import openai
from deepgram import Deepgram
import os

app = FastAPI()

# Initialize services
openai.api_key = os.getenv("OPENAI_API_KEY")
deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY"))

# Store active sessions
sessions: Dict[str, dict] = {}

@app.post("/api/voice")
async def process_voice(audio: UploadFile = File(...)):
    """REST endpoint for voice processing"""

    # Read audio file
    audio_bytes = await audio.read()

    # 1. Transcribe
    transcript = await transcribe_audio(audio_bytes)

    # 2. Generate response
    response = await generate_response(transcript)

    # 3. Synthesize speech
    audio_response = await synthesize_speech(response)

    return JSONResponse({
        "transcript": transcript,
        "response": response,
        "audio": audio_response  # base64 encoded
    })

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming voice"""

    await websocket.accept()
    session_id = generate_session_id()

    # Initialize session
    session = {
        "websocket": websocket,
        "conversation_history": [],
        "deepgram_connection": None
    }
    sessions[session_id] = session

    # Create Deepgram connection
    dg_connection = deepgram.transcription.live({
        "model": "nova-3",
        "language": "en-US",
        "smart_format": True,
        "interim_results": True,
        "endpointing": 300
    })

    session["deepgram_connection"] = dg_connection

    # Handle Deepgram transcripts
    async def handle_transcript(transcript_data):
        transcript = transcript_data["channel"]["alternatives"][0]["transcript"]
        is_final = transcript_data["is_final"]

        if transcript:
            await websocket.send_json({
                "type": "transcript",
                "text": transcript,
                "isFinal": is_final
            })

            if is_final:
                # Generate response
                await handle_user_input(session_id, transcript)

    dg_connection.register_handler(
        deepgram.transcription.live_handler.TranscriptReceivedHandler(handle_transcript)
    )

    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_bytes()

            # Send to Deepgram
            dg_connection.send(data)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        dg_connection.finish()
        del sessions[session_id]

async def handle_user_input(session_id: str, user_input: str):
    """Handle user input and generate response"""

    session = sessions.get(session_id)
    if not session:
        return

    # Add to history
    session["conversation_history"].append({
        "role": "user",
        "content": user_input
    })

    # Generate response
    response = await generate_response_from_history(session["conversation_history"])

    session["conversation_history"].append({
        "role": "assistant",
        "content": response
    })

    # Send text response
    await session["websocket"].send_json({
        "type": "response",
        "text": response
    })

    # Generate and send audio
    audio_data = await synthesize_speech(response)
    await session["websocket"].send_json({
        "type": "audio",
        "audio": audio_data
    })

async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using Deepgram or Whisper"""

    # Using Deepgram
    response = await deepgram.transcription.prerecorded({
        "buffer": audio_bytes,
        "mimetype": "audio/webm"
    }, {
        "model": "nova-3",
        "smart_format": True
    })

    return response["results"]["channels"][0]["alternatives"][0]["transcript"]

async def generate_response(user_input: str) -> str:
    """Generate response using LLM"""

    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "user", "content": user_input}
        ]
    )

    return response.choices[0].message.content

async def generate_response_from_history(history: List[Dict]) -> str:
    """Generate response from conversation history"""

    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=history
    )

    return response.choices[0].message.content

async def synthesize_speech(text: str) -> str:
    """Synthesize speech using OpenAI TTS"""

    response = await openai.Audio.create_speech(
        model="tts-1",
        voice="alloy",
        input=text
    )

    # Return base64 encoded audio
    import base64
    return base64.b64encode(response.content).decode()

def generate_session_id() -> str:
    """Generate unique session ID"""
    import uuid
    return str(uuid.uuid4())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## WebSocket vs REST

### Comparison

| Aspect | WebSocket | REST |
|--------|-----------|------|
| **Latency** | 150-300ms first token | 500-2000ms response |
| **Streaming** | Native bidirectional | Not supported |
| **Use Case** | Real-time voice agents | Batch processing |
| **Complexity** | Higher (connection mgmt) | Lower (stateless) |
| **Scalability** | Requires sticky sessions | Easy horizontal scaling |
| **Industry Standard** | OpenAI Realtime, Deepgram | Whisper API |

### Recommendation

**Use WebSocket for:**
- Real-time voice agents
- Sub-500ms latency required
- Bidirectional streaming
- Interruption handling

**Use REST for:**
- Batch transcription
- Simple voice commands
- Async processing
- When simplicity > latency

---

## Voice Activity Detection

### Client-Side VAD (Browser)

```typescript
// utils/vad.ts
import { MicVAD } from '@ricky0123/vad-web';

export async function initializeVAD(callbacks: {
  onSpeechStart: () => void;
  onSpeechEnd: (audio: Float32Array) => void;
  onVADMisfire: () => void;
}) {
  const vad = await MicVAD.new({
    // Sensitivity settings
    positiveSpeechThreshold: 0.8,  // 0-1, higher = less sensitive
    negativeSpeechThreshold: 0.3,  // 0-1, lower = faster end detection

    // Frame settings
    minSpeechFrames: 3,      // Minimum consecutive speech frames
    preSpeechPadFrames: 10,  // Frames to include before speech
    redemptionFrames: 8,     // Frames to wait before ending

    // Callbacks
    onSpeechStart: callbacks.onSpeechStart,
    onSpeechEnd: callbacks.onSpeechEnd,
    onVADMisfire: callbacks.onVADMisfire,

    // Audio settings
    workletURL: '/vad.worklet.bundle.min.js',
    modelURL: '/silero_vad.onnx',
    ortConfig(ort) {
      ort.env.wasm.wasmPaths = '/';
    }
  });

  return vad;
}

// Usage
const vad = await initializeVAD({
  onSpeechStart: () => {
    console.log("User started speaking");
    setIsUserSpeaking(true);
  },

  onSpeechEnd: async (audio) => {
    console.log("User stopped speaking");
    setIsUserSpeaking(false);

    // Process audio
    await processAudio(audio);
  },

  onVADMisfire: () => {
    console.log("False positive - no actual speech");
  }
});

// Start VAD
await vad.start();

// Pause/resume
vad.pause();
vad.start();
```

---

### Server-Side VAD (Python)

```python
# server/vad.py
import torch
import numpy as np
from collections import deque

class VoiceActivityDetector:
    def __init__(self, model_path='snakers4/silero-vad', threshold=0.5):
        # Load Silero VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir=model_path,
            model='silero_vad',
            force_reload=False
        )

        self.get_speech_timestamps = utils[0]
        self.threshold = threshold
        self.sample_rate = 16000

    def detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech

        Args:
            audio_chunk: numpy array of audio samples (float32, 16kHz)

        Returns:
            bool: True if speech detected
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk)

        # Get speech probability
        speech_prob = self.model(audio_tensor, self.sample_rate).item()

        return speech_prob > self.threshold

    def get_speech_segments(self, audio: np.ndarray):
        """
        Get timestamps of speech segments

        Args:
            audio: numpy array of full audio

        Returns:
            list of dicts with 'start' and 'end' timestamps
        """
        audio_tensor = torch.from_numpy(audio)

        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )

        return speech_timestamps

# Usage
vad = VoiceActivityDetector(threshold=0.5)

# Real-time detection
audio_chunk = np.array([...])  # Float32 audio at 16kHz
has_speech = vad.detect_speech(audio_chunk)

if has_speech:
    print("Speech detected!")

# Batch processing
full_audio = np.array([...])
segments = vad.get_speech_segments(full_audio)

for segment in segments:
    start_ms = segment['start']
    end_ms = segment['end']
    print(f"Speech from {start_ms}ms to {end_ms}ms")
```

---

## Interruption Handling

### Pattern 1: Client-Side Interruption

```typescript
// components/VoiceAgent.tsx
import { useState, useRef } from 'react';

export function VoiceAgent() {
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);
  const [currentAudioElement, setCurrentAudioElement] = useState<HTMLAudioElement | null>(null);
  const abortController = useRef<AbortController | null>(null);

  async function handleUserSpeech(audio: Float32Array) {
    // If agent is currently speaking, interrupt it
    if (isAgentSpeaking) {
      interruptAgent();
    }

    // Process user speech
    await processUserInput(audio);
  }

  function interruptAgent() {
    console.log("Interrupting agent");

    // 1. Stop audio playback
    if (currentAudioElement) {
      currentAudioElement.pause();
      currentAudioElement.currentTime = 0;
    }

    // 2. Cancel in-flight API requests
    if (abortController.current) {
      abortController.current.abort();
    }

    // 3. Clear any pending audio
    setIsAgentSpeaking(false);

    // 4. Notify server of interruption
    fetch('/api/interrupt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: 'user_speech' })
    });
  }

  async function playAgentResponse(audioUrl: string) {
    const audio = new Audio(audioUrl);
    setCurrentAudioElement(audio);
    setIsAgentSpeaking(true);

    audio.onended = () => {
      setIsAgentSpeaking(false);
      setCurrentAudioElement(null);
    };

    try {
      await audio.play();
    } catch (error) {
      console.error("Failed to play audio:", error);
      setIsAgentSpeaking(false);
    }
  }

  async function processUserInput(audio: Float32Array) {
    // Create new abort controller for this request
    abortController.current = new AbortController();

    try {
      const response = await fetch('/api/voice', {
        method: 'POST',
        body: audioToBlob(audio),
        signal: abortController.current.signal
      });

      if (!response.ok) throw new Error('Request failed');

      const data = await response.json();

      // Play agent response
      await playAgentResponse(data.audioUrl);

    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request cancelled due to interruption');
      } else {
        console.error('Error processing voice:', error);
      }
    }
  }

  return (
    <div>
      {/* Voice interface here */}
    </div>
  );
}
```

---

### Pattern 2: Server-Side Interruption Handling

```python
# server/interruption.py
from typing import Dict, Optional
import asyncio

class InterruptionManager:
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.conversation_state: Dict[str, dict] = {}

    async def handle_user_input(
        self,
        session_id: str,
        user_input: str
    ) -> Optional[str]:
        """Handle user input with interruption support"""

        # Cancel any active response generation for this session
        await self.cancel_active_response(session_id)

        # Create new task for this response
        task = asyncio.create_task(
            self.generate_response(session_id, user_input)
        )

        self.active_tasks[session_id] = task

        try:
            response = await task
            return response
        except asyncio.CancelledError:
            print(f"Response generation cancelled for session {session_id}")
            return None
        finally:
            if session_id in self.active_tasks:
                del self.active_tasks[session_id]

    async def cancel_active_response(self, session_id: str):
        """Cancel any active response generation"""

        if session_id in self.active_tasks:
            task = self.active_tasks[session_id]

            if not task.done():
                print(f"Cancelling active task for session {session_id}")
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def generate_response(
        self,
        session_id: str,
        user_input: str
    ) -> str:
        """Generate response (can be cancelled)"""

        # Get conversation history
        history = self.conversation_state.get(session_id, {}).get('history', [])
        history.append({"role": "user", "content": user_input})

        # Generate response (this can be interrupted)
        response = await self.call_llm(history)

        # Update conversation history if not cancelled
        history.append({"role": "assistant", "content": response})

        if session_id not in self.conversation_state:
            self.conversation_state[session_id] = {}
        self.conversation_state[session_id]['history'] = history

        return response

    async def call_llm(self, history: list) -> str:
        """Call LLM (placeholder - implement with your LLM)"""

        # Simulate LLM call that can be cancelled
        await asyncio.sleep(2)  # Simulated API call

        return "This is a response from the LLM"

    def handle_false_interruption(self, session_id: str):
        """Handle case where interruption was a false positive"""

        # Resume previous response if available
        state = self.conversation_state.get(session_id, {})

        if 'interrupted_response' in state:
            print(f"Resuming interrupted response for session {session_id}")
            return state['interrupted_response']

        return None

# Usage in FastAPI
manager = InterruptionManager()

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    session_id = generate_session_id()

    try:
        while True:
            data = await websocket.receive_json()

            if data['type'] == 'user_speech':
                # This will automatically cancel any in-flight responses
                response = await manager.handle_user_input(
                    session_id,
                    data['transcript']
                )

                if response:
                    await websocket.send_json({
                        'type': 'response',
                        'text': response
                    })

            elif data['type'] == 'interrupt':
                await manager.cancel_active_response(session_id)

    except WebSocketDisconnect:
        await manager.cancel_active_response(session_id)
```

---

## State Management

### Conversation State with Redis

```python
# server/state.py
import redis
import json
from typing import List, Dict, Optional
from datetime import timedelta

class ConversationStateManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = timedelta(hours=24)

    def get_conversation(self, session_id: str) -> List[Dict]:
        """Get conversation history"""

        key = f"conversation:{session_id}"
        data = self.redis_client.get(key)

        if data:
            return json.loads(data)
        return []

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """Add message to conversation"""

        history = self.get_conversation(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        history.append(message)

        # Save back to Redis
        key = f"conversation:{session_id}"
        self.redis_client.setex(
            key,
            self.default_ttl,
            json.dumps(history)
        )

    def get_context_window(
        self,
        session_id: str,
        max_messages: int = 20
    ) -> List[Dict]:
        """Get recent messages for LLM context"""

        history = self.get_conversation(session_id)

        # Return last N messages
        return history[-max_messages:]

    def clear_conversation(self, session_id: str):
        """Clear conversation history"""

        key = f"conversation:{session_id}"
        self.redis_client.delete(key)

    def set_session_metadata(
        self,
        session_id: str,
        metadata: Dict
    ):
        """Store session metadata"""

        key = f"session_meta:{session_id}"
        self.redis_client.setex(
            key,
            self.default_ttl,
            json.dumps(metadata)
        )

    def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """Get session metadata"""

        key = f"session_meta:{session_id}"
        data = self.redis_client.get(key)

        if data:
            return json.loads(data)
        return None

# Usage
state_manager = ConversationStateManager()

# Add user message
state_manager.add_message(
    session_id="abc123",
    role="user",
    content="Hello!",
    metadata={"audio_duration": 1.5}
)

# Add agent response
state_manager.add_message(
    session_id="abc123",
    role="assistant",
    content="Hi! How can I help you?",
    metadata={"latency_ms": 450}
)

# Get context for LLM
context = state_manager.get_context_window("abc123", max_messages=10)

# Get full history
full_history = state_manager.get_conversation("abc123")
```

---

## Complete Examples

### Complete Next.js + FastAPI Voice Agent

**Frontend (Next.js):**
```typescript
// app/page.tsx
'use client';

import { useMicVAD } from '@ricky0123/vad-react';
import { useState } from 'react';

export default function Home() {
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [isAgentSpeaking, setIsAgentSpeaking] = useState(false);

  const vad = useMicVAD({
    startOnLoad: false,

    onSpeechEnd: async (audio) => {
      // Convert to WAV
      const wavBlob = await audioToWav(audio);

      // Send to backend
      const formData = new FormData();
      formData.append('audio', wavBlob, 'audio.wav');

      const response = await fetch('http://localhost:8000/api/voice', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      // Update messages
      setMessages(prev => [
        ...prev,
        { role: 'user', content: data.transcript },
        { role: 'assistant', content: data.response }
      ]);

      // Play audio
      const audioBlob = base64ToBlob(data.audio, 'audio/mp3');
      const audioUrl = URL.createObjectURL(audioBlob);
      await playAudio(audioUrl);
    }
  });

  async function audioToWav(float32Audio: Float32Array): Promise<Blob> {
    // Implementation of Float32Array to WAV conversion
    // ... (see previous examples)
  }

  async function playAudio(url: string) {
    setIsAgentSpeaking(true);
    const audio = new Audio(url);
    audio.onended = () => setIsAgentSpeaking(false);
    await audio.play();
  }

  return (
    <div className="container">
      <h1>Voice Agent</h1>

      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <strong>{msg.role}:</strong> {msg.content}
          </div>
        ))}
      </div>

      <button
        onClick={() => vad.listening ? vad.pause() : vad.start()}
        disabled={vad.loading || isAgentSpeaking}
      >
        {vad.loading ? 'Loading...' : vad.listening ? 'Stop' : 'Start'} Voice
      </button>

      {vad.userSpeaking && <div>🔴 Listening...</div>}
      {isAgentSpeaking && <div>🤖 Agent speaking...</div>}
    </div>
  );
}
```

**Backend (FastAPI):**
```python
# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import openai
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
whisper_model = WhisperModel("medium", device="cuda")

@app.post("/api/voice")
async def process_voice(audio: UploadFile = File(...)):
    # 1. Transcribe
    audio_bytes = await audio.read()

    # Save temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)

    segments, _ = whisper_model.transcribe("temp_audio.wav")
    transcript = " ".join([seg.text for seg in segments])

    # 2. Generate response
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "user", "content": transcript}
        ]
    )

    response_text = response.choices[0].message.content

    # 3. Synthesize speech
    audio_response = await openai.Audio.create_speech(
        model="tts-1",
        voice="alloy",
        input=response_text
    )

    # Encode to base64
    audio_base64 = base64.b64encode(audio_response.content).decode()

    return {
        "transcript": transcript,
        "response": response_text,
        "audio": audio_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

**This implementation guide covers all the core patterns needed to build production voice agents!**

---

**Last Updated:** November 21, 2025
