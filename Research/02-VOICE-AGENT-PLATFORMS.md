# Voice Agent Platforms - Detailed Analysis

**Research Date:** November 2025

---

## Table of Contents

1. [Commercial Platforms](#commercial-platforms)
2. [Open Source Frameworks](#open-source-frameworks)
3. [LangGraph Integration](#langgraph-integration)
4. [Comparison Matrix](#comparison-matrix)
5. [Integration Examples](#integration-examples)

---

## Commercial Platforms

### 1. Retell AI

**Overview:** Complete managed audio pipeline with custom LLM integration

**Architecture:**
- Integrated ASR, LLM, and TTS processing
- Optimized network architecture
- WebSocket-based for custom LLM
- ~800ms end-to-end latency

**Conversation Flow:**
- Dedicated turn-taking model combining:
  - Transcript analysis
  - Emotion detection
  - Tonality analysis
  - Pause detection
- Replicates human conversation patterns
- Real-time streaming across all components

**Agent Framework Support:**
- ❌ NO native LangGraph support
- ✅ **Custom LLM via WebSocket protocol**
- Your server hosts WebSocket at `{endpoint}/{call_id}`
- Retell sends transcript updates and turn signals
- Your server responds with agent speech

**WebSocket Message Types:**
```
Retell → Server:
- ping/pong (heartbeat)
- call_details (start of call)
- transcript (user speech)
- response_required (turn signal)

Server → Retell:
- config (initial setup)
- response (agent speech)
- agent_interrupt (cancel current speech)
- update_agent (modify behavior mid-call)
- tool_calls (function calling)
```

**Interruption Handling:**
- **Blazingly fast barge-in** detection (<200ms target)
- User can interrupt at any time
- Interruption sensitivity slider (configurable)
- Core platform strength

**Streaming Capabilities:**
- Real-time streaming across ASR, LLM, TTS
- Continuous response streaming
- Minimal latency optimized

**Integration:**
- Direct Twilio integration for telephony
- Open-sourced web frontend code
- Function calling support
- Official demos:
  - Python: github.com/RetellAI/retell-custom-llm-python-demo
  - Node.js: github.com/RetellAI/retell-custom-llm-node-demo

**Pricing (2025):**
- Base: $0.07+ per minute (pay-as-you-go)
- Component breakdown:
  - Voice: $0.07-0.08/min
  - LLM: $0.006-0.06/min (depends on model)
  - Telephony: $0.015/min (Retell Twilio) or $0 (BYO)
  - Knowledge base: $0.005/min
  - Phone numbers: $2/month
- Enterprise: $3k+/month ($0.05/min with volume discounts)
- Example total: ~$0.14/min (ElevenLabs + Claude 3.5 + Twilio)

**Compliance:**
- HIPAA compliant
- SOC 2 Type 1 & 2
- GDPR compliant
- Available across all plans

**Setup Example:**
```bash
# 1. Install dependencies
npm install retell-sdk

# 2. Start ngrok tunnel
ngrok http 8080

# 3. Run WebSocket server
node server.js

# 4. Configure Retell dashboard
# WebSocket URL: wss://your-ngrok-url.ngrok.io/llm-websocket

# 5. Make test call
```

**Code Example - Node.js WebSocket Server:**
```javascript
const express = require('express');
const expressWs = require('express-ws');
const { Retell } = require('retell-sdk');

const app = express();
expressWs(app);

// WebSocket endpoint for Retell
app.ws('/llm-websocket/:call_id', (ws, req) => {
  const callId = req.params.call_id;
  console.log(`Call connected: ${callId}`);

  ws.on('message', async (message) => {
    const data = JSON.parse(message);

    switch (data.type) {
      case 'call_details':
        // Call started
        console.log('Call started:', data.call);
        break;

      case 'transcript':
        // User spoke
        console.log('User said:', data.transcript);
        break;

      case 'response_required':
        // Time for agent to respond
        const response = await generateResponse(data.transcript);

        ws.send(JSON.stringify({
          type: 'response',
          response_id: data.response_id,
          content: response,
          content_complete: true
        }));
        break;

      case 'ping':
        // Heartbeat
        ws.send(JSON.stringify({ type: 'pong' }));
        break;
    }
  });

  ws.on('close', () => {
    console.log(`Call ended: ${callId}`);
  });
});

async function generateResponse(transcript) {
  // Your LLM/agent logic here (LangGraph, etc.)
  // Return string or stream responses
  return "This is where your agent responds";
}

app.listen(8080, () => {
  console.log('Server listening on port 8080');
});
```

**Best For:**
- Production voice agents
- Telephony integration required
- Fast barge-in critical
- Want managed audio pipeline
- Need LangGraph or custom agent logic

---

### 2. Vapi.ai

**Overview:** Orchestration layer over real-time STT, LLM, and TTS modules

**Architecture:**
```
User Audio → VAD → Transcription → Start Speaking Decision → LLM → TTS → waitSeconds → Assistant Audio
```

**Conversation Flow:**

**Start Speaking Process:**
1. VAD detects user stops speaking
2. System evaluates utterance completion:
   - Transcriber end-of-turn signal
   - Custom rules
   - Smart endpointing
3. LLM generates response while TTS processes
4. Final delay (waitSeconds) before speaking

**Stop Speaking Process (Interruptions):**
1. Check for special phrases (triggers/acknowledgements)
2. Evaluate if input meets threshold
3. Clear pipeline if true interruption

**Agent Framework Support:**
- ❌ NO native LangGraph support
- ✅ **Custom LLM integration** - OpenAI-compatible endpoint
- ✅ **Community-confirmed LangGraph works**:
  - Configure Vapi to use LangGraph endpoint
  - Requires OpenAI structured response format
  - Must be streaming response
  - Send full conversation history each request
  - Implement queuing/cancellation for overlaps
- ✅ Recent MCP (Model Context Protocol) integration

**Interruption Handling:**

**Two Detection Methods:**
- **VAD-based** (numWords = 0):
  - 50-100ms response time
  - Language-independent
  - More noise-sensitive

- **Transcription-based** (numWords > 0):
  - Waits for specified words
  - 200-500ms delay
  - More accurate

**Configuration Parameters:**
```javascript
{
  backoffSeconds: 0.5,    // Recovery period after interruption (0-10s)
  voiceSeconds: 0.2,      // VAD duration threshold (0-0.5s)
  waitSeconds: 0.1        // Final delay before speaking (0-5s)
}
```

**Streaming Capabilities:**
- Real-time streaming across all stages
- 50-100ms VAD sensitivity
- Target latency: 500-700ms voice-to-voice
- Generated replies via Server-Sent Events (SSE)
- Sub-600ms response times

**Integration:**
- Custom LLM endpoint (OpenAI-compatible)
- Tool calling: transferCall, endCall, sms, dtmf, apiRequest
- Authentication: API Key or OAuth2
- Official examples: github.com/VapiAI/example-custom-llm
- Multiple deployment examples (Vercel, Supabase, Bun, Deno)

**Pricing (2025):**
- Platform fee: $0.05/min (advertised)
- **True total cost: $0.13-0.31/min** with third-party services:
  - LLM: ~$0.06-0.07/min
  - TTS/STT providers
  - Telephony (Twilio, etc.)
- Free trial: $10 credits (~30-75 minutes)
- Add-ons:
  - HIPAA/SOC 2 Compliance: $1,000/month
  - Extra SIP lines: $10/line/month
- Enterprise: $3,000-6,000/month typical
- Annual commitments: $40k-70k

**Infrastructure:**
- Kubernetes-based
- Scales to millions of concurrent calls
- Multi-region deployment

**Code Example - Custom LLM (Python/Flask):**
```python
from flask import Flask, request, jsonify, Response
import openai
import json

app = Flask(__name__)

@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()

    # Extract conversation from Vapi format
    messages = data.get("messages", [])

    # Your LangGraph agent logic here
    # Must return OpenAI-compatible streaming format

    def generate():
        # Call your agent
        for chunk in your_langgraph_agent(messages):
            # Format as OpenAI streaming response
            response = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(response)}\n\n"

        # Send final chunk
        final = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=8080)
```

**Best For:**
- Enterprise scale required
- High customization needs
- Want provider flexibility (choose your own STT/TTS)
- Need detailed configuration control
- LangGraph integration (with custom LLM endpoint)

---

### 3. Bland AI

**Overview:** Visual workflow system with conversational pathways

**Architecture:**
- **Nodes:** Individual conversation points
- **Pathways:** Connections with conditional routing
- **Conditions:** Requirements for progression
- Low-code/no-code visual interface

**Conversation Flow:**
- Agent starts at first node
- Traverses based on decision logic
- **Node Types:**
  - Default/Base: Generates responses via prompts
  - Webhook: Executes external integrations
  - Knowledge Base: Accesses information
  - Transfer Call: Routes to humans
  - End Call: Terminates with closing
  - Wait for Response: Handles user delays
- **Global Nodes:** Handle unexpected questions
- **Variable Extraction:** Captures user information
- **Fine-tuning:** Live call log corrections

**Agent Framework Support:**
- ❌ NO LangGraph support
- ❌ NO native agent framework integration
- Primarily visual pathway-based
- API-driven with webhooks for custom logic

**Interruption Handling:**
- Interruption thresholds adjustable per node
- Can reduce interruptions at specific points
- Conditional routing based on responses

**Streaming Capabilities:**
- Real-time data fetching via webhooks
- Live call logs displayed real-time
- Agent can speak while webhook processing

**Integration:**
- Official SDKs: TypeScript/JavaScript, Python
- API-driven architecture
- Webhook integrations

**Pricing (2025):**
- Per-minute: $0.09 per connected minute (billed by second)
- Minimum charge: $0.015 per dispatched call (effective June 2025)
- Monthly plans:
  - Build: $299/mo
  - Scale: $499/mo
  - (Still pay per-minute on top of plan)
- Transfer rate: $0.025/min for merged calls
- SMS, TTS, integrations: Billed separately
- Multi-regional: Data doesn't cross borders

**Code Example - TypeScript:**
```typescript
import { BlandAIClient, BlandAI } from 'bland-ai';

const bland = new BlandAIClient({
  apiKey: process.env.BLAND_API_KEY
});

// Make a call
await bland.call({
  phoneNumber: "+1234567890",
  task: "Ask if they're interested in our AI voice agents and schedule a demo",
  temperature: 0.5,
  model: BlandAI.ModelEnum.Enhanced,
  voice: "nat",
  webhook: "https://your-server.com/webhook",
  pathway_id: "your-pathway-id"
});

// Handle webhook
app.post('/webhook', (req, res) => {
  const { call_id, status, transcript, variables } = req.body;

  // Process call results
  console.log('Call completed:', call_id);
  console.log('Extracted variables:', variables);

  res.json({ success: true });
});
```

**Best For:**
- Non-technical users
- No-code/low-code preference
- Simple conversational flows
- Telephony focus
- Don't need programmatic agent frameworks

---

### 4. Vocode (Open Source)

**Overview:** Open-source Python library with modular architecture

**Architecture:**
Three core components:
- **Transcriber:** Speech-to-text (Deepgram, AssemblyAI, Google, Azure, Whisper)
- **Agent:** Conversation logic (ChatGPT-based)
- **Synthesizer:** Text-to-speech (ElevenLabs, Azure, Google, Rime.ai)

**Conversation Flow:**
- **EndpointingConfig** determines when speakers finish
- Time-based (silence duration) or punctuation-based
- **Deepgram-based endpointing:**
  - VAD threshold: 500ms default
  - Utterance cutoff: 1000ms default
  - Time-silent config for marking final
- **conversation_speed** parameter adjusts latency dynamically

**Agent Framework Support:**
- ✅ **Native LangChain support** via `LangchainAgent` class
- ❌ **NO direct LangGraph support**
- ⚠️ **Could potentially wrap LangGraph** in custom agent
- Custom agent creation via `RespondAgent` subclassing:
  - `respond()`: Turn-based conversations
  - `generate_response()`: Async streaming

**Code Example - Custom Agent:**
```python
from vocode.streaming.agent.base_agent import RespondAgent

class CustomAgent(RespondAgent):
    def respond(self, human_input, is_interrupt: bool = False):
        # Your agent logic here
        # Could wrap LangGraph
        response = process_with_langgraph(human_input)
        return response

    async def generate_response(self, human_input, is_interrupt: bool = False):
        # Streaming version
        for chunk in stream_from_langgraph(human_input):
            yield chunk, False  # (content, is_final)
        yield "", True  # Signal completion
```

**Interruption Handling:**
- **interrupt_sensitivity** parameter in AgentConfig:
  - Low (default): Filters backchannels ("sure", "uh-huh")
  - High: Treats any speech as interruption
- Tracks with `current_transcription_is_interrupt` flag
- Implementation in `StreamingConversation.TranscriptionsWorker`

**Streaming Capabilities:**
- Real-time streaming with async/await
- Non-blocking conversation loop
- Automatic inter-component communication
- Token-by-token streaming

**Integration:**
- GitHub: github.com/vocodedev/vocode-core
- Documentation: docs.vocode.dev
- Telephony Server: FastAPI-based with Twilio
- Self-hosted or hosted deployment

**Pricing:**
- **Open-source: 100% free**
- Hosted Developer Plan: $25/month (API access, analytics, multilingual)
- Enterprise: Custom pricing
- Additional costs: Pay for underlying APIs (OpenAI, Deepgram, etc.)
- Deployment: Self-hosted (Docker, Redis) or hosted service

**Code Example - Complete Setup:**
```python
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.models.agent import ChatGPTAgentConfig

async def main():
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_telephone_input_device()
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                prompt_preamble="You are a helpful AI assistant",
                interrupt_sensitivity="high"
            )
        ),
        synthesizer=ElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_telephone_output_device()
        )
    )

    await conversation.start()
    print("Conversation started")

    # Wait for conversation to end
    while conversation.is_active():
        await asyncio.sleep(1)
```

**Best For:**
- Complete control and customization
- Self-hosting preference
- Open-source requirement
- Telephony integration (Twilio)
- Want to own your infrastructure
- LangChain integration (not LangGraph)

---

## Open Source Frameworks

### 1. Pipecat

**Overview:** Simplest open-source voice agent framework

**Setup Time:** 5-10 minutes

**Ease of Use:** Easiest among all frameworks

**Architecture:**
- Modular pipeline architecture
- Bring your own STT, LLM, TTS
- WebRTC or WebSocket transport
- Frame-based processing

**Agent Framework Support:**
- ✅ **Bring your own agent** (including LangGraph)
- ✅ Framework-agnostic
- Full control over agent logic
- Easy integration

**Code Example:**
```python
from pipecat.pipeline import Pipeline
from pipecat.transports.websocket import WebsocketTransport
from pipecat.services.deepgram import DeepgramSTT
from pipecat.services.openai import OpenAILLM
from pipecat.services.elevenlabs import ElevenLabsTTS

async def main():
    # Create pipeline
    pipeline = Pipeline([
        WebsocketTransport(),
        DeepgramSTT(api_key=os.getenv("DEEPGRAM_API_KEY")),
        OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")),
        # Or use your LangGraph agent here
        ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY"))
    ])

    await pipeline.run()
```

**Pricing:**
- 100% free (open source)
- Pay for underlying APIs only
- No platform fees

**Best For:**
- Fastest setup
- Want simplest implementation
- Bring your own components
- LangGraph integration easy
- Production-ready

**GitHub:** github.com/pipecat-ai/pipecat
**Documentation:** docs.pipecat.ai

---

### 2. Rasa

**Overview:** Mature conversational AI framework

**Downloads:** 50+ million

**Best For:** Complex dialogue management

**Agent Framework Support:**
- ✅ Own dialogue management system
- ⚠️ Steeper learning curve
- Custom actions and forms

**Pricing:**
- Open source: Free
- Enterprise features: $35K+ per year
- Some features require paid plans

**Best For:**
- Complex conversation flows
- Mature ecosystem needed
- Enterprise support required
- Not specifically voice-focused (but supports it)

---

## LangGraph Integration

### Summary: NO Native LangGraph Support

**None of the four major commercial platforms have native LangGraph integration.**

### Possible Integration Approaches

**1. Retell AI ✅ Most Feasible**
- Custom LLM WebSocket protocol
- Build WebSocket server hosting LangGraph agent
- Retell sends transcripts → LangGraph processes → return responses
- Full control over agent logic
- **Recommended approach**

**2. Vapi ✅ Community-Confirmed**
- Custom LLM integration works
- LangGraph endpoint as OpenAI-compatible API
- Must handle streaming and conversation history
- Requires careful state synchronization
- **Second-best approach**

**3. Bland AI ❌ Least Suitable**
- Visual pathway system, not programmatic
- No clear integration path
- **Not recommended for LangGraph**

**4. Vocode ⚠️ Possible with Work**
- LangChain support only (not LangGraph)
- Could wrap LangGraph in custom RespondAgent
- Would need custom adapter layer
- **Possible but not ideal**

### Community Independent Implementations

Developers have created custom stacks:
- **LiveKit + LangGraph:** github.com/ahmad2b/langgraph-voice-call-agent
- **Whisper + Groq + LangGraph + ElevenLabs:** Custom pipelines
- These bypass managed platforms for direct control

---

## Comparison Matrix

### Platform Comparison

| Platform | LangGraph Support | Setup Difficulty | Latency | Cost/min | Open Source |
|----------|-------------------|------------------|---------|----------|-------------|
| **Retell AI** | ✅ Custom LLM | Easy | ~800ms | $0.14 | ❌ No |
| **Vapi.ai** | ✅ Custom LLM | Medium | 500-700ms | $0.13-0.31 | ❌ No |
| **Bland AI** | ❌ No | Easy (visual) | Not specified | $0.09 | ❌ No |
| **Vocode** | ⚠️ LangChain | Medium | Configurable | APIs only | ✅ Yes |
| **Pipecat** | ✅ Native | Easy | <500ms | APIs only | ✅ Yes |
| **Rasa** | ❌ Own system | Hard | Varies | Free/Enterprise | ✅ Yes |

### Feature Comparison

| Platform | Interruptions | Telephony | Streaming | Custom LLM | Best Feature |
|----------|---------------|-----------|-----------|------------|--------------|
| **Retell AI** | Excellent (<200ms) | ✅ Built-in | ✅ Yes | ✅ WebSocket | Interruption handling |
| **Vapi.ai** | Good (50-500ms) | ✅ Built-in | ✅ Yes | ✅ HTTP | Scalability |
| **Bland AI** | Configurable | ✅ Built-in | ✅ Yes | ⚠️ Limited | Visual builder |
| **Vocode** | Good | ✅ Twilio | ✅ Yes | ✅ Full | Open source |
| **Pipecat** | Customizable | ⚠️ BYO | ✅ Yes | ✅ Full | Simplicity |
| **Rasa** | Customizable | ⚠️ BYO | ⚠️ Limited | ✅ Full | Dialogue mgmt |

### Cost Comparison (per 1000 minutes)

| Platform | Base Cost | STT | LLM | TTS | Total |
|----------|-----------|-----|-----|-----|-------|
| **Retell AI** | Included | Included | $360 | Included | **$8,400** |
| **Vapi.ai** | $3,000 | Provider | $360 | Provider | **$7,800-18,600** |
| **Bland AI** | $5,400 | Included | Custom | Included | **$5,400+** |
| **Vocode** | $25/mo | $276 | $360 | $180 | **$841/mo** |
| **Pipecat** | $0 | $276 | $360 | $180 | **$816/mo** |

*Note: Costs vary significantly based on usage patterns and provider choices*

---

## Integration Examples

### Example 1: LangGraph + Retell AI

```javascript
// server.js - WebSocket server for Retell + LangGraph
const express = require('express');
const expressWs = require('express-ws');
const { CompiledGraph } = require('@langchain/langgraph');

const app = express();
expressWs(app);

// Load your LangGraph
const graph = loadYourLangGraph();

app.ws('/llm-websocket/:call_id', (ws, req) => {
  const callId = req.params.call_id;
  let conversationHistory = [];

  ws.on('message', async (message) => {
    const data = JSON.parse(message);

    if (data.type === 'response_required') {
      // Get user's latest input
      const userInput = data.transcript;
      conversationHistory.push({ role: 'user', content: userInput });

      // Run through LangGraph
      const result = await graph.invoke({
        messages: conversationHistory
      });

      const agentResponse = result.messages[result.messages.length - 1].content;
      conversationHistory.push({ role: 'assistant', content: agentResponse });

      // Send back to Retell
      ws.send(JSON.stringify({
        type: 'response',
        response_id: data.response_id,
        content: agentResponse,
        content_complete: true
      }));
    }

    if (data.type === 'ping') {
      ws.send(JSON.stringify({ type: 'pong' }));
    }
  });
});

app.listen(8080);
```

### Example 2: LangGraph + Vapi.ai

```python
# server.py - Custom LLM endpoint for Vapi + LangGraph
from flask import Flask, request, Response
from langgraph.graph import StateGraph
import json

app = Flask(__name__)

# Your LangGraph setup
graph = build_your_langgraph()

@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    messages = data.get("messages", [])

    def generate():
        # Run LangGraph
        for event in graph.stream({"messages": messages}):
            # Extract response
            if "messages" in event:
                chunk = event["messages"][-1].content

                # Format as OpenAI streaming
                response = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "gpt-4",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(response)}\n\n"

        # Final chunk
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=8080)
```

### Example 3: LangGraph + Pipecat

```python
# Direct LangGraph integration with Pipecat
from pipecat.pipeline import Pipeline
from pipecat.processors.base import BaseProcessor
from langgraph.graph import StateGraph

class LangGraphProcessor(BaseProcessor):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.conversation = []

    async def process_frame(self, frame):
        if frame.type == "text":
            # User input
            self.conversation.append({"role": "user", "content": frame.text})

            # Run through LangGraph
            result = await self.graph.ainvoke({"messages": self.conversation})

            # Extract response
            response = result["messages"][-1].content
            self.conversation.append({"role": "assistant", "content": response})

            # Output
            return TextFrame(response)

        return frame

# Build pipeline
async def main():
    graph = build_your_langgraph()

    pipeline = Pipeline([
        WebsocketTransport(),
        DeepgramSTT(),
        LangGraphProcessor(graph),  # Your LangGraph here!
        ElevenLabsTTS()
    ])

    await pipeline.run()
```

---

## Recommendations

### For LangGraph Integration:
**1st Choice: Retell AI** - Best WebSocket integration, managed audio
**2nd Choice: Vapi.ai** - Flexible but requires more work
**3rd Choice: Pipecat** - Most control, fully open source

### For Fastest Production:
**1st Choice: Retell AI** - Best interruptions, easiest setup
**2nd Choice: Bland AI** - Visual builder, non-technical friendly

### For Complete Control:
**1st Choice: Pipecat** - Simplest open source
**2nd Choice: Vocode** - More features, LangChain support

### For Enterprise Scale:
**1st Choice: Vapi.ai** - Kubernetes-based, millions of calls
**2nd Choice: Retell AI** - Excellent reliability, compliance

---

**Last Updated:** November 21, 2025
