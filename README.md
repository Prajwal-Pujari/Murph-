# MURPH - Intelligent Voice Assistant & Browser Automation Agent

[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/yourusername/murph-ai-assistant)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)
[![Svelte](https://img.shields.io/badge/Svelte-Latest-FF3E00.svg)](https://svelte.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MURPH is an advanced, voice-activated AI assistant featuring natural language processing, browser automation, and contextual memory. Designed for seamless voice interactions, MURPH combines state-of-the-art speech recognition, intelligent conversation management, and system-level control capabilities to deliver a comprehensive virtual assistant experience.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [System Dependencies](#system-dependencies)
  - [Ollama Setup](#ollama-setup)
  - [ChromaDB Installation](#chromadb-installation)
  - [Backend Installation](#backend-installation)
  - [Frontend Installation](#frontend-installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

### Core Capabilities

- **Advanced Speech Recognition**: Leverages OpenAI Whisper for high-accuracy speech-to-text conversion
- **Contextual Conversation Memory**: Utilizes ChromaDB vector database for persistent conversation history and context retention
- **Natural Voice Synthesis**: Implements Piper TTS for natural-sounding male voice output with gTTS fallback support
- **Adaptive Personality System**: Configurable humor levels (0-100%) ranging from professional to highly personable interactions
- **Full Browser Automation**: Selenium-powered web browser control for autonomous navigation and interaction

### Browser Control

- Website navigation and URL management
- Integrated search across major platforms (Google, Wikipedia, YouTube)
- YouTube video playback control (play/pause/volume/navigation)
- Page manipulation (scrolling, content reading, tab management)
- Multi-tab session handling
- Application switching and focus management

### System Integration

- Application launch and control
- Automated text input across applications
- File system operations (read/write/list)
- Directory navigation
- Inter-application communication

### Intelligence Features

- Real-time weather data retrieval
- Time and date queries
- Web search integration
- Conversation history tracking and analysis
- Context-aware response generation

## Architecture

MURPH employs a modern, microservices-inspired architecture:

- **Backend**: FastAPI server handling voice processing, AI inference, and system operations
- **Frontend**: Svelte-based responsive UI for voice interaction and visual feedback
- **AI Engine**: Ollama-powered LLM (Llama 3) for natural language understanding
- **Vector Database**: ChromaDB for semantic search and conversation memory
- **Speech Pipeline**: Whisper → Ollama → Piper/gTTS for complete voice interaction

## Prerequisites

Ensure the following software is installed on your system:

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Backend runtime |
| Node.js | 18+ | Frontend development |
| FFmpeg | Latest | Audio processing |
| Ollama | Latest | LLM inference |
| Chrome | Latest | Browser automation |
| Git | Latest | Version control |

## Installation

### System Dependencies

#### FFmpeg Installation

**Windows**:
```bash
# Download from https://ffmpeg.org/download.html
# Extract and add bin folder to System PATH
# Verify installation
ffmpeg -version
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install ffmpeg -y
ffmpeg -version
```

**macOS**:
```bash
brew install ffmpeg
ffmpeg -version
```

### Ollama Setup

1. **Install Ollama**:
   
   Visit [https://ollama.ai](https://ollama.ai) and download the appropriate installer for your operating system.

2. **Pull Required Model**:
   ```bash
   # Download Llama3.1:8b-instruct-q4_K_M
   ollama pull llama3.1:8b-instruct-q4_K_M
   
   # Verify installation
   ollama list
   
   # Test model (optional)
   ollama run llama3:8b "Hello, test message"
   ```

3. **Start Ollama Service**:
   ```bash
   # Ollama typically runs as a background service
   # If not running, start manually:
   ollama serve
   ```

### ChromaDB Installation

ChromaDB is included in the Python dependencies and will be installed automatically. However, for optimal performance, ensure the following:

1. **System Requirements**:
   - Minimum 4GB RAM available
   - 1GB free disk space for database storage

2. **Installation Verification**:
   ```bash
   # After pip install, verify ChromaDB
   python -c "import chromadb; print(chromadb.__version__)"
   ```

3. **Database Initialization**:
   
   ChromaDB will automatically initialize on first run. The database files will be stored in:
   ```
   ./memory_db/
   ```

4. **Configuration** (Optional):
   
   For production deployments, consider ChromaDB's client-server mode:
   ```bash
   # Install ChromaDB server
   pip install chromadb[server]
   
   # Run ChromaDB server
   chroma run --path ./memory_db
   ```

### Backend Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/Prajwal-Pujari/Murph-.git
   cd murph-ai-assistant
   ```

2. **Create Virtual Environment**:
   ```bash
   # Create environment
   python -m venv venv
   
   # Activate environment
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   **requirements.txt contents**:
   ```txt
   fastapi==0.104.1
   uvicorn[standard]==0.24.0
   aiohttp==3.9.1
   torch==2.1.0
   openai-whisper==20231117
   pyttsx3==2.90
   chromadb==0.4.18
   gtts==2.4.0
   piper-tts==1.2.0
   selenium==4.15.2
   webdriver-manager==4.0.1
   pyautogui==0.9.54
   pygetwindow==0.0.9
   python-multipart==0.0.6
   python-dotenv==1.0.0
   ```
   
   **Note**: If you encounter any dependency conflicts, consider using these compatible versions. For the latest versions, remove version specifications.

4. **Download Piper TTS Model**:
   ```bash
   # Create models directory
   mkdir -p models
   cd models
   
   # Download male voice model
   # Visit: https://github.com/rhasspy/piper/releases/
   # Download both files:
   # - en_US-hfc_male-medium.onnx
   # - en_US-hfc_male-medium.onnx.json
   
   # Or use wget (Linux/macOS):
   wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-hfc_male-medium.onnx
   wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-hfc_male-medium.onnx.json
   
   cd ..
   ```

5. **Initialize ChromaDB**:
   ```bash
   # ChromaDB will auto-initialize, but you can pre-create the directory
   mkdir -p memory_db
   
   # Test ChromaDB setup
   python -c "import chromadb; client = chromadb.PersistentClient(path='./memory_db'); print('ChromaDB initialized successfully')"
   ```

6. **Start Backend Server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Installation

1. **Navigate to Frontend Directory**:
   ```bash
   cd frontend
   ```

2. **Install Node Dependencies**:
   ```bash
   npm install
   ```

3. **Start Development Server**:
   ```bash
   npm run dev
   ```

4. **Build for Production** (Optional):
   ```bash
   npm run build
   npm run preview
   ```

### Access Application

Open your web browser and navigate to:
```
http://localhost:5173
```

The backend API will be available at:
```
http://localhost:8000
```

API documentation can be accessed at:
```
http://localhost:8000/docs
```

## Configuration

### Backend Configuration

Edit `main.py` to customize settings:

```python
# Ollama Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_TIMEOUT = 120  # seconds

# ChromaDB Configuration
CHROMA_DB_PATH = "./memory_db"
COLLECTION_NAME = "conversation_history"

# CORS Settings
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

# Voice Configuration
PIPER_MODEL_PATH = "models/en_US-hfc_male-medium.onnx"
USE_PIPER_TTS = True  # Set to False to use gTTS

# Personality Settings
DEFAULT_HUMOR_LEVEL = 85  # 0-100
```

### Frontend Configuration

Update API endpoint in `frontend/src/routes/+page.svelte`:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

### Environment Variables

Create a `.env` file in the project root:

```env
OLLAMA_API_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3:8b
OLLAMA_TIMEOUT=120

CHROMA_DB_PATH=./memory_db
PIPER_MODEL_PATH=./models/en_US-hfc_male-medium.onnx

CORS_ORIGINS=http://localhost:5173,http://localhost:3000

DEFAULT_HUMOR_LEVEL=85
```

## Usage

### Voice Interaction

**Primary Input Method**: Press and hold **SPACEBAR** to record audio, release to process.

### Command Examples

#### General Queries
```
"Hey MURPH, what's the current time?"
"What's the weather like in San Francisco?"
"Tell me about yourself"
```

#### Browser Automation
```
"Open Google and search for artificial intelligence"
"Navigate to Wikipedia and look up quantum computing"
"Play 'Stairway to Heaven' on YouTube"
"Pause the video"
"Increase volume to 80%"
"Go to the next video"
"Close this tab"
"Scroll down the page"
```

#### System Operations
```
"Open Visual Studio Code"
"Switch to Chrome"
"List files in the current directory"
"Read the contents of readme.txt"
"Write 'Hello World' to test.txt"
```

#### Personality Adjustment
```
"Set humor level to 100"  # Maximum personality
"Set humor level to 50"   # Balanced mode
"Set humor level to 0"    # Professional mode
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **SPACE** (hold) | Record voice input |
| **Ctrl + Shift + H** | View conversation history |
| **Esc** | Cancel recording |

## API Reference

### Endpoints

#### POST `/voice-chat`
Process voice input and return AI response.

**Request**: `multipart/form-data`
- `audio`: Audio file (WAV, MP3, etc.)

**Response**: `application/json`
```json
{
  "text": "Transcribed text",
  "response": "AI response text",
  "audio_url": "/audio/response.mp3",
  "timestamp": "2025-11-17T10:30:00Z"
}
```

#### GET `/history`
Retrieve conversation history.

**Response**: Array of conversation entries

#### POST `/set-humor`
Adjust personality humor level.

**Request**: `application/json`
```json
{
  "level": 85
}
```

## Troubleshooting

### Voice Synthesis Issues

**Symptom**: Female voice instead of male voice

**Solution**:
1. Verify Piper model files exist in `models/` directory
2. Check backend logs for Piper initialization messages
3. Ensure both `.onnx` and `.onnx.json` files are present
4. Restart backend server after adding models

### Ollama Connection Errors

**Symptom**: "Ollama is taking too long to respond" or timeout errors

**Solution**:
1. Verify Ollama service is running:
   ```bash
   ollama list
   ```
2. Pre-load the model to reduce first-request latency:
   ```bash
   ollama run llama3:8b
   ```
3. Check system resources (minimum 8GB RAM recommended)
4. Increase timeout in `main.py`:
   ```python
   OLLAMA_TIMEOUT = 180
   ```

### ChromaDB Issues

**Symptom**: Database initialization errors or persistence failures

**Solution**:
1. Ensure write permissions for `memory_db/` directory:
   ```bash
   chmod -R 755 memory_db
   ```
2. Delete and reinitialize database:
   ```bash
   rm -rf memory_db
   python -c "import chromadb; chromadb.PersistentClient(path='./memory_db')"
   ```
3. Check available disk space (minimum 1GB required)
4. Verify ChromaDB version compatibility:
   ```bash
   pip install --upgrade chromadb
   ```

### Browser Automation Failures

**Symptom**: Selenium commands not executing

**Solution**:
1. Update ChromeDriver:
   ```bash
   pip install --upgrade webdriver-manager
   ```
2. Ensure Chrome browser is up to date
3. Check Chrome is in system PATH
4. Grant necessary permissions for browser automation

### Audio Recording Issues

**Symptom**: "Failed to load audio" or microphone access denied

**Solution**:
1. Grant microphone permissions in browser settings
2. Verify FFmpeg installation:
   ```bash
   ffmpeg -version
   ```
3. Check browser console for detailed error messages
4. Test microphone with other applications
5. Use HTTPS or localhost (required for microphone access)

## Contributing

We welcome contributions from the community! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Commit your changes**:
   ```bash
   git commit -m "Add: Brief description of changes"
   ```
4. **Push to your fork**:
   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Use ESLint for JavaScript/Svelte code
- Write unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.

## Acknowledgments

MURPH is built upon several outstanding open-source projects:

- **[OpenAI Whisper](https://github.com/openai/whisper)** - State-of-the-art speech recognition
- **[Piper TTS](https://github.com/rhasspy/piper)** - High-quality neural text-to-speech
- **[Ollama](https://ollama.ai)** - Efficient local LLM inference
- **[ChromaDB](https://www.trychroma.com/)** - AI-native vector database
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[Svelte](https://svelte.dev/)** - Reactive frontend framework
- **[Selenium](https://www.selenium.dev/)** - Browser automation framework

## Contact

**Developer**: [@Gravity_Exists](https://x.com/Gravity_Exists)

**Project Repository**: [https://github.com/Prajwal-Pujari/Murph-](https://github.com/Prajwal-Pujari/Murph-)

**Issues & Support**: [GitHub Issues](https://github.com/yourusername/murph-ai-assistant/issues)

## Roadmap

### Upcoming Features

- [ ] Multi-language support (Spanish, French, German, Japanese)
- [ ] Custom wake word detection 
- [ ] Plugin architecture for third-party extensions
- [ ] Native mobile applications (iOS/Android)
- [ ] Voice cloning and custom voice profiles
- [ ] Calendar integration (Google Calendar, Outlook)
- [ ] Music streaming service integration (Spotify, Apple Music)
- [ ] Smart home device control (Home Assistant, HomeKit)
- [ ] Email management and notifications
- [ ] Document analysis and summarization
- [ ] Code generation and debugging assistance

### Long-term Vision

- Distributed deployment architecture
- Multi-user support with individual profiles
- Enterprise security features
- Cloud synchronization
- Advanced sentiment analysis
- Proactive assistance based on user patterns

---

**Version**: 1.0.1  
**Last Updated**: December 2025  
**Status**: Active Development
