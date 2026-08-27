# Call Center Intelligence System

A production-grade, multi-agent AI pipeline that transforms raw call center audio into comprehensive analysis and actionable insights. Built with LangGraph, faster-whisper, and dual LLM-powered analysis to deliver structured transcripts, quality scores, compliance reports, and downloadable artifacts.

## 🎯 Overview

This system provides an intelligent audio processing pipeline designed for call centers, combining advanced speech-to-text, security-first PII redaction, prompt injection defense, and multi-dimensional quality analysis. The platform delivers a web-based interface for audio analysis, call history browsing, and real-time pipeline observability.

## ✨ Key Features

- **Audio Processing**: Fast and accurate speech-to-text using faster-whisper with speaker identification
- **Security First**: Built-in prompt injection detection and PII redaction to ensure compliance
- **Multi-Agent Analysis**: Dual LLM-powered sequential analysis for comprehensive call evaluation
- **Quality Scoring**: Five-dimensional quality scorecard covering key call center KPIs
- **Compliance Tracking**: Automated compliance flag detection and reporting
- **Downloadable Reports**: Export analysis as PDF and JSON artifacts
- **Call History**: Master-detail browser interface for exploring all analyzed calls
- **Pipeline Observability**: Real-time dashboard showing pipeline health, LangSmith tracing, and audit logs

## 🏗️ Architecture

The system is built around a **LangGraph state machine** with 8 named nodes:

### Processing Stages (7 nodes)
1. **Transcription**: Speech-to-text conversion with speaker labeling
2. **Intake**: Initial data validation and preparation
3. **Injection Detection**: Prompt injection defense and sanitization
4. **PII Redaction**: Personal information detection and masking
5. **Summarization**: Executive summary generation
6. **QA Scoring**: Five-dimension quality assessment
7. **Report Generation**: Structured analysis compilation

### Terminal Outcomes (3 nodes)
- **Report**: Successful analysis with generated report
- **Supervisor Review**: Flagged cases requiring manual intervention
- **Error**: Processing failures with diagnostic information

## 📊 Agent Graph Flow

The pipeline orchestrates the following workflow using conditional routing:

```
                             ┌─────────────────────────────────┐
                             │      START                      │
                             └────────────┬────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────┐
                        │  1. Intake Validation       │
                        │  (validate audio & metadata)│
                        └────────┬────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ (intake_failed?)        │
                    ▼                         ▼
              ┌──────────┐          ┌─────────────────┐
              │  ERROR   │          │  2. Transcribe  │
              │  NODE    │          │  (Whisper STT)  │
              └────┬─────┘          └────────┬────────┘
                   │                         │
                   │              ┌──────────┴──────────┐
                   │              │ (transcription     │
                   │              │  failed?)          │
                   │              ▼                    ▼
                   │         ┌──────────┐    ┌──────────────────┐
                   │         │  ERROR   │    │  3. Injection    │
                   │         │  NODE    │    │  Check           │
                   │         └────┬─────┘    │  (Defense)       │
                   │              │          └────────┬─────────┘
                   │              │                   │
                   │              │      ┌────────────┴────────────┐
                   │              │      │ (flagged_for_review?)   │
                   │              │      ▼                        ▼
                   │              │  ┌──────────┐      ┌──────────────────┐
                   │              │  │ SUPERVISOR│     │  4. PII Redact   │
                   │              │  │ REVIEW    │     │  (Mask PII data) │
                   │              │  │ NODE      │     └────────┬─────────┘
                   │              │  └────┬─────┘              │
                   │              │       │                    ▼
                   │              │       │      ┌──────────────────────────┐
                   │              │       │      │  5. Summarize & QA       │
                   │              │       │      │  (Generate summary &     │
                   │              │       │      │   quality scorecard)     │
                   │              │       │      └────────┬────────────────┘
                   │              │       │              │
                   │              │       │   ┌──────────┴──────────┐
                   │              │       │   │ (error or critical  │
                   │              │       │   │  compliance flags?) │
                   │              │       ▼   ▼                    ▼
                   │              │   ┌──────────┐      ┌─────────────────┐
                   │              │   │ SUPERVISOR│     │  6. Generate    │
                   │              │   │ REVIEW    │     │  Report         │
                   │              │   │ NODE      │     │  (Compile final)│
                   │              │   └────┬─────┘     └────────┬────────┘
                   │              │        │                   │
                   └──────────────┴────────┴───────────────────┴────────┐
                                                                         │
                                        ┌────────────────────────────────┘
                                        │
                                        ▼
                                    ┌────────┐
                                    │  END   │
                                    └────────┘

Routing Logic:
- intake_step → (intake_failed) → error_step | else → transcribe_step
- transcribe_step → (transcription_failed) → error_step | else → injection_check_step
- injection_check_step → (flagged_for_review) → error_step | else → pii_redact_step
- pii_redact_step → summarize_and_qa_step (deterministic)
- summarize_and_qa_step → (error) → error_step | (critical compliance flags) → supervisor_step | else → report_step
- report_step / error_step / supervisor_step → END
```

### Node Execution Details

| Node | Function | Handles | Routing |
|------|----------|---------|---------|
| **Intake** | Validate audio & metadata | Input validation, file checks | Error if invalid |
| **Transcribe** | Speech-to-text conversion | Converts audio to text, speaker detection | Error if fails |
| **Injection Check** | Prompt injection defense | Detects malicious input in transcript | Supervisor review if detected |
| **PII Redact** | Personal data masking | Redacts SSN, phone, email, etc. | Always proceeds |
| **Summarize & QA** | Analysis & scoring | Executive summary, 5-dim quality score, compliance flags | Error if fails; supervisor if critical flags |
| **Report** | Final compilation | Persists analysis, generates PDF/JSON | Terminal: Report outcome |
| **Error** | Failure handler | Logs errors, persists partial state | Terminal: Error outcome |
| **Supervisor** | Manual review queue | Routes flagged/critical cases | Terminal: Supervisor review outcome |

## 🖥️ Web Application Features

### Tab 1: Analyze Call
- Upload or record call audio
- View speaker-labeled transcript
- Read executive summary
- Review five-dimension quality scorecard
- Check compliance flags
- Download PDF and JSON reports

### Tab 2: All MP3 History
- Browse complete history of analyzed calls
- Master-detail interface for filtering and searching
- Quick access to previous analyses
- Historical trend analysis

### Tab 3: Observability
- Pipeline health metrics and status
- LangSmith tracing integration and debugging
- Audit log viewer with event details
- Performance monitoring and analytics

## 🛠️ Technology Stack

- **Audio Processing**: [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- **Workflow Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/)
- **LLM Integration**: Claude API (dual models for sequential analysis)
- **Security**: Custom PII redaction and prompt injection detection
- **Frontend**: Gradio (web-based UI with three tabs)
- **Database**: SQLite with ORM models
- **Audit & Observability**: LangSmith integration for tracing and logging

## 📋 Getting Started

### Prerequisites
- Python 3.9+
- FFmpeg (for audio processing)
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Call-Center-Intelligence-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

4. Initialize the database:
```bash
make init-db
```

### Running the Application

Start the web application:
```bash
make run
# or
python app.py
```

The application will be available at `http://localhost:7860`

## 🧪 Testing

Run the test suite:
```bash
make test
```

Run integration tests:
```bash
pytest tests/integration/ -v
```

## 📁 Project Structure

```
.
├── app.py                          # Main application entry point
├── src/
│   ├── agents/                     # Agent implementations (7 processing stages)
│   │   ├── transcription.py
│   │   ├── intake.py
│   │   ├── qa_scoring.py
│   │   ├── report.py
│   │   └── summarization.py
│   ├── database/                   # Data persistence layer
│   │   ├── connection.py
│   │   ├── models.py
│   │   └── init_db.py
│   ├── graph/                      # LangGraph workflow
│   │   ├── state.py                # State machine definition
│   │   └── workflow.py             # Graph orchestration
│   ├── security/                   # Security & compliance
│   │   ├── audit.py                # Audit logging
│   │   ├── pii_redactor.py         # PII detection and redaction
│   │   └── injection_detector.py   # Prompt injection defense
│   ├── services/                   # Core services
│   │   └── pipeline.py             # End-to-end pipeline orchestration
│   ├── ui/                         # Gradio web interface
│   │   ├── main.py                 # Main UI
│   │   └── tabs/                   # UI tabs
│   │       ├── analyze.py          # Analyze Call tab
│   │       ├── history.py          # All MP3 History tab
│   │       └── observability.py    # Observability tab
│   ├── utils/                      # Utilities
│   │   ├── audio.py                # Audio processing utilities
│   │   ├── config.py               # Configuration management
│   │   └── llm_factory.py          # LLM initialization
│   └── app_globals.py              # Shared application state
├── tests/                          # Test suite
│   ├── integration/                # End-to-end integration tests
│   └── unit/                       # Unit tests
├── Dockerfile                      # Container configuration
├── Makefile                        # Build and run commands
└── requirements.txt                # Python dependencies
```

## 🔒 Security & Compliance

- **PII Redaction**: Automatic detection and masking of sensitive information (SSNs, phone numbers, email addresses, etc.)
- **Prompt Injection Detection**: Defense mechanism against prompt injection attacks
- **Audit Logging**: Complete audit trail of all system actions and data access
- **Compliance Flags**: Automated detection of compliance violations and suspicious patterns

## 📊 Data Flow

```
Audio Input → Transcription → Intake Validation → Injection Detection → 
PII Redaction → Summarization → QA Scoring → Report Generation → 
[Report/Supervisor Review/Error]
```

## 🚀 Deployment

### Docker
```bash
docker build -t call-center-intelligence .
docker run -p 7860:7860 call-center-intelligence
```

### Production Considerations
- Configure LangSmith API key for production tracing
- Set up database backups and replication
- Configure audit log retention policies
- Set appropriate resource limits for audio processing
- Use environment-based configuration for sensitive credentials

## 📈 Observability

The system integrates with **LangSmith** for detailed tracing and debugging:
- View complete execution traces for each pipeline run
- Monitor performance metrics and latency
- Debug failures with full context
- Track LLM token usage and costs

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests to ensure quality
4. Submit a pull request

## 📝 License

[Add appropriate license]

## 📞 Support

For issues or questions, please open an issue in the repository.