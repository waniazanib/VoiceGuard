# VoiceGuard

**AI-Powered Acoustic Deepfake & Voice Spoofing Detection Engine**

VoiceGuard is a production-ready acoustic deepfake classifier designed for live voice authentication, recruitment verification, and identity protection. Using a modified ResNet-18 CNN with Constant-Q Transform (CQT) spectrograms, it achieves fast, accurate detection of synthetic and spoofed voices with explainable predictions via GradCAM heatmaps.

---

## Features

- **High-Performance Classification**: ~5.4% Equal Error Rate (EER) on ASVspoof 2019 dataset
- **Ultra-Fast Inference**: 2.7ms per prediction using ONNX FP16 quantization (6.7× speedup vs PyTorch)
- **Explainable AI**: Gradient-weighted Class Activation Mapping (GradCAM) reveals decision drivers
- **Advanced Audio Representation**: Constant-Q Transform (CQT) captures logarithmic human auditory scale
- **Graceful Fallback**: Heuristic demo mode using spectral analysis when models unavailable
- **Dual Interface**: Gradio web UI + TypeScript/React backend server
- **Production-Ready**: ONNX FP16 export for lightweight CPU inference

---

## Tech Stack

| Component | Technologies |
|-----------|---------------|
| **Backend** | Python 3.9+, PyTorch 2.1, ONNX Runtime 1.16 |
| **ML Models** | ResNet-18 CNN, Constant-Q Transform, GradCAM |
| **Frontend** | Gradio 6.15, React 19, TypeScript, Vite |
| **Audio Processing** | Librosa, SoundFile, SciPy |
| **Data** | ASVspoof 2019 (via KaggleHub) |
| **Server** | Express.js, Node.js |
| **ML Ops** | ONNX model export, FP16 quantization |

---

## Installation

### Prerequisites
- Python 3.9 or higher
- Git
- 2GB disk space (models + dependencies)

### Step 1: Clone Repository
```bash
git clone https://github.com/waniazanib/VoiceGuard.git
cd VoiceGuard
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
npm install  # For TypeScript/React components
```

### Step 4: Configure Environment (Optional)
```bash
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and APP_URL if using external APIs
```

---

## Usage

### Web Interface (Recommended)
```bash
python app.py
```
Opens at `http://127.0.0.1:7860`

**Features:**
- Upload or record voice samples
- View real-time spoof risk assessment
- See GradCAM heatmap explanation overlay
- Benchmark & architecture tabs

### Model Training
```bash
python train.py
```
**Process:**
1. Downloads ASVspoof 2019 dataset from Kaggle
2. Trains ResNet-18 on CQT spectrograms (30 epochs)
3. Tracks Equal Error Rate (EER) per epoch
4. Saves PyTorch checkpoint: `saved_models/best_model.pt`
5. Exports ONNX FP32 & FP16 quantized models
6. Generates metadata JSON

**Requirements:**
- Kaggle credentials at `~/.kaggle/kaggle.json`
- 24GB+ GPU memory (or CPU fallback, slower)

### ONNX Export
```bash
python export_onnx.py
```
Converts PyTorch model to ONNX FP16 for production deployment.

### Inference API
```python
from inference import OnnxInferenceEngine

engine = OnnxInferenceEngine()
result = engine.predict("voice_sample.wav")
print(result)  # {"label": "BONAFIDE", "confidence": 0.95, "spoof_score": 0.05}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini AI API key | (optional) |
| `APP_URL` | Application deployment URL | (auto-detected) |
| `SAMPLE_RATE` | Audio preprocessing rate | 16000 Hz |
| `CLIP_DURATION` | Audio clip truncation/padding | 4 seconds |
| `CQT_BINS` | Constant-Q Transform bins | 84 |
| `THRESHOLD` | Spoof classification threshold | 0.5 |

See `.env.example` for template.

---

## Folder Structure

```
VoiceGuard/
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js dependencies
│
├── app.py                       # Gradio web interface (main entry point)
├── config.py                    # Configuration constants
├── server.ts                    # Express.js backend server
│
├── model.py                     # ResNet-18 architecture definition
├── train.py                     # Training pipeline
├── inference.py                 # ONNX inference engine
├── export_onnx.py              # PyTorch to ONNX converter
│
├── features.py                  # Audio preprocessing & CQT extraction
├── gradcam.py                   # GradCAM visualization
│
├── saved_models/               # (Generated) Model checkpoints
│   ├── best_model.pt
│   ├── model_fp32.onnx
│   ├── model_fp16.onnx
│   └── model_metadata.json
│
├── src/                         # Frontend React/TypeScript source (Vite)
├── index.html                   # HTML entry point
├── vite.config.ts              # Vite build configuration
├── tsconfig.json               # TypeScript configuration
│
└── .env.example                # Environment template
```

---

## Performance Benchmarks

| Model | Parameters | EER | CPU Latency | Notes |
|-------|-----------|-----|-------------|-------|
| **ResNet-18 + CQT (Ours)** | 11.2M | 5.4% | 2.7ms (ONNX FP16) | Production model |
| ResNet-18 + MFCC | 11.2M | 9.2% | 18ms (PyTorch) | Baseline |
| AASIST (SOTA) | 0.3M | 0.83% | 45ms | Reference only |

**EER Definition**: Equal Error Rate where False Acceptance Rate (FAR) = False Rejection Rate (FRR). Lower is better.

---

## Architecture

```
Audio Input (Upload/Microphone)
    ↓
Preprocessing (16kHz mono, 4s clip)
    ↓
CQT Spectrogram (84 bins × 126 frames)
    ↓
ResNet-18 CNN (Grayscale convolution)
    ↓
GradCAM Activation Mapper (Explainability)
    ↓
Softmax Output → BONAFIDE vs SPOOF
```

**Key Components:**
- **Constant-Q Transform**: Logarithmically-spaced frequency bins matching human auditory perception
- **Modified ResNet-18**: Single-channel input for grayscale spectrograms
- **GradCAM**: Backpropagation-based heatmap showing decision-critical time-frequency regions
- **ONNX FP16**: Quantized model for embedded & edge deployment

---

## Future Improvements

- [ ] Multi-language speaker adaptation
- [ ] Real-time streaming inference (WebRTC integration)
- [ ] Mobile deployment (TensorFlow Lite/ONNX for iOS/Android)
- [ ] Fine-tuning on custom datasets via API
- [ ] Ensemble models (combining CQT + MFCC + Mel-spectrogram)
- [ ] Database logging for fraud detection patterns
- [ ] Hardware acceleration (GPU batch inference)
- [ ] Confidence calibration & uncertainty quantification
- [ ] Voice activity detection (VAD) preprocessing
- [ ] Support for multiple audio formats (FLAC, OGG, MP3)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature description"`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a Pull Request

---

## Author

**waniazanib** — [GitHub](https://github.com/waniazanib)

For questions or support, open an issue on the [GitHub Issues](https://github.com/waniazanib/VoiceGuard/issues) page.
