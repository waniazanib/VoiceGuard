"""
Lightweight ONNX Runtime inference engine for performing real-time voice spoofing detections.
"""

import os
import numpy as np
import onnxruntime as ort

import config
import features


def softmax(logits: np.ndarray) -> np.ndarray:
    """Computes the softmax activation of log-odds scores (converts logits to probabilities)."""
    # Safely subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class OnnxInferenceEngine:
    """
    Inference Engine using ONNX Runtime for high-performance acoustic spoofing checks.
    Prefers GPU acceleration but seamlessly cascades to CPU.
    """
    def __init__(self, model_path: str = None):
        """
        Initializes the ONNX runtime session.
        If model_path is not specified, probes for FP16 and FP32 models respectively.
        """
        if model_path is None:
            # Prefer optimized FP16 model
            if os.path.exists(config.MODEL_ONNX_FP16_PATH):
                model_path = str(config.MODEL_ONNX_FP16_PATH)
            elif os.path.exists(config.MODEL_ONNX_FP32_PATH):
                model_path = str(config.MODEL_ONNX_FP32_PATH)
            else:
                raise FileNotFoundError(
                    "Aucun modèle ONNX disponible. "
                    "Please execute 'train.py' or 'export_onnx.py' to generate model artifacts."
                )
                
        print(f"[*] Initializing ONNX session with model: {model_path}")
        
        # Configure execution providers - prefer CUDA if available
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            # Let's add CUDAExecutionProvider if possible (and run CPU otherwise)
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
            
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, audio_path: str) -> dict:
        """
        Performs inference on a single audio file.
        
        Args:
            audio_path: Local path to target voice recording.
            
        Returns:
            result: Dictionary containing predictions:
                {
                    "label": "BONAFIDE" | "SPOOF",
                    "confidence": float (0.0 to 1.0),
                    "spoof_score": float (probability of being spoofed voice)
                }
        """
        # Load and preprocess using feature script
        waveform = features.load_audio(audio_path)
        cqt_feat = features.extract_cqt(waveform)
        
        # Shape: [1, 1, bins, frames]
        input_data = cqt_feat[np.newaxis, np.newaxis, :, :].astype(np.float32)
        
        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        logits = outputs[0]  # shape [1, 2]
        
        # Get probabilities via softmax
        probs = softmax(logits)[0]  # [p_bonafide, p_spoof]
        p_bonafide, p_spoof = probs[0], probs[1]
        
        label = "SPOOF" if p_spoof >= config.THRESHOLD else "BONAFIDE"
        confidence = p_spoof if label == "SPOOF" else p_bonafide
        
        return {
            "label": label,
            "confidence": float(confidence),
            "spoof_score": float(p_spoof)
        }

    def predict_batch(self, audio_paths: list[str]) -> list[dict]:
        """
        Batches preprocessing and runs single inference pass over multiple audio files.
        Useful for benchmarks, bulk audits, and bulk evaluations.
        
        Args:
            audio_paths: List of string file paths.
            
        Returns:
            predictions: List of dictionaries identical to predict().
        """
        if not audio_paths:
            return []
            
        features_list = []
        for path in audio_paths:
            waveform = features.load_audio(path)
            cqt_feat = features.extract_cqt(waveform)
            features_list.append(cqt_feat)
            
        # Stack inputs: [B, 1, bins, frames]
        input_data = np.stack(features_list, axis=0)[:, np.newaxis, :, :].astype(np.float32)
        
        # Run bulk session pass
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        logits_batch = outputs[0]  # shape [B, 2]
        
        results = []
        for logits in logits_batch:
            probs = softmax(logits[np.newaxis, :])[0]
            p_bonafide, p_spoof = probs[0], probs[1]
            
            label = "SPOOF" if p_spoof >= config.THRESHOLD else "BONAFIDE"
            confidence = p_spoof if label == "SPOOF" else p_bonafide
            
            results.append({
                "label": label,
                "confidence": float(confidence),
                "spoof_score": float(p_spoof)
            })
            
        return results
