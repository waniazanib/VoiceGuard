"""
Exports the trained PyTorch model weights to ONNX (FP32) and optimizes/quantizes them to FP16.
Provides verification and execution benchmarking out of the box.
"""

import os
import time
import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import convert_float_to_float16

import config
from model import SpoofDetector


def export(pt_path: str = None) -> str:
    """
    Exports PyTorch model weights into ONNX FP32 and subsequently outputs ONNX FP16 version.
    Validates model consistency on randomized dummy data and evaluates execution latency.
    
    Args:
        pt_path: Local string path to PyTorch checkpoint. Defaults to config if not provided.
        
    Returns:
        fp16_onnx_path: String of the optimized FP16 model path.
    """
    if pt_path is None:
        pt_path = str(config.MODEL_PT_PATH)
        
    print(f"[*] Starting ONNX export pipeline from checkpoint: {pt_path}")
    
    # 1. Load model from weights
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PyTorch weight checkpoint not found at: {pt_path}")
        
    model = SpoofDetector.load_pretrained(pt_path)
    model.eval()
    
    # Check shape: a 4s wave @ 16kHz, CQT hop 512 yields 126 frames
    # CQT bins is configured as 84
    dummy_input = torch.randn(1, 1, config.CQT_BINS, 126, dtype=torch.float32)
    
    # 2. Export to ONNX FP32 with dynamic batch and time dimensions
    print(f"[*] Exporting PyTorch to ONNX FP32...")
    torch.onnx.export(
        model,
        dummy_input,
        str(config.MODEL_ONNX_FP32_PATH),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "time_steps"},
            "output": {0: "batch_size"}
        },
        opset_version=15
    )
    print(f"[+] Saved ONNX FP32 model: {config.MODEL_ONNX_FP32_PATH}")
    
    # 3. Convert FP32 ONNX model to FP16 ONNX format
    print(f"[*] Quantizing/Converting to ONNX FP16...")
    onnx_model_fp32 = onnx.load(str(config.MODEL_ONNX_FP32_PATH))
    onnx_model_fp16 = convert_float_to_float16(
        onnx_model_fp32,
        keep_io_types=True,  # Keeps entry / exit points compatible (e.g., FLOAT/INT labels)
    )
    onnx.save(onnx_model_fp16, str(config.MODEL_ONNX_FP16_PATH))
    print(f"[+] Saved ONNX FP16 model: {config.MODEL_ONNX_FP16_PATH}")
    
    # 4. Consistency verification
    print("[*] Validating correctness and output conformance...")
    # Instantiate inference engines
    with torch.no_grad():
        pytorch_out = model(dummy_input).numpy()
        
    # Run FP16 model on same dummy input
    ort_session = ort.InferenceSession(
        str(config.MODEL_ONNX_FP16_PATH),
        providers=["CPUExecutionProvider"]
    )
    
    ort_inputs = {"input": dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_out = ort_outputs[0]
    
    # Measure error bound
    max_diff = np.max(np.abs(pytorch_out - onnx_out))
    print(f"[i] Maximum absolute discrepancy (PyTorch vs ONNX FP16): {max_diff:.6f}")
    assert max_diff < 0.05, f"Validation failure! Discrepancy ({max_diff:.6f}) exceeds threshold 0.05!"
    print("[+] Model output consistency check: PASSED!")
    
    # 5. Review file sizes
    size_pt = os.path.getsize(pt_path) / 1024
    size_fp32 = os.path.getsize(config.MODEL_ONNX_FP32_PATH) / 1024
    size_fp16 = os.path.getsize(config.MODEL_ONNX_FP16_PATH) / 1024
    
    print("\nFile Size Summary:")
    print(f"  - PyTorch (.pt):       {size_pt:.2f} KB")
    print(f"  - ONNX FP32:          {size_fp32:.2f} KB")
    print(f"  - ONNX FP16:          {size_fp16:.2f} KB (Reduction: {100 * (1 - size_fp16/size_fp32):.1f}%)")
    
    # 6. Latency Benchmark
    print("\n[*] Benchmarking latency on 100 random runs (ONNX FP16 on CPU)...")
    latencies = []
    # Warmup
    for _ in range(10):
        ort_session.run(None, ort_inputs)
        
    for _ in range(100):
        start_time = time.perf_counter()
        ort_session.run(None, ort_inputs)
        latencies.append((time.perf_counter() - start_time) * 1000) # milliseconds
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    print(f"[+] Benchmark Results:")
    print(f"  - P50 (Median) Latency: {p50:.2f} ms")
    print(f"  - P95 Latency:          {p95:.2f} ms")
    
    return str(config.MODEL_ONNX_FP16_PATH)


if __name__ == "__main__":
    if not config.MODEL_PT_PATH.exists():
        print(f"[!] PyTorch model weight file not found at {config.MODEL_PT_PATH}.")
        print("[!] Generating mock PyTorch weights for export/testing/demo verification purposes...")
        # Create a blank model and save it to allow end-to-end evaluation
        mock_model = SpoofDetector()
        torch.save(mock_model.state_dict(), str(config.MODEL_PT_PATH))
        print(f"[+] Blank/random weights written to {config.MODEL_PT_PATH}")
        
    export(str(config.MODEL_PT_PATH))
