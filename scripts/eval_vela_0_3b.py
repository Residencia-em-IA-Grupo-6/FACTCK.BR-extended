#!/usr/bin/env python3
"""
Avaliação do Modelo 3: Vela 2.0 0.3B (vllm-sr/Vela-2.0-0.3B)
Tamanho: ~594 MB (ONNX FP16).
Arquitetura: ModernBERT (307M) + Decisão SystemOne nativa via ONNX.
"""
import json
import os
import sys
import time
import tracemalloc
import numpy as np
from huggingface_hub import snapshot_download

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")

def run_evaluation():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    model_id = "vllm-sr/Vela-2.0-0.3B"
    print(f"=== [3/4] AVALIAÇÃO: Vela 2.0 0.3B ({model_id} via ONNX) ===")
    
    tracemalloc.start()
    t0 = time.perf_counter()
    
    d = snapshot_download(model_id,
                          allow_patterns=["vela2_inference.py", "vela2_serve.py", "config.json", "calibration.json",
                                          "tokenizer.json", "onnx/model_fp16.onnx"])
    if d not in sys.path:
        sys.path.insert(0, d)
    from vela2_inference import Vela2
    
    model = Vela2.from_pretrained(d, backend="onnx", onnx_file="onnx/model_fp16.onnx")
    load_time = time.perf_counter() - t0
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    onnx_file = os.path.join(d, "onnx", "model_fp16.onnx")
    file_size_mb = os.path.getsize(onnx_file) / (1024 * 1024) if os.path.exists(onnx_file) else 594.0
    print(f"Modelo carregado em {load_time:.2f}s | Arquivo: {file_size_mb:.1f} MB | Pico RAM: {peak_mem / (1024*1024):.1f} MB")
    
    results = []
    latencies = []
    correct_3class = 0
    correct_binary = 0
    
    print("\nExecutando inferências SystemOne...")
    for item in dataset:
        claim = item["claim"]
        evidence = item["evidence"]
        gt = item["ground_truth"]
        
        state_text = f"Alegação: {claim}\nEvidências e registros jornalísticos checados: {evidence}"
        
        questions = {
            "veredicto": {
                "type": "choice",
                "instructions": "Avalie a veracidade da alegação estritamente confrontando com as evidências apresentadas.",
                "criteria": {
                    "FAKE": "alegação falsa, mito, desmentida por agências ou autoridades, boato ou fraude médica/comercial",
                    "VERDADEIRO": "alegação faticamente verdadeira, comprovada por fontes oficiais, jornalismo de referência ou ciência",
                    "SUSPEITO": "alegação exagerada, parcialmente verdadeira, imprecisa, desatualizada ou sem confirmação total",
                }
            }
        }
        
        t_start = time.perf_counter()
        res = model.system_one(state=state_text, questions=questions)
        lat = (time.perf_counter() - t_start) * 1000
        latencies.append(lat)
        
        ans = res.get("answers", {}).get("veredicto", {})
        pred_choice = ans.get("choice", "SUSPEITO")
        probs = ans.get("probabilities", {})
        conf = ans.get("confidence", 0.0)
        
        is_exact = (pred_choice == gt)
        if is_exact:
            correct_3class += 1
            
        if gt in ("FAKE", "VERDADEIRO"):
            if pred_choice == gt:
                correct_binary += 1
                
        results.append({
            "id": item["id"],
            "claim": claim,
            "ground_truth": gt,
            "predicted_verdict": pred_choice,
            "confidence": conf,
            "probabilities": probs,
            "latency_ms": round(lat, 2),
            "exact_match": is_exact,
        })
        
        sym = "✅" if is_exact else "❌"
        p_fake = probs.get("FAKE", 0.0)
        p_true = probs.get("VERDADEIRO", 0.0)
        p_susp = probs.get("SUSPEITO", 0.0)
        print(f"[{item['id']:02d}] {sym} GT: {gt:<10} | Pred: {pred_choice:<10} (conf: {conf:.2f}) | F:{p_fake:.2f} V:{p_true:.2f} S:{p_susp:.2f} | {lat:.1f}ms")

    total = len(dataset)
    binary_total = 20
    acc_3class = (correct_3class / total) * 100
    acc_binary = (correct_binary / binary_total) * 100
    avg_lat = sum(latencies) / len(latencies)
    
    print("\n--- RESUMO CONSOLIDADO: Vela 2.0 0.3B ---")
    print(f"Acurácia Geral (3 classes): {acc_3class:.1f}% ({correct_3class}/{total})")
    print(f"Acurácia Binária (Fake/True): {acc_binary:.1f}% ({correct_binary}/{binary_total})")
    print(f"Latência Média: {avg_lat:.2f} ms")
    print(f"P95 Latência: {np.percentile(latencies, 95):.2f} ms")
    
    out_path = os.path.join(os.path.dirname(__file__), "results_vela_0_3b.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Vela-2.0-0.3B-ONNX",
            "file_size_mb": file_size_mb,
            "accuracy_3class": acc_3class,
            "accuracy_binary": acc_binary,
            "avg_latency_ms": avg_lat,
            "peak_mem_mb": peak_mem / (1024*1024),
            "items": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Resultados detalhados salvos em {out_path}\n")

if __name__ == "__main__":
    run_evaluation()
