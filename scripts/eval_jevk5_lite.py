#!/usr/bin/env python3
"""
Avaliação do Modelo 4: JevK5-Lite (alibiserikbay/JevK5-Lite)
Tamanho: ~1.65 GB (PyTorch FP32).
Arquitetura: DeBERTa-v3-large (437M) + Cabeça de decisão multi-tarefa calibrada.
"""
import json
import os
import time
import tracemalloc
import numpy as np

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "jevk5_lite")

def run_evaluation():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"=== [4/4] AVALIAÇÃO: JevK5-Lite ({MODEL_DIR}) ===")
    
    tracemalloc.start()
    t0 = time.perf_counter()
    
    from jevk5 import JevK5Lite
    model = JevK5Lite.from_pretrained(MODEL_DIR, threads=6)
    load_time = time.perf_counter() - t0
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Modelo carregado em {load_time:.2f}s | Pico RAM: {peak_mem / (1024*1024):.1f} MB")
    
    results = []
    latencies = []
    correct_3class = 0
    correct_binary = 0
    
    print("\nExecutando inferências JevK5Lite...")
    for item in dataset:
        claim = item["claim"]
        evidence = item["evidence"]
        gt = item["ground_truth"]
        
        text_input = f"Alegação: {claim}. Evidências fáticas e registros checados: {evidence}"
        
        t_start = time.perf_counter()
        res = model.classify(
            text_input,
            {"veredicto": ["falso", "verdadeiro", "suspeito"]}
        )
        lat = (time.perf_counter() - t_start) * 1000
        latencies.append(lat)
        
        ver_data = res.get("veredicto", {})
        chosen_labels = ver_data.get("labels", ["suspeito"])
        raw_choice = chosen_labels[0] if chosen_labels else "suspeito"
        probs = ver_data.get("probabilities", {})
        
        pred_map = {
            "falso": "FAKE",
            "verdadeiro": "VERDADEIRO",
            "suspeito": "SUSPEITO"
        }
        pred_choice = pred_map.get(raw_choice.lower(), "SUSPEITO")
        
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
            "probabilities": probs,
            "latency_ms": round(lat, 2),
            "exact_match": is_exact,
        })
        
        sym = "✅" if is_exact else "❌"
        p_f = probs.get("falso", 0.0)
        p_v = probs.get("verdadeiro", 0.0)
        p_s = probs.get("suspeito", 0.0)
        print(f"[{item['id']:02d}] {sym} GT: {gt:<10} | Pred: {pred_choice:<10} | F:{p_f:.2f} V:{p_v:.2f} S:{p_s:.2f} | {lat:.1f}ms")

    total = len(dataset)
    binary_total = 20
    acc_3class = (correct_3class / total) * 100
    acc_binary = (correct_binary / binary_total) * 100
    avg_lat = sum(latencies) / len(latencies)
    
    print("\n--- RESUMO CONSOLIDADO: JevK5-Lite ---")
    print(f"Acurácia Geral (3 classes): {acc_3class:.1f}% ({correct_3class}/{total})")
    print(f"Acurácia Binária (Fake/True): {acc_binary:.1f}% ({correct_binary}/{binary_total})")
    print(f"Latência Média: {avg_lat:.2f} ms")
    print(f"P95 Latência: {np.percentile(latencies, 95):.2f} ms")
    
    out_path = os.path.join(os.path.dirname(__file__), "results_jevk5_lite.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "JevK5-Lite",
            "file_size_mb": 1655.0,
            "accuracy_3class": acc_3class,
            "accuracy_binary": acc_binary,
            "avg_latency_ms": avg_lat,
            "peak_mem_mb": peak_mem / (1024*1024),
            "items": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Resultados detalhados salvos em {out_path}\n")

if __name__ == "__main__":
    run_evaluation()
