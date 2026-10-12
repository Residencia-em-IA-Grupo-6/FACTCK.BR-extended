#!/usr/bin/env python3
"""
Avaliação do Modelo 1: MiniLMv2 NLI ONNX (onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX)
Tamanho: ~102 MB (INT8 ONNX).
"""
import json
import os
import time
import tracemalloc
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")

def run_evaluation():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    model_id = "onnx-community/multilingual-MiniLMv2-L6-mnli-xnli-ONNX"
    print(f"=== [1/4] AVALIAÇÃO: MiniLMv2 NLI ONNX ({model_id}) ===")
    
    tracemalloc.start()
    t0 = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    onnx_path = hf_hub_download(repo_id=model_id, filename="onnx/model_int8.onnx")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    load_time = time.perf_counter() - t0
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Modelo carregado em {load_time:.2f}s | Arquivo: {file_size_mb:.1f} MB | Pico RAM: {peak_mem / (1024*1024):.1f} MB")
    
    # Relações NLI:
    # 0: entailment (apóia -> VERDADEIRO)
    # 1: neutral (sem relação / parcial -> SUSPEITO)
    # 2: contradiction (contradiz -> FAKE)
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    results = []
    latencies = []
    correct_raw = 0
    correct_binary = 0
    
    print("\nExecutando inferências...")
    for item in dataset:
        claim = item["claim"]
        evidence = item["evidence"]
        gt = item["ground_truth"]
        expected_nli = item["expected_nli"]
        
        # Premissa = Notícia/Evidência; Hipótese = Alegação a ser checada
        t_start = time.perf_counter()
        inputs = tokenizer(evidence, claim, return_tensors="np", truncation=True, max_length=512)
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        logits = session.run(None, ort_inputs)[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        lat = (time.perf_counter() - t_start) * 1000
        latencies.append(lat)
        
        pred_label = id2label[int(np.argmax(probs))]
        
        # Mapeamento do veredito a partir do estado NLI
        if pred_label == "contradiction":
            pred_verdict = "FAKE"
        elif pred_label == "entailment":
            pred_verdict = "VERDADEIRO"
        else:
            pred_verdict = "SUSPEITO"
            
        prob_dict = {
            "entailment": round(float(probs[0]), 4),
            "neutral": round(float(probs[1]), 4),
            "contradiction": round(float(probs[2]), 4),
        }
        
        is_exact = (pred_verdict == gt)
        if is_exact:
            correct_raw += 1
            
        # Avaliação binária (Fake vs Verdadeiro ignorando os neutros)
        if gt in ("FAKE", "VERDADEIRO"):
            if pred_verdict == gt:
                correct_binary += 1
                
        results.append({
            "id": item["id"],
            "claim": claim,
            "ground_truth": gt,
            "predicted_verdict": pred_verdict,
            "predicted_nli": pred_label,
            "expected_nli": expected_nli,
            "probabilities": prob_dict,
            "latency_ms": round(lat, 2),
            "exact_match": is_exact,
        })
        
        symbol = "✅" if is_exact else "❌"
        print(f"[{item['id']:02d}] {symbol} GT: {gt:<10} | Pred: {pred_verdict:<10} ({pred_label}) | Contrad: {prob_dict['contradiction']:.2f} | Entail: {prob_dict['entailment']:.2f} | Neut: {prob_dict['neutral']:.2f} | {lat:.1f}ms")

    total = len(dataset)
    binary_total = 20 # 10 fakes + 10 true
    acc_exact = (correct_raw / total) * 100
    acc_binary = (correct_binary / binary_total) * 100
    avg_lat = sum(latencies) / len(latencies)
    
    print("\n--- RESUMO CONSOLIDADO: MiniLMv2 NLI ONNX ---")
    print(f"Acurácia Geral (3 classes): {acc_exact:.1f}% ({correct_raw}/{total})")
    print(f"Acurácia Binária (Fake/True): {acc_binary:.1f}% ({correct_binary}/{binary_total})")
    print(f"Latência Média: {avg_lat:.2f} ms")
    print(f"P95 Latência: {np.percentile(latencies, 95):.2f} ms")
    
    out_path = os.path.join(os.path.dirname(__file__), "results_minilm_nli.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "MiniLMv2-NLI-ONNX",
            "file_size_mb": file_size_mb,
            "accuracy_3class": acc_exact,
            "accuracy_binary": acc_binary,
            "avg_latency_ms": avg_lat,
            "peak_mem_mb": peak_mem / (1024*1024),
            "items": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Resultados detalhados salvos em {out_path}\n")

if __name__ == "__main__":
    run_evaluation()
