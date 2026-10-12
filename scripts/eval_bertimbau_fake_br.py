#!/usr/bin/env python3
"""
Avaliação do Modelo 2: BERTimbau Fake-BR (vzani/portuguese-fake-news-classifier-bertimbau-combined)
Tamanho: ~415 MB (PyTorch FP32).
"""
import json
import os
import time
import tracemalloc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bertimbau_fake_br")

def run_evaluation():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("=== [2/4] AVALIAÇÃO: BERTimbau Fake-BR (vzani/portuguese-fake-news-classifier-bertimbau-combined) ===")
    
    tracemalloc.start()
    t0 = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    
    load_time = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    weights_path = os.path.join(MODEL_DIR, "model.safetensors")
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"Modelo carregado em {load_time:.2f}s | Arquivo: {file_size_mb:.1f} MB | Pico RAM: {peak_mem / (1024*1024):.1f} MB")
    
    # Label 0: FAKE | Label 1: VERDADEIRO
    id2label = {0: "FAKE", 1: "VERDADEIRO"}
    
    results = []
    latencies = []
    correct_claim_only = 0
    correct_with_evidence = 0
    
    print("\nExecutando inferências (Modo 1: Alegação Isolada | Modo 2: Alegação + Evidência)...")
    for item in dataset:
        claim = item["claim"]
        evidence = item["evidence"]
        gt = item["ground_truth"]
        
        # Teste A: Alegação Pura
        t_start = time.perf_counter()
        inputs_claim = tokenizer(claim, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs_claim = model(**inputs_claim)
            probs_claim = torch.softmax(outputs_claim.logits, dim=-1)[0].numpy()
        lat_claim = (time.perf_counter() - t_start) * 1000
        latencies.append(lat_claim)
        
        pred_label_claim = id2label[int(np.argmax(probs_claim))]
        
        # Teste B: Alegação + Evidência
        combined_text = f"Alegação: {claim}. Fato e evidências apuradas: {evidence}"
        inputs_comb = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs_comb = model(**inputs_comb)
            probs_comb = torch.softmax(outputs_comb.logits, dim=-1)[0].numpy()
        
        pred_label_comb = id2label[int(np.argmax(probs_comb))]
        
        # Mapeamento do veredito:
        # Se prob de FAKE for > 0.65 -> FAKE
        # Se prob de VERDADEIRO for > 0.65 -> VERDADEIRO
        # Se entre 0.35 e 0.65 -> SUSPEITO / DÚVIDA
        def calibrate_verdict(prob_fake, prob_true):
            if prob_fake > 0.70:
                return "FAKE"
            elif prob_true > 0.70:
                return "VERDADEIRO"
            else:
                return "SUSPEITO"
                
        calib_claim = calibrate_verdict(probs_claim[0], probs_claim[1])
        calib_comb = calibrate_verdict(probs_comb[0], probs_comb[1])
        
        is_exact_claim = (calib_claim == gt)
        is_exact_comb = (calib_comb == gt)
        
        if is_exact_claim:
            correct_claim_only += 1
        if is_exact_comb:
            correct_with_evidence += 1
            
        results.append({
            "id": item["id"],
            "claim": claim,
            "ground_truth": gt,
            "pred_claim_only": {
                "verdict": calib_claim,
                "raw_pred": pred_label_claim,
                "prob_fake": round(float(probs_claim[0]), 4),
                "prob_true": round(float(probs_claim[1]), 4),
                "latency_ms": round(lat_claim, 2),
                "exact": is_exact_claim,
            },
            "pred_with_evidence": {
                "verdict": calib_comb,
                "raw_pred": pred_label_comb,
                "prob_fake": round(float(probs_comb[0]), 4),
                "prob_true": round(float(probs_comb[1]), 4),
                "exact": is_exact_comb,
            }
        })
        
        sym_claim = "✅" if is_exact_claim else "❌"
        sym_comb = "✅" if is_exact_comb else "❌"
        print(f"[{item['id']:02d}] GT: {gt:<10} | Alegação: {sym_claim} {calib_claim:<10} (pFake:{probs_claim[0]:.2f}) | Com Evidência: {sym_comb} {calib_comb:<10} (pFake:{probs_comb[0]:.2f}) | {lat_claim:.1f}ms")

    total = len(dataset)
    acc_claim = (correct_claim_only / total) * 100
    acc_comb = (correct_with_evidence / total) * 100
    avg_lat = sum(latencies) / len(latencies)
    
    print("\n--- RESUMO CONSOLIDADO: BERTimbau Fake-BR ---")
    print(f"Acurácia (Alegação Isolada): {acc_claim:.1f}% ({correct_claim_only}/{total})")
    print(f"Acurácia (Alegação + Evidência): {acc_comb:.1f}% ({correct_with_evidence}/{total})")
    print(f"Latência Média: {avg_lat:.2f} ms")
    print(f"P95 Latência: {np.percentile(latencies, 95):.2f} ms")
    
    out_path = os.path.join(os.path.dirname(__file__), "results_bertimbau_fake_br.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "BERTimbau-Fake-BR",
            "file_size_mb": file_size_mb,
            "accuracy_claim_only": acc_claim,
            "accuracy_with_evidence": acc_comb,
            "avg_latency_ms": avg_lat,
            "peak_mem_mb": peak_mem / (1024*1024),
            "items": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Resultados detalhados salvos em {out_path}\n")

if __name__ == "__main__":
    run_evaluation()
