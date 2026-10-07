#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_onnx_classifier.py

Treinamento de classificador neural de temas para claims jornalísticas com:
1. Fine-tuning de DistilBERT multilíngue em PyTorch (com aceleração Apple Silicon MPS / CUDA / CPU)
2. Exportação para formato ONNX (FP32)
3. Quantização dinâmica para INT8 via ONNX Runtime (onnxruntime.quantization)
4. Benchmarking completo (Acurácia, F1-Score, Latência em ms e Redução de Tamanho em MB)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class TextClaimDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        logger.info("Usando acelerador Apple Silicon (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Usando acelerador NVIDIA (CUDA)")
        return torch.device("cuda")
    else:
        logger.info("Usando CPU")
        return torch.device("cpu")


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_pytorch(model, dataloader, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def evaluate_onnx_session(session: ort.InferenceSession, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
    all_preds = []
    latencies = []

    for text in texts:
        inputs = tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        t0 = time.perf_counter()
        outputs = session.run(None, ort_inputs)
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(dt)

        logits = outputs[0]
        pred = int(np.argmax(logits, axis=-1)[0])
        all_preds.append(pred)

    y_true = np.array(labels)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_samples_sec": float(1000.0 / np.mean(latencies)),
        "predictions": y_pred,
    }


def main():
    parser = argparse.ArgumentParser(description="Treinamento de Classificador com ONNX Runtime INT8")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/dataset_classificacao_temas.tsv",
        help="Caminho para o dataset TSV de classificação",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-multilingual-cased",
        help="Nome ou caminho do modelo base Hugging Face",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Diretório onde os modelos e artefatos serão salvos",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas de treinamento")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamanho do lote (batch size)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Taxa de aprendizado inicial")
    parser.add_argument("--max-len", type=int, default=128, help="Comprimento máximo de tokens")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidade")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dataset
    logger.info(f"Carregando dataset de {args.dataset}...")
    df = pd.read_csv(args.dataset, sep="\t")
    logger.info(f"Total de registros carregados: {len(df)}")

    # Validar colunas
    assert "Claim" in df.columns and "tema" in df.columns, "O dataset deve conter 'Claim' e 'tema'"
    df = df.dropna(subset=["Claim", "tema"]).copy()
    df["Claim"] = df["Claim"].astype(str).str.strip()
    df["tema"] = df["tema"].astype(str).str.strip()

    # Mapeamento de rótulos
    unique_labels = sorted(df["tema"].unique().tolist())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    num_labels = len(unique_labels)
    logger.info(f"Classes ({num_labels}): {unique_labels}")

    df["label_id"] = df["tema"].map(label2id)

    # Salvar mapeamento
    with open(output_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)
    with open(output_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2, ensure_ascii=False)

    # 2. Divisão estratificada (80% train, 10% val, 10% test)
    logger.info("Realizando divisão estratificada: 80% treino, 10% validação, 10% teste...")
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=args.seed,
        stratify=df["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=temp_df["label_id"],
    )

    logger.info(f"Divisão: Treino={len(train_df)}, Validação={len(val_df)}, Teste={len(test_df)}")

    # 3. Inicializar Tokenizer e Dataloaders
    tok_source = args.model_name
    if Path(tok_source).is_dir() and not (Path(tok_source) / "tokenizer.json").exists():
        if (output_dir / "tokenizer" / "tokenizer.json").exists():
            tok_source = str(output_dir / "tokenizer")
        else:
            tok_source = "distilbert-base-multilingual-cased"

    logger.info(f"Carregando tokenizer de '{tok_source}'...")
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    train_dataset = TextClaimDataset(
        train_df["Claim"].tolist(), train_df["label_id"].tolist(), tokenizer, max_length=args.max_len
    )
    val_dataset = TextClaimDataset(
        val_df["Claim"].tolist(), val_df["label_id"].tolist(), tokenizer, max_length=args.max_len
    )
    test_dataset = TextClaimDataset(
        test_df["Claim"].tolist(), test_df["label_id"].tolist(), tokenizer, max_length=args.max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Inicializar Modelo PyTorch
    device = get_device()
    logger.info(f"Instanciando modelo '{args.model_name}' com {num_labels} classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )

    # 5. Loop de Treinamento
    best_val_f1 = -1.0
    best_pytorch_dir = output_dir / "pytorch_distilbert"

    logger.info("=" * 60)
    logger.info("INICIANDO TREINAMENTO PYTORCH")
    logger.info("=" * 60)

    start_train_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_preds, val_targets = evaluate_pytorch(model, val_loader, device)

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        epoch_sec = time.time() - t_epoch_start

        logger.info(
            f"Época {epoch}/{args.epochs} [{epoch_sec:.1f}s] - "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            logger.info(f"  --> Novo melhor modelo! Salvando checkpoint em {best_pytorch_dir}...")
            model.save_pretrained(best_pytorch_dir)

    total_train_sec = time.time() - start_train_time
    logger.info(f"Treinamento concluído em {total_train_sec:.1f}s. Melhor Val Macro-F1: {best_val_f1:.4f}")

    # 6. Carregar melhor modelo para avaliação do conjunto de teste
    logger.info("Carregando o melhor checkpoint para avaliação no conjunto de teste...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_pytorch_dir)
    best_model.to(device)

    _, test_preds_pt, test_targets_pt = evaluate_pytorch(best_model, test_loader, device)
    pt_acc = accuracy_score(test_targets_pt, test_preds_pt)
    pt_macro_f1 = f1_score(test_targets_pt, test_preds_pt, average="macro", zero_division=0)
    pt_weighted_f1 = f1_score(test_targets_pt, test_preds_pt, average="weighted", zero_division=0)

    logger.info("-" * 60)
    logger.info(f"PyTorch Test Acc: {pt_acc:.4f} | Macro-F1: {pt_macro_f1:.4f} | Weighted-F1: {pt_weighted_f1:.4f}")
    logger.info("-" * 60)

    # 7. Exportar para ONNX (FP32)
    logger.info("=" * 60)
    logger.info("EXPORTANDO PARA ONNX (FP32)")
    logger.info("=" * 60)

    best_model.eval()
    best_model.to("cpu")  # Exportar em CPU para consistência dos tensores

    fp32_onnx_path = output_dir / "classifier_fp32.onnx"
    dummy_text = "Notícia de exemplo para rastreamento de tensores de entrada do classificador."
    dummy_inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=args.max_len,
    )

    t0_export = time.time()
    torch.onnx.export(
        best_model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        str(fp32_onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    logger.info(f"Modelo ONNX FP32 exportado em {time.time() - t0_export:.2f}s: {fp32_onnx_path}")

    # Verificar modelo ONNX
    onnx_proto = onnx.load(str(fp32_onnx_path))
    onnx.checker.check_model(onnx_proto)
    logger.info("Verificação ONNX FP32 bem-sucedida (check_model OK).")

    # 8. Quantização Dinâmica para INT8 com ONNX Runtime
    logger.info("=" * 60)
    logger.info("QUANTIZANDO COM ONNX RUNTIME INT8 (Dynamic Quantization)")
    logger.info("=" * 60)

    int8_onnx_path = output_dir / "classifier_int8.onnx"
    t0_quant = time.time()

    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Quantização INT8 concluída em {time.time() - t0_quant:.2f}s: {int8_onnx_path}")

    # Verificar modelo INT8
    onnx_int8_proto = onnx.load(str(int8_onnx_path))
    onnx.checker.check_model(onnx_int8_proto)
    logger.info("Verificação ONNX INT8 bem-sucedida (check_model OK).")

    # 9. Benchmarking ONNX FP32 vs ONNX INT8
    logger.info("=" * 60)
    logger.info("BENCHMARKING NO CONJUNTO DE TESTE (ONNX FP32 vs ONNX INT8)")
    logger.info("=" * 60)

    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    int8_size_mb = os.path.getsize(int8_path := int8_onnx_path) / (1024 * 1024)
    compression_ratio = (1.0 - (int8_size_mb / fp32_size_mb)) * 100.0

    test_texts = test_df["Claim"].tolist()
    test_labels = test_df["label_id"].tolist()

    # Sessões ONNX Runtime em CPU
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    logger.info("Avaliando ONNX FP32 no conjunto de teste...")
    sess_fp32 = ort.InferenceSession(str(fp32_onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    fp32_results = evaluate_onnx_session(sess_fp32, test_texts, test_labels, tokenizer, args.max_len)

    logger.info("Avaliando ONNX INT8 no conjunto de teste...")
    sess_int8 = ort.InferenceSession(str(int8_onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    int8_results = evaluate_onnx_session(sess_int8, test_texts, test_labels, tokenizer, args.max_len)

    speedup = fp32_results["mean_latency_ms"] / int8_results["mean_latency_ms"]

    # Relatório de Classificação por Classe no modelo INT8
    target_names = [id2label[i] for i in range(num_labels)]
    int8_clf_report = classification_report(
        test_labels,
        int8_results["predictions"],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    int8_conf_matrix = confusion_matrix(test_labels, int8_results["predictions"]).tolist()

    # Montar relatório consolidado
    benchmark_report = {
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "classes": unique_labels,
        },
        "model_comparison": {
            "fp32": {
                "size_mb": round(fp32_size_mb, 2),
                "accuracy": round(fp32_results["accuracy"], 4),
                "macro_f1": round(fp32_results["macro_f1"], 4),
                "weighted_f1": round(fp32_results["weighted_f1"], 4),
                "mean_latency_ms": round(fp32_results["mean_latency_ms"], 2),
                "median_latency_ms": round(fp32_results["median_latency_ms"], 2),
                "p95_latency_ms": round(fp32_results["p95_latency_ms"], 2),
                "throughput_samples_sec": round(fp32_results["throughput_samples_sec"], 2),
            },
            "int8": {
                "size_mb": round(int8_size_mb, 2),
                "accuracy": round(int8_results["accuracy"], 4),
                "macro_f1": round(int8_results["macro_f1"], 4),
                "weighted_f1": round(int8_results["weighted_f1"], 4),
                "mean_latency_ms": round(int8_results["mean_latency_ms"], 2),
                "median_latency_ms": round(int8_results["median_latency_ms"], 2),
                "p95_latency_ms": round(int8_results["p95_latency_ms"], 2),
                "throughput_samples_sec": round(int8_results["throughput_samples_sec"], 2),
            },
            "gains": {
                "size_reduction_pct": round(compression_ratio, 2),
                "latency_speedup_x": round(speedup, 2),
                "accuracy_difference": round(int8_results["accuracy"] - fp32_results["accuracy"], 4),
                "macro_f1_difference": round(int8_results["macro_f1"] - fp32_results["macro_f1"], 4),
            },
        },
        "classification_report_int8": int8_clf_report,
        "confusion_matrix_int8": int8_conf_matrix,
    }

    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Relatório de benchmarking salvo em {report_path}")

    # Exibição Amigável dos Resultados
    print("\n" + "=" * 70)
    print("                 RESULTADOS FINAIS DO BENCHMARK")
    print("=" * 70)
    print(f"{'Métrica / Propriedade':<25} | {'ONNX FP32':<18} | {'ONNX INT8':<18} | {'Diferença / Ganho':<15}")
    print("-" * 70)
    print(f"{'Tamanho em Disco':<25} | {fp32_size_mb:>10.2f} MB     | {int8_size_mb:>10.2f} MB     | -{compression_ratio:>5.1f}%")
    print(f"{'Acurácia no Teste':<25} | {fp32_results['accuracy']:>14.4f}     | {int8_results['accuracy']:>14.4f}     | {int8_results['accuracy'] - fp32_results['accuracy']:>+6.4f}")
    print(f"{'Macro F1-Score':<25} | {fp32_results['macro_f1']:>14.4f}     | {int8_results['macro_f1']:>14.4f}     | {int8_results['macro_f1'] - fp32_results['macro_f1']:>+6.4f}")
    print(f"{'Weighted F1-Score':<25} | {fp32_results['weighted_f1']:>14.4f}     | {int8_results['weighted_f1']:>14.4f}     | {int8_results['weighted_f1'] - fp32_results['weighted_f1']:>+6.4f}")
    print(f"{'Latência Média (CPU)':<25} | {fp32_results['mean_latency_ms']:>12.2f} ms     | {int8_results['mean_latency_ms']:>12.2f} ms     | {speedup:>5.2f}x mais rápido")
    print(f"{'Throughput':<25} | {fp32_results['throughput_samples_sec']:>10.1f} s/sec  | {int8_results['throughput_samples_sec']:>10.1f} s/sec  | +{(speedup - 1)*100:>5.1f}%")
    print("=" * 70)

    print("\nDesempenho por Tema no ONNX INT8:")
    print("-" * 70)
    print(f"{'Tema':<22} | {'Precisão':<10} | {'Revocação':<10} | {'F1-Score':<10} | {'Suporte':<8}")
    print("-" * 70)
    for theme in unique_labels:
        stats = int8_clf_report[theme]
        print(f"{theme:<22} | {stats['precision']:>8.4f}   | {stats['recall']:>8.4f}   | {stats['f1-score']:>8.4f}   | {int(stats['support']):>7}")
    print("-" * 70)
    print(f"{'Acurácia Global':<22} | {'-':<10} | {'-':<10} | {int8_results['accuracy']:>8.4f}   | {len(test_labels):>7}")
    print("=" * 70)


if __name__ == "__main__":
    main()

