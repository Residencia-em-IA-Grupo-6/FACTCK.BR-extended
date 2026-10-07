#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_theme.py

Mecanismo de inferência ultra-rápido para classificação temática de claims e notícias
usando o modelo quantizado em INT8 com o ONNX Runtime.

Uso:
  python scripts/predict_theme.py "Texto da alegação ou notícia"
  python scripts/predict_theme.py --interactive
  python scripts/predict_theme.py --input-file claims.tsv --output-file predicoes.tsv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class ThemePredictor:
    def __init__(self, model_dir: str = "models", prefer_int8: bool = True):
        self.model_dir = Path(model_dir)
        int8_path = self.model_dir / "classifier_int8.onnx"
        fp32_path = self.model_dir / "classifier_fp32.onnx"

        if prefer_int8 and int8_path.exists():
            self.model_path = int8_path
            self.model_type = "ONNX INT8"
        elif fp32_path.exists():
            self.model_path = fp32_path
            self.model_type = "ONNX FP32"
        else:
            raise FileNotFoundError(
                f"Nenhum modelo ONNX encontrado em {self.model_dir}. Execute o script train_onnx_classifier.py primeiro."
            )

        tokenizer_dir = self.model_dir / "tokenizer"
        if not tokenizer_dir.exists():
            # Fallback para o nome base
            tokenizer_dir = "distilbert-base-multilingual-cased"

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

        id2label_path = self.model_dir / "id2label.json"
        if not id2label_path.exists():
            raise FileNotFoundError(f"Arquivo id2label.json não encontrado em {self.model_dir}.")

        with open(id2label_path, "r", encoding="utf-8") as f:
            raw_id2label = json.load(f)
            self.id2label = {int(k): v for k, v in raw_id2label.items()}

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_opts, providers=["CPUExecutionProvider"])

    def predict(self, text: str, max_length: int = 128) -> Dict:
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        outputs = self.session.run(None, ort_inputs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        logits = outputs[0][0]
        probs = softmax(logits)

        best_id = int(np.argmax(probs))
        best_theme = self.id2label[best_id]
        best_score = float(probs[best_id])

        prob_dict = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
        # Ordenar por maior probabilidade
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        return {
            "claim": text,
            "theme": best_theme,
            "confidence": best_score,
            "probabilities": sorted_probs,
            "latency_ms": latency_ms,
            "engine": self.model_type,
        }

    def predict_batch(self, texts: List[str], max_length: int = 128) -> List[Dict]:
        return [self.predict(t, max_length=max_length) for t in texts]


def format_bar(prob: float, length: int = 20) -> str:
    filled = int(round(prob * length))
    return "█" * filled + "░" * (length - filled)


def main():
    parser = argparse.ArgumentParser(description="Classificação Temática com ONNX Runtime INT8")
    parser.add_argument("text", nargs="?", default=None, help="Texto da claim ou notícia a ser classificada")
    parser.add_argument("--interactive", action="store_true", help="Inicia modo interativo no terminal")
    parser.add_argument("--input-file", type=str, default=None, help="Arquivo TSV/CSV contendo claims a classificar")
    parser.add_argument("--output-file", type=str, default=None, help="Caminho de saída para os resultados da predição")
    parser.add_argument("--model-dir", type=str, default="models", help="Diretório onde o modelo ONNX reside")
    parser.add_argument("--fp32", action="store_true", help="Força o uso do modelo FP32 em vez do INT8")
    args = parser.parse_args()

    predictor = ThemePredictor(model_dir=args.model_dir, prefer_int8=not args.fp32)

    if args.input_file:
        in_path = Path(args.input_file)
        sep = "\t" if in_path.suffix == ".tsv" else ","
        df = pd.read_csv(in_path, sep=sep)
        col = "Claim" if "Claim" in df.columns else df.columns[0]
        print(f"Classificando {len(df)} registros do arquivo {in_path} (coluna: '{col}')...")

        t0 = time.time()
        results = predictor.predict_batch(df[col].astype(str).tolist())
        dt = time.time() - t0

        df["tema_predito"] = [r["theme"] for r in results]
        df["confianca"] = [r["confidence"] for r in results]

        out_path = args.output_file or in_path.with_name(f"{in_path.stem}_predito{in_path.suffix}")
        out_sep = "\t" if str(out_path).endswith(".tsv") else ","
        df.to_csv(out_path, sep=out_sep, index=False)
        print(f"Predições salvas em {out_path} ({len(df)} linhas em {dt:.2f}s, {len(df)/dt:.1f} amostras/s).")
        return

    if args.interactive:
        print("=" * 65)
        print(f"  Classificador Temático de Notícias ({predictor.model_type})")
        print("  Digite 'sair' ou 'exit' para encerrar.")
        print("=" * 65)
        while True:
            try:
                user_text = input("\nTexto da Claim > ").strip()
                if not user_text or user_text.lower() in ["sair", "exit", "quit"]:
                    break
                res = predictor.predict(user_text)
                print(f"\n👉 Tema Predito: \033[1;32m{res['theme'].upper()}\033[0m (Confiança: {res['confidence']*100:.1f}%)")
                print(f"⏱️  Tempo de resposta: {res['latency_ms']:.2f} ms")
                print("Probabilidades:")
                for theme, prob in res["probabilities"].items():
                    bar = format_bar(prob, 20)
                    print(f"  {theme:<18} [{bar}] {prob*100:>5.1f}%")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nEncerrado.")
        return

    if args.text:
        res = predictor.predict(args.text)
        print("\n" + "=" * 65)
        print(f"Claim: \"{res['claim']}\"")
        print(f"Engine: {res['engine']}")
        print(f"Tema Predito: {res['theme'].upper()} (Confiança: {res['confidence']*100:.1f}%)")
        print(f"Latência de Inferência: {res['latency_ms']:.2f} ms")
        print("-" * 65)
        print("Distribuição de Probabilidades:")
        for theme, prob in res["probabilities"].items():
            bar = format_bar(prob, 20)
            print(f"  {theme:<18} [{bar}] {prob*100:>5.1f}%")
        print("=" * 65)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

