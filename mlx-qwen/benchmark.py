"""
MLX 模型效率评测
指标：吞吐量(TPS)、首Token延迟(TTFT)、每Token延迟(TPOT)、峰值内存
"""

import os
import time
import json
import statistics
from dataclasses import dataclass, asdict
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_ID.replace("/", "--"))

BENCH_PROMPTS = [
    ("短文本", "请用一句话解释什么是人工智能。"),
    ("中文本", "请详细介绍Python语言的优缺点，包括语法特点、应用领域和生态系统。"),
    ("长文本", "请写一篇500字左右的短文，主题为：人工智能在医疗领域的应用与挑战。"),
    ("代码", "请用Python实现一个快速排序算法，并添加详细注释。"),
    ("推理", "小明有5个苹果，给了小红2个，又从小华那里得到3个，请问小明现在有几个苹果？请逐步推理。"),
]


@dataclass
class BenchResult:
    prompt_name: str
    prompt_tokens: int
    generation_tokens: int
    ttft_ms: float           # 首 Token 延迟
    tpot_ms: float           # 每 Token 平均延迟
    tps: float               # 生成吞吐量
    prompt_tps: float        # Prefill 吞吐量
    total_time_ms: float     # 总耗时
    peak_memory_gb: float    # 峰值内存 (GB)


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model {MODEL_ID} to {MODEL_PATH} ...")
        snapshot_download(MODEL_ID, local_dir=MODEL_PATH)
    print(f"Loading model from {MODEL_PATH} ...")
    model, tokenizer = load(MODEL_PATH)
    print("Model loaded.\n")
    return model, tokenizer


def benchmark_single(model, tokenizer, prompt_text, max_tokens=512):
    messages = [
        {"role": "system", "content": "你是一个智能助手。"},
        {"role": "user", "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    sampler = make_sampler(temp=0.0)  # 贪心解码，排除采样随机性

    ttft = None
    token_times = []
    prompt_tokens = 0
    gen_tokens = 0
    peak_mem = 0.0
    prompt_tps = 0.0

    t_start = time.perf_counter()

    for chunk in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
        if ttft is None:
            ttft = time.perf_counter() - t_start
        token_times.append(time.perf_counter())
        prompt_tokens = chunk.prompt_tokens
        gen_tokens = chunk.generation_tokens
        prompt_tps = chunk.prompt_tps
        peak_mem = chunk.peak_memory

    total_time = time.perf_counter() - t_start

    # TPOT: 生成 tokens 之间的平均间隔
    if len(token_times) >= 2:
        intervals = [token_times[i+1] - token_times[i] for i in range(len(token_times)-1)]
        tpot = statistics.mean(intervals) * 1000  # ms
    else:
        tpot = 0.0

    tps = gen_tokens / total_time if total_time > 0 else 0.0

    return BenchResult(
        prompt_name="",
        prompt_tokens=prompt_tokens,
        generation_tokens=gen_tokens,
        ttft_ms=ttft * 1000 if ttft else 0.0,
        tpot_ms=tpot,
        tps=tps,
        prompt_tps=prompt_tps,
        total_time_ms=total_time * 1000,
        peak_memory_gb=peak_mem,
    )


def run_benchmark(warmup=True, repeat=3):
    model, tokenizer = load_model()

    # Warmup
    if warmup:
        print("🔥 预热中 ...")
        for chunk in stream_generate(model, tokenizer, "你好", max_tokens=16, sampler=make_sampler(temp=0.0)):
            pass
        print("预热完成.\n")

    results = []
    for name, prompt in BENCH_PROMPTS:
        run_results = []
        for i in range(repeat):
            r = benchmark_single(model, tokenizer, prompt)
            r.prompt_name = name
            run_results.append(r)

        # 取各指标平均值
        avg = BenchResult(
            prompt_name=name,
            prompt_tokens=round(statistics.mean(r.prompt_tokens for r in run_results)),
            generation_tokens=round(statistics.mean(r.generation_tokens for r in run_results)),
            ttft_ms=round(statistics.mean(r.ttft_ms for r in run_results), 1),
            tpot_ms=round(statistics.mean(r.tpot_ms for r in run_results), 1),
            tps=round(statistics.mean(r.tps for r in run_results), 1),
            prompt_tps=round(statistics.mean(r.prompt_tps for r in run_results), 1),
            total_time_ms=round(statistics.mean(r.total_time_ms for r in run_results), 1),
            peak_memory_gb=round(statistics.mean(r.peak_memory_gb for r in run_results), 2),
        )
        results.append(avg)

    return results


def print_report(results):
    print("=" * 90)
    print(f"  模型效率评测报告 — {MODEL_ID}")
    print("=" * 90)
    print()
    print(f"{'场景':<8} {'Prompt':>6} {'Gen':>6} {'TTFT':>8} {'TPOT':>8} {'TPS':>8} {'Prefill':>10} {'总耗时':>8} {'峰值内存':>8}")
    print(f"{'':8} {'tokens':>6} {'tokens':>6} {'(ms)':>8} {'(ms)':>8} {'(tok/s)':>8} {'(tok/s)':>10} {'(ms)':>8} {'(GB)':>8}")
    print("-" * 90)

    for r in results:
        print(
            f"{r.prompt_name:<8} {r.prompt_tokens:>6} {r.generation_tokens:>6} "
            f"{r.ttft_ms:>8.1f} {r.tpot_ms:>8.1f} {r.tps:>8.1f} "
            f"{r.prompt_tps:>10.1f} {r.total_time_ms:>8.1f} {r.peak_memory_gb:>8.2f}"
        )

    print("-" * 90)

    # 汇总
    avg_tps = statistics.mean(r.tps for r in results)
    avg_ttft = statistics.mean(r.ttft_ms for r in results)
    avg_tpot = statistics.mean(r.tpot_ms for r in results)
    avg_mem = statistics.mean(r.peak_memory_gb for r in results)
    max_mem = max(r.peak_memory_gb for r in results)

    print(f"{'平均':<8} {'':>6} {'':>6} {avg_ttft:>8.1f} {avg_tpot:>8.1f} {avg_tps:>8.1f} {'':>10} {'':>8} {avg_mem:>8.2f}")
    print()
    print("📊 评估:")
    if avg_ttft < 200:
        print(f"  ✅ TTFT {avg_ttft:.1f}ms — 首 Token 响应迅速，交互体验流畅")
    elif avg_ttft < 500:
        print(f"  ⚠️  TTFT {avg_ttft:.1f}ms — 首 Token 响应尚可，交互有轻微等待感")
    else:
        print(f"  ❌ TTFT {avg_ttft:.1f}ms — 首 Token 延迟较高，交互体验有卡顿感")

    if avg_tpot < 250:
        print(f"  ✅ TPOT {avg_tpot:.1f}ms — 逐 Token 生成流畅")
    else:
        print(f"  ⚠️  TPOT {avg_tpot:.1f}ms — 逐 Token 生成有间隔感")

    print(f"  📈 生成吞吐量: {avg_tps:.1f} tokens/s")
    print(f"  💾 峰值内存: {max_mem:.2f} GB")
    print()


def save_json(results, path="benchmark_result.json"):
    data = {
        "model": MODEL_ID,
        "results": [asdict(r) for r in results],
        "summary": {
            "avg_tps": round(statistics.mean(r.tps for r in results), 1),
            "avg_ttft_ms": round(statistics.mean(r.ttft_ms for r in results), 1),
            "avg_tpot_ms": round(statistics.mean(r.tpot_ms for r in results), 1),
            "peak_memory_gb": round(max(r.peak_memory_gb for r in results), 2),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {path}")


if __name__ == "__main__":
    results = run_benchmark(warmup=True, repeat=3)
    print_report(results)
    save_json(results)
