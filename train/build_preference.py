# -*- coding: utf-8 -*-
"""
build_preference.py —— 自构 DPO 偏好数据（方案 A）
====================================================
从 dpo_source.jsonl（SFT 不可见的独立集合）构造 chosen/rejected 对：

  chosen   = 该样本的标准答案（output 字段，权威标注）
  rejected = **Stage 2 模型**对同一问题采样的回答（on-policy，同一"会答"分布）

设计依据（DPO 语义）：偏好对应编码"同一任务分布下的质量偏好"，而非
"会不会答"。因此 rejected 不用未微调的基座（胡编/不会答，偏离 DPO 本意），
而用 stage2 模型采样——chosen（标准答案，精炼准确）与 stage2 采样输出
（同分布但质量较低）的差异即质量维度偏好。区分度通过 temperature 采样保证。

输出 trl DPOTrainer 格式：
  {"prompt": [{"role":"user","content": input}],
   "chosen": [{"role":"assistant","content": output}],
   "rejected":[{"role":"assistant","content": <stage2 采样>}]}

质量控制：
- rejected 用 stage2 模型 + 较高温度采样
- 过滤 rejected 与 chosen 相同 / 输出为空的样本（无区分度）

引擎选择（--engine）：
  hf     —— HF transformers + 手动 batching（Windows / 默认，2-4x 提速）
  vllm   —— vLLM 离线推理（仅 Linux/WSL2，5-10x 提速；7B 25k 对 <1h）

用法（HF 引擎，本地 0.5B 冒烟）：
  python train/build_preference.py --engine hf \
    --model_path D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct \
    --stage2_dir ckpt/smoke_stage2 --data_dir data_prep/output --output_dir data_prep/output \
    --max_pairs 50 --max_new_tokens 96 --temperature 0.9 --use_4bit

用法（vLLM 引擎，Linux/WSL2）：
  python train/build_preference.py --engine vllm \
    --model_path /path/to/Qwen2.5-7B-Instruct \
    --stage2_dir ckpt/stage2_s1 --data_dir data_prep/output --output_dir data_prep/output \
    --max_pairs 25000 --max_new_tokens 96 --temperature 0.9 --vllm_gpu_mem 0.85
"""
import argparse
import json
import sys
import os
import time

import sklearn
import torch
from peft import PeftModel

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import SYSTEM_PROMPT, add_common_args, load_model_tokenizer, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    # build_preference 覆藇 common 的 --output_dir required 默认 = --data_dir
    # （让 stage3 能在同目录下找到 dpo_pairs.jsonl，避免忘填跳坑）
    parser.set_defaults(output_dir=None)
    parser.add_argument("--stage2_dir", type=str, required=True,
                        help="Stage 2 模型 adapter（生成 rejected 用，on-policy）")
    # output_dir 不再用 common 的 required，改用下面的默认（=data_dir），避免忘填
    parser.add_argument("--max_pairs", type=int, default=25000, help="构造偏好对数量")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--in_file", type=str, default="dpo_source.jsonl")
    parser.add_argument("--out_file", type=str, default="dpo_pairs.jsonl")
    # ────── 引擎选择 + 引擎专属参数 ──────
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf",
                        help="推理引擎：hf (Windows 默认，2-4x) / vllm (Linux，5-10x)")
    parser.add_argument("--hf_batch_size", type=int, default=16,
                        help="HF 引擎 batching 大小（4090/4060 4bit 默认 16；OOM 降到 8）")
    parser.add_argument("--vllm_gpu_mem", type=float, default=0.85,
                        help="vLLM gpu_memory_utilization（4060 8G 用 0.85；OOM 降到 0.7）")
    parser.add_argument("--vllm_max_model_len", type=int, default=1024,
                        help="vLLM max_model_len（短 prompt 用 1024 省 KV cache）")
    parser.add_argument("--vllm_dtype", default="auto",
                        choices=["auto", "bfloat16", "float16"],
                        help="vLLM 模型 dtype；4060 bf16 不支持会 fallback fp16")
    args = parser.parse_args()
    # 默认 = data_dir（与 stage3_dpo --data_dir 一致，能在同目录找到产出）
    if args.output_dir is None:
        args.output_dir = args.data_dir
        print(f"[Pref] --output_dir 未传，默认 = --data_dir = {args.output_dir}")
    sys.stdout.reconfigure(encoding="utf-8")
    # 【防误用】此脚本是生成式推理，不要跑 torchrun 多卡：DDP 会让每个 rank 都加载全量
    # policy+ref 双 7B 模型，立刻 OOM。这里 hard assert 抓装误调者。
    # （vLLM 多卡请传 tensor_parallel_size，不要用 torchrun——vLLM 自己管进程组。）
    if "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "[Pref] build_preference 不支持 torchrun 多卡（生成式脚本，不需 DDP）。"
            "\n  请用 `python train/build_preference.py ...` 直接跑。"
            "\n  如需加速，请改用多卡并行推理框架（vLLM 等）而不是 DDP。"
        )
    IS_MAIN = int(os.environ.get("LOCAL_RANK", "0")) == 0

    # ─────────────── 引擎选择：HF vs vLLM ───────────────
    # 关键区别：
    #   HF     —— PeftModel 加载 LoRA，model.generate() 串行 batch（手动 padding）
    #   vLLM   —— 把 stage2 LoRA adapter merge 进 vLLM 引擎，llm.generate() 一次送所有 prompt
    #
    # vLLM 在 Windows 下不能 import（vllm._C 扩展未编译），必须 Linux/WSL2。
    # Windows 选了 vllm 在加载阶段就会 ImportError 而报错，不会走到生成循环。
    if args.engine == "vllm":
        if IS_MAIN:
            print(f"[Pref] engine=vllm | base: {args.model_path} | LoRA: {args.stage2_dir} | "
                  f"gpu_mem={args.vllm_gpu_mem} max_len={args.vllm_max_model_len}")
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        # vLLM dtype：4060 是 Ada Lovelace 架构，bf16 支持但 vLLM 偶有兼容问题，默认 auto
        # 让 vLLM 自己选（通常 fp16）；显存充裕且要一致行为时传 bfloat16。
        vllm_dtype = None if args.vllm_dtype == "auto" else args.vllm_dtype
        llm = LLM(
            model=args.model_path,
            enable_lora=True,
            max_lora_rank=64,  # 覆盖常见 LoRA r=16/32/64；stage2 用 r=16
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_mem,
            dtype=vllm_dtype,
            trust_remote_code=True,
            # 重要：不要传 quantization="bitsandbytes"——4060 8G 跑 7B 4bit 用 vLLM 加载比 HF 慢且不稳；
            # 如要 4bit，在 --model_path 直接传 GPTQ/AWQ 量化模型路径，这里保持 bf16。
        )
        # 注册 LoRA：vLLM 每次 generate 用 lora_request 参数挑一个 adapter；
        # 这里只有 stage2 一个，所以建一次复用。
        lora_req = LoRARequest("stage2", 1, args.stage2_dir)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.9,
            max_tokens=args.max_new_tokens,
        )
        # vLLM 没有传统意义上的 tokenizer 对象，需要单独拿 apply_chat_template 的能力；
        # transformers tokenizer 仍可独立加载用于拼 prompt 文本。
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # 生成函数（vLLM 版本）：一次性送所有 prompt 给 vLLM，输出 RequestOutput 列表。
        def _generate_batch(prompts):
            outs = llm.generate(
                prompts,
                sampling_params,
                lora_request=lora_req,
                use_tqdm=False,  # 避免和外层进度条打架
            )
            return [o.outputs[0].text.strip() for o in outs]
    else:  # hf
        if IS_MAIN:
            print(f"[Pref] engine=hf | base: {args.model_path} + LoRA: {args.stage2_dir} | "
                  f"batch_size={args.hf_batch_size} use_4bit={args.use_4bit}")
        model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
        model = PeftModel.from_pretrained(model, args.stage2_dir)
        model.eval()
        # 用左侧 padding：generate 时 batch 内 prompt 左对齐 padding，
        # 这样每个样本的有效 token 都在右侧尾部，generation 一致；右侧 padding 会污染生成起点。
        tokenizer.padding_side = "left"
        # 生成函数（HF 版本）：手动 batch + left-padding + model.generate()
        def _generate_batch(prompts):
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            ).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # 只取 prompt 之后的新 token（按 prompt 长度切片，每个样本各自长度）
            prompt_len = inputs["input_ids"].shape[1]
            decoded = []
            for row in gen:
                new_ids = row[prompt_len:]
                decoded.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
            return decoded

    rows = read_jsonl(f"{args.data_dir}/{args.in_file}", args.max_pairs)
    if IS_MAIN:
        print(f"[Pref] 读取 {args.in_file}: {len(rows)} 条")

    # 多卡 sharding：每个 rank 处理自己的子集，避免 4 卡各生成全部 N 对的 4× 浪费
    # 和同一 out_file 被写 4 次（last-write-wins，rank 0 不一定最后写）。
    # 顺序轮转切分 [rank::world]；rank 0 写主名，其他 rank 写 ".rank{i}" 后缀，循环结束后 rank 0 合并。
    import torch.distributed as dist
    rank, world = 0, 1
    if dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
        rows = rows[rank::world]
        if IS_MAIN:
            print(f"[Pref] rank sharding: world={world}, 本 rank {rank} 处理 {len(rows)} 条")

    # 统一走 .rank{rank} 后缀：单卡 world=1 也写 .rank0（下面 atomic publish 逻辑统一处理）
    rank_suffix = f".rank{rank}"
    out_path = f"{args.output_dir}/{args.out_file}{rank_suffix}"

    # 清理上一轮残留的 .rank* 文件，防止 merge 读到过期内容
    import glob as _glob_cleanup
    import os as _os_cleanup
    # 所有 rank 先各自清自己的 .rank{rank}（以防上次以同样 rank 写了一半）
    my_old = f"{args.output_dir}/{args.out_file}.rank{rank}"
    if _os_cleanup.path.exists(my_old):
        _os_cleanup.remove(my_old)
        if IS_MAIN:
            print(f"[Pref] rank {rank} 清理残留 {my_old}")
    # rank 0 额外清任何其他 .rank* 残留（防止上轮不同 world 没清干净）
    if rank == 0:
        for old in _glob_cleanup.glob(f"{args.output_dir}/{args.out_file}.rank*"):
            if old != my_old:
                _os_cleanup.remove(old)
                print(f"[Pref] rank 0 清理其他残留 {old}")

    written, skipped = 0, 0
    # ─────────────── 主循环：按 batch 切片送 _generate_batch ───────────────
    # HF 路径：batch_size 默认 16（4060 4bit 安全，OOM 降到 8）；
    # vLLM 路径：batch_size 设大一点（比如 64），因为 vLLM 内部还会再 micro-batch。
    bs = args.hf_batch_size if args.engine == "hf" else max(args.hf_batch_size * 4, 64)
    t0 = time.time()
    last_log_t = t0
    with open(out_path, "w", encoding="utf-8") as f:
        i = 0
        while i < len(rows):
            batch_rows = rows[i:i + bs]
            # 构造 prompt 文本（chat_template）
            prompts = []
            for r in batch_rows:
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["input"]},
                ]
                prompts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
            rejected_list = _generate_batch(prompts)
            # 逐条写盘（vLLM 一次返回全部，HF 一次返回 bs 个，按 bs 切就够）
            for r, rejected in zip(batch_rows, rejected_list):
                chosen = r["output"]
                # 过滤无区分度样本（rejected 与 chosen 相同 / 空输出）
                if not rejected or rejected == chosen:
                    skipped += 1
                    continue
                f.write(json.dumps({
                    "prompt": [{"role": "user", "content": r["input"]}],
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                }, ensure_ascii=False) + "\n")
                written += 1
            # 进度日志：每处理 1 个 batch 打一条，含速率 + ETA（每 30s 至少打一条防"沉默"误判）
            i += len(batch_rows)
            now = time.time()
            if IS_MAIN and ((i % max(bs, 50) == 0) or (now - last_log_t > 30)):
                elapsed = now - t0
                speed = i / elapsed if elapsed > 0 else 0
                eta = (len(rows) - i) / speed if speed > 0 else float("inf")
                print(f"  [进度] {i}/{len(rows)} | 有效 {written} | 跳过 {skipped} | "
                      f"{speed:.1f} it/s | 已用 {elapsed/60:.1f}min | "
                      f"ETA {eta/60:.1f}min",
                      flush=True)
                last_log_t = now

        # fsync 强制落盘：f.close()（with 退出）只 flush Python buffer → OS page cache
        # os.fsync() 才走 OS → 物理磁盘；保证随后 barrier 后 rank 0 能读到完整文件
        # （避免 OS page cache / NFS 延迟造成读不到——屏障同步的是状态，不是磁盘落定）
        try:
            f.flush()
            os.fsync(f.fileno())
        except (AttributeError, OSError) as _e:
            # 某些 FS（NFS 等）不支持 fsync 或 fd 已失效，警告但不中断
            # —— 即使 fsync 失败，下面 barrier 仍保证跨 rank 同步
            if IS_MAIN:
                print(f"[Pref] rank {rank} fsync 失败（跳过 fsync，但 barrier 仍在此后）: {_e}")

    if IS_MAIN:
        print(f"[Pref] rank {rank} 完成: 写入 {written} 对 -> {out_path}（跳过 {skipped}）")

    # 不论单卡/多卡，都走原子 publishes 流程：写 .tmp → fsync → os.replace() → 验证
    # ——多卡这里是 merge .rank*（多卡内 barrier 保证齐），单卡是 move .rank0 到主名
    import glob as _glob_final
    import os as _os_final
    import json as _json_final

    if world > 1:
        from torch.distributed import barrier as _barrier
        _barrier()

    if rank == 0:
        final_path = f"{args.output_dir}/{args.out_file}"
        tmp_path = final_path + ".tmp"
        # 多卡：merge .rank*；单卡：rank 0 就是 .rank0
        rank_files = sorted(_glob_final.glob(f"{args.output_dir}/{args.out_file}.rank*"))
        if len(rank_files) != world:
            print(f"[Pref] 警告：期望 {world} 个 .rank* 文件，实际 {len(rank_files)} 个：{[os.path.basename(p) for p in rank_files]}")
        if not rank_files:
            # 【二补】merger 之前在 barrier 里错过问题，trace 让使用者能区分"被 elastic SIGTERM
            # 前中杀死"还是"本身没写入"。列出同目录内容帮使用者诊断路径/cwd 问题。
            siblings = sorted(_os_final.listdir(args.output_dir)) if _os_final.path.isdir(args.output_dir) else []
            raise RuntimeError(
                f"[Pref] 未找到 .rank* 文件（{args.output_dir}/{args.out_file}.rank*），请检查 build_preference 是否成功写入。"
                f"\n  当前 output_dir 内容: {siblings[:20]}"
                f"\n  提示：多卡 DDP 跑下如果其他 rank 由于 OOM 或其他异常退出"
                f"（只 rank 0 写完），elastic launcher 默认 60s 超时后会 SIGTERM 主进程；"
                f"你可以重新跑一次，或用单卡 world=1 跑（保证 4 个 .rank* 都到位）"
            )
        # 原子写：先写 .tmp 临时文件，fsync 后原子 rename 到主名
        # （不论单卡还是多卡，都走这步——避免任何场景下产出半合并文件）
        with open(tmp_path, "w", encoding="utf-8") as fout:
            total_lines = 0
            for rp in rank_files:
                with open(rp, encoding="utf-8") as fin:
                    chunk = fin.read()
                    fout.write(chunk)
                    total_lines += chunk.count("\n")
                _os_final.remove(rp)
            fout.flush()
            _os_final.fsync(fout.fileno())
        _os_final.replace(tmp_path, final_path)  # atomic rename
        # 验证：合并后文件每行应能被 json.loads 解析
        ok = bad = 0
        with open(final_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    _json_final.loads(line); ok += 1
                except Exception:
                    bad += 1
        print(f"[Pref] {'merged' if world > 1 else 'published'} {world} rank file(s) -> {final_path}（valid={ok}, invalid={bad}, lines={total_lines}）")
        if bad > 0:
            raise RuntimeError(f"[Pref] {'merge' if world > 1 else 'publish'} 后 dpo_pairs.jsonl 有 {bad} 条损坏 JSON——可能上次 race 残留未被清理干净，请检查 .rank* 文件是否已被其他进程占用")


if __name__ == "__main__":
    main()
