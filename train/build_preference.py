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

用法（本地 0.5B 冒烟）：
  python train/build_preference.py \
    --model_path D:/data/programe/AI/LM/Qwen2.5-0.5B-Instruct \
    --stage2_dir ckpt/smoke_stage2 --data_dir data_prep/output --output_dir data_prep/output \
    --max_pairs 50 --max_new_tokens 96 --temperature 0.9 --use_4bit
"""
import argparse
import json
import sys
import os

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
    args = parser.parse_args()
    # 默认 = data_dir（与 stage3_dpo --data_dir 一致，能在同目录找到产出）
    if args.output_dir is None:
        args.output_dir = args.data_dir
        print(f"[Pref] --output_dir 未传，默认 = --data_dir = {args.output_dir}")
    sys.stdout.reconfigure(encoding="utf-8")
    IS_MAIN = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if IS_MAIN:
        print(f"[Pref] 加载 base: {args.model_path} + stage2 adapter: {args.stage2_dir}（生成 rejected 用）")
    model, tokenizer = load_model_tokenizer(args.model_path, args.use_4bit, args.max_len)
    model = PeftModel.from_pretrained(model, args.stage2_dir)
    model.eval()

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
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            # 构造 prompt（system + user，generate 时不加 assistant 前缀，让模型直接续写）
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["input"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                               max_length=args.max_len).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = gen[0][inputs["input_ids"].shape[1]:]
            rejected = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

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
            if (i + 1) % 50 == 0:
                if IS_MAIN:
                    print(f"  已处理 {i+1}/{len(rows)}，有效 {written}，跳过 {skipped}")

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
