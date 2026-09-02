#!/bin/bash
# =============================================================================
# BioAlign 训练 / 推理 / 评测 一体化脚本
#
# 每个训练 stage 独立启动: 该后台任务内自动串行执行 训练 -> 推理 -> 评测
#
# 用法:
#   bash run.sh smoke                           冒烟测试(前台, 小规模全步骤)
#   bash run.sh stage1                         预训练 -> 自动推理评测 ckpt/stage1
#   bash run.sh stage2                         SFT(resume stage1) -> 自动推理评测 ckpt/stage2_s1
#   bash run.sh stage2o                        SFT 独立训练 -> 自动推理评测 ckpt/stage2_only
#   bash run.sh bpref                          仅构建偏好数据(无推理评测)
#   bash run.sh stage3                         DPO -> 自动推理评测 ckpt/stage3
#   bash run.sh infer                          推理全部 4 个模型(后台, 4路并行)
#   bash run.sh eval                           评测全部 4 个模型(后台, 4路并行)
#
#   SKIP_CLONE=1 bash run.sh stage1            跳过顶部 git clone(代码已最新时更快)
#
# stage 命令终端立即返回 PID 与日志文件, 训练/推理/评测全写进同一份日志,
# 用 tail -f 查看进度; 训练失败时自动中止, 不会继续推理评测
# =============================================================================

SKIP_CLONE="${SKIP_CLONE:-0}"
if [ "$SKIP_CLONE" != "1" ]; then
  rm -rf bio-align
  git clone https://gitee.com/kip_LS/bio-align.git
fi

ADB_PATH="/"
CODE_DIR="$ADB_PATH/bio-align"
MODEL_7B="$ADB_PATH/Qwen2.5-7B-Instruct"
DATA_DIR="$ADB_PATH/input_data"
OUT_DIR="$ADB_PATH/output"

LOG_DATE=$(date +%Y%m%d_%H%M)

# 冒烟测试
if [ "$1" == "smoke" ]; then
  torchrun --nproc_per_node=4 $CODE_DIR/train/stage1_pretrain.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/ckpt/stage1_smoke \
    --max_len 1024 --max_steps 30 --max_samples 200 \
    --per_device_batch 4 --grad_accum 4

  torchrun --nproc_per_node=4 $CODE_DIR/train/stage2_sft.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    --resume_adapter $OUT_DIR/ckpt/stage1_smoke \
    --output_dir $OUT_DIR/ckpt/stage2_smoke \
    --max_len 2048 --max_steps 60 --max_samples 100 \
    --per_device_batch 4 --grad_accum 4

  python $CODE_DIR/train/build_preference.py \
    --model_path $MODEL_7B \
    --stage2_dir $OUT_DIR/ckpt/stage2_smoke \
    --data_dir $DATA_DIR \
    --output_dir $DATA_DIR --max_pairs 50

  torchrun --nproc_per_node=4 $CODE_DIR/train/stage3_dpo.py \
    --model_path $MODEL_7B \
    --stage2_dir $OUT_DIR/ckpt/stage2_smoke \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/ckpt/stage3_smoke \
    --max_len 1024 --max_samples 100 --max_steps 20 \
    --per_device_batch 4 --grad_accum 4
fi

# ---------------------------------------------------------------------------
# 各步骤训练命令(数组形式, 单步后台 与 自动推理评测 共用一份)
# ---------------------------------------------------------------------------
CMD_STAGE1=(torchrun --nproc_per_node=4 $CODE_DIR/train/stage1_pretrain.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/ckpt/stage1 \
    --max_len 1024 --epochs 1 \
    --per_device_batch 4 --grad_accum 4 \
    --lr 1e-4 --lora_plus_scaler 4)

CMD_STAGE2=(torchrun --nproc_per_node=4 $CODE_DIR/train/stage2_sft.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    --resume_adapter $OUT_DIR/ckpt/stage1 \
    --output_dir $OUT_DIR/ckpt/stage2_s1 \
    --task_prefix_ratio 0.30 --use_class_weight --task_weight_power 0.5 \
    --use_4bit --max_len 1024 --epochs 1 \
    --per_device_batch 4 --grad_accum 4 \
    --lr 2e-4)

CMD_STAGE2O=(torchrun --nproc_per_node=4 $CODE_DIR/train/stage2_sft.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/ckpt/stage2_only \
    --task_prefix_ratio 0.30 --use_class_weight --task_weight_power 0.5 \
    --use_4bit --max_len 1024 --epochs 1 \
    --per_device_batch 4 --grad_accum 4 \
    --lr 2e-4)

CMD_BPREF=(python -u $CODE_DIR/train/build_preference.py \
    --model_path $MODEL_7B --stage2_dir $OUT_DIR/ckpt/stage2_s1 \
    --data_dir $DATA_DIR \
    --output_dir $DATA_DIR \
    --max_pairs 25000 \
    --use_4bit \
    --max_new_tokens 96 --temperature 0.9)

CMD_STAGE3=(torchrun --nproc_per_node=4 $CODE_DIR/train/stage3_dpo.py \
    --model_path $MODEL_7B --stage2_dir $OUT_DIR/ckpt/stage2_s1 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/ckpt/stage3 \
    --dpo_data dpo_pairs.jsonl \
    --max_len 768 --epochs 1 \
    --per_device_batch 4 --grad_accum 4 \
    --lr 1e-5 --beta 0.1 --use_4bit)

# ---------------------------------------------------------------------------
# 推理 + 评测单个模型(串行), 供各 stage 训练完自动调用
#   用法: infer_eval_one <ckpt相对OUT_DIR的路径> <out_file> <model_name> [GPU]
#   ckpt 传空字符串表示 base 模型(不带 --ckpt_dir)
# ---------------------------------------------------------------------------
infer_eval_one() {
  local ckpt="$1" out="$2" mname="$3" gpu="${4:-0}"
  local rc=0 args=()
  if [ -n "$ckpt" ]; then
    [ -d "$OUT_DIR/$ckpt" ] || { echo "[$(date)] 警告: $OUT_DIR/$ckpt 不存在, 跳过推理评测"; return 1; }
    args+=(--ckpt_dir "$OUT_DIR/$ckpt")
  fi
  echo "[$(date)] ===== 推理 [$mname] -> $OUT_DIR/$out ====="
  CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/train/infer_eval.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    "${args[@]}" \
    --output_dir $OUT_DIR --out_file "$out" \
    --max_len 1024 --max_samples 18900 --batch_size 8 || rc=$?
  if [ $rc -ne 0 ]; then echo "[$(date)] 推理 [$mname] 失败(exit=$rc)"; return $rc; fi
  echo "[$(date)] ===== 评测 [$mname] ($OUT_DIR/$out) ====="
  CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/eval/evaluate_v2.py \
    --model_name "$mname" --OMICS all_omics \
    --input_file_path "$OUT_DIR/$out" || rc=$?
  if [ $rc -ne 0 ]; then echo "[$(date)] 评测 [$mname] 失败(exit=$rc)"; return $rc; fi
  echo "[$(date)] ===== [$mname] 推理评测完成 ====="
  return 0
}

# ---------------------------------------------------------------------------
# 训练 stage: 非 AUTO 时后台启动 AUTO 子进程(终端立即返回);
# AUTO 子进程内: 训练(前台) -> 成功则自动推理评测该模型, 失败即终止
# ---------------------------------------------------------------------------
if [ "$1" == "stage1" ]; then
  if [ "$AUTO" == "1" ]; then
    echo "[$(date)] ===== stage1 训练开始 ====="
    "${CMD_STAGE1[@]}"; rc=$?
    if [ $rc -ne 0 ]; then echo "[$(date)] stage1 训练失败(exit=$rc), 不执行推理评测"; exit $rc; fi
    echo "[$(date)] stage1 训练完成, 自动推理评测..."
    infer_eval_one ckpt/stage1 eval_stage1.jsonl stage1
    exit $?
  else
    LOGFILE=stage1_${LOG_DATE}.log
    AUTO=1 SKIP_CLONE=1 setsid bash "$0" stage1 > $LOGFILE 2>&1 < /dev/null &
    PID=$!
    disown
    echo "[$(date)] stage1 启动(训练完自动推理评测), PID=$PID, 日志=$LOGFILE"
  fi
fi

if [ "$1" == "stage2" ]; then
  if [ "$AUTO" == "1" ]; then
    echo "[$(date)] ===== stage2 训练开始 ====="
    "${CMD_STAGE2[@]}"; rc=$?
    if [ $rc -ne 0 ]; then echo "[$(date)] stage2 训练失败(exit=$rc), 不执行推理评测"; exit $rc; fi
    echo "[$(date)] stage2 训练完成, 自动推理评测..."
    infer_eval_one ckpt/stage2_s1 eval_s1_s2.jsonl s1_s2
    exit $?
  else
    LOGFILE=stage2_s1_${LOG_DATE}.log
    AUTO=1 SKIP_CLONE=1 setsid bash "$0" stage2 > $LOGFILE 2>&1 < /dev/null &
    PID=$!
    disown
    echo "[$(date)] stage2 启动(训练完自动推理评测), PID=$PID, 日志=$LOGFILE"
  fi
fi

if [ "$1" == "stage2o" ]; then
  if [ "$AUTO" == "1" ]; then
    echo "[$(date)] ===== stage2o 训练开始 ====="
    "${CMD_STAGE2O[@]}"; rc=$?
    if [ $rc -ne 0 ]; then echo "[$(date)] stage2o 训练失败(exit=$rc), 不执行推理评测"; exit $rc; fi
    echo "[$(date)] stage2o 训练完成, 自动推理评测..."
    infer_eval_one ckpt/stage2_only eval_s2_only.jsonl s2_only
    exit $?
  else
    LOGFILE=stage2_only_${LOG_DATE}.log
    AUTO=1 SKIP_CLONE=1 setsid bash "$0" stage2o > $LOGFILE 2>&1 < /dev/null &
    PID=$!
    disown
    echo "[$(date)] stage2o 启动(训练完自动推理评测), PID=$PID, 日志=$LOGFILE"
  fi
fi

if [ "$1" == "stage3" ]; then
  if [ "$AUTO" == "1" ]; then
    echo "[$(date)] ===== stage3 训练开始 ====="
    stdbuf -oL -eL env PYTHONUNBUFFERED=1 "${CMD_STAGE3[@]}"; rc=$?
    if [ $rc -ne 0 ]; then echo "[$(date)] stage3 训练失败(exit=$rc), 不执行推理评测"; exit $rc; fi
    echo "[$(date)] stage3 训练完成, 自动推理评测..."
    infer_eval_one ckpt/stage3 eval_stage3.jsonl stage3
    exit $?
  else
    LOGFILE=stage3_${LOG_DATE}.log
    AUTO=1 SKIP_CLONE=1 setsid bash "$0" stage3 > $LOGFILE 2>&1 < /dev/null &
    PID=$!
    disown
    echo "[$(date)] stage3 启动(训练完自动推理评测), PID=$PID, 日志=$LOGFILE"
  fi
fi

# 构建偏好数据(不训练模型, 无推理评测)
if [ "$1" == "bpref" ]; then
  LOGFILE=buildpref_${LOG_DATE}.log
  PYTHONUNBUFFERED=1 setsid "${CMD_BPREF[@]}" > $LOGFILE 2>&1 < /dev/null &
  PID=$!
  disown
  echo "[$(date)] bpref 启动, PID=$PID, 日志=$LOGFILE"
fi

# 推理全部 4 个模型(后台, 4路并行)
if [ "$1" == "infer" ]; then
  CUDA_VISIBLE_DEVICES=0 setsid stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/train/infer_eval.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    --output_dir $OUT_DIR --out_file eval_base.jsonl \
    --max_len 1024 --max_samples 18900 --batch_size 8 \
    > infer_base_${LOG_DATE}.log 2>&1 </dev/null &

  CUDA_VISIBLE_DEVICES=1 setsid stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/train/infer_eval.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    --ckpt_dir $OUT_DIR/ckpt/stage2_only \
    --output_dir $OUT_DIR --out_file eval_s2_only.jsonl \
    --max_len 1024 --max_samples 18900 --batch_size 8 \
    > infer_s2_${LOG_DATE}.log 2>&1 </dev/null &

  CUDA_VISIBLE_DEVICES=2 setsid stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/train/infer_eval.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    --ckpt_dir $OUT_DIR/ckpt/stage2_s1 \
    --output_dir $OUT_DIR --out_file eval_s1_s2.jsonl \
    --max_len 1024 --max_samples 18900 --batch_size 8 \
    > infer_s12_${LOG_DATE}.log 2>&1 </dev/null &

  CUDA_VISIBLE_DEVICES=3 setsid stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/train/infer_eval.py \
    --model_path $MODEL_7B --data_dir $DATA_DIR \
    --ckpt_dir $OUT_DIR/ckpt/stage3 \
    --output_dir $OUT_DIR --out_file eval_stage3.jsonl \
    --max_len 1024 --max_samples 18900 --batch_size 8 \
    > infer_s3_${LOG_DATE}.log 2>&1 </dev/null &
  echo "[$(date)] infer 启动(4路并行), 日志: infer_base/s2/s12/s3_${LOG_DATE}.log"
fi

# 评测全部 4 个模型(后台, 4路并行)
if [ "$1" == "eval" ]; then
  CUDA_VISIBLE_DEVICES=0 setsid stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/eval/evaluate_v2.py \
    --model_name base --OMICS all_omics \
    --input_file_path $OUT_DIR/eval_base.jsonl \
    > eval_base_${LOG_DATE}.log 2>&1 </dev/null &

  CUDA_VISIBLE_DEVICES=1 setsid stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/eval/evaluate_v2.py \
    --model_name s2_only --OMICS all_omics \
    --input_file_path $OUT_DIR/eval_s2_only.jsonl \
    > eval_s2_${LOG_DATE}.log 2>&1 </dev/null &

  CUDA_VISIBLE_DEVICES=2 setsid stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/eval/evaluate_v2.py \
    --model_name s1_s2 --OMICS all_omics \
    --input_file_path $OUT_DIR/eval_s1_s2.jsonl \
    > eval_s12_${LOG_DATE}.log 2>&1 </dev/null &

  CUDA_VISIBLE_DEVICES=3 setsid stdbuf -oL -eL env PYTHONUNBUFFERED=1 \
    python $CODE_DIR/eval/evaluate_v2.py \
    --model_name stage3 --OMICS all_omics \
    --input_file_path $OUT_DIR/eval_stage3.jsonl \
    > eval_s3_${LOG_DATE}.log 2>&1 </dev/null &
  echo "[$(date)] eval 启动(4路并行), 日志: eval_base/s2/s12/s3_${LOG_DATE}.log"
fi
