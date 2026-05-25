#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${ROOT_DIR}:${PYTHONPATH}"

# ---- DenseGroup MaxAUC SOTA: four residual branches + EMA selection ----
# Mainline choice:
#   - keep the original row-group validation split as default
#   - enable bf16 autocast only (no compile)
#   - raise emb_skip_threshold to 1.1M so `domain_c_seq_34` comes back,
#     while `domain_c_seq_29 / 47` and `domain_b_seq_69` still stay skipped
#   - add absolute sample-time features (day-of-week + hour-of-day embeddings + cyclic hour sin/cos)
#   - sequence encoder outputs re-mask padding positions to zero
#   - add a zero-initialized target-to-history matching residual branch
#   - enable four MaxAUC residual branches: user dense / DIN / DCNv2 / auxiliary logit
#   - maintain dense-parameter EMA (decay=0.9995) and validate raw+EMA, saving whichever has higher AUC
#
# Changed:
#   raw item -> target_anchor -> query modulation + final head
#   target_anchor -> cross-attend seq_a/b/c/d -> zero-init residual into fused head
#   target_anchor -> DIN-style latest-history attention -> zero-init residual
#   NS tokens -> low-rank DCNv2 cross branch -> zero-init residual
#   user dense token -> transformed dense residual encoder -> zero-init residual
#   fused_output + target_anchor -> auxiliary logit residual, zero-init
#   loss: BCE -> Focal(alpha=0.1, gamma=2.0)
#   high-card seq: restore only `domain_c_seq_34`
#   precision: CUDA bf16 autocast on
#   time features: sample-level day_id / hour_id / hour_sin / hour_id broadcast-added to all NS tokens
#   padding fix: SwiGLU/Transformer/LongerEncoder outputs re-masked
#   selection: raw/EMA dual eval by AUC
# Unchanged:
#   validation split: original row-group tail split
#   compile: off
#   num_queries / num_ns / RankMixer token arithmetic / multi-seq blocks
export USE_BF16=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --seq_encoder_type transformer \
    --ns_groups_json "" \
    --emb_skip_threshold 1100000 \
    --batch_size 256 \
    --num_workers 8 \
    --save_epoch_ckpt 8 \
    --use_bf16 \
    --use_target_history_matching \
    --use_user_dense_groups \
    --use_user_dense_residual \
    --use_din_residual \
    --din_top_k 80 \
    --use_dcn_residual \
    --dcn_low_rank 128 \
    --use_aux_logit_residual \
    --ema_decay 0.9995 \
    --ema_start_step 100 \
    --ema_update_every 1 \
    --eval_raw_and_ema \
    --loss_type focal \
    --focal_alpha 0.1 \
    --focal_gamma 2.0 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
