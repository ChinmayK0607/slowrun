# Hybrid Attention

Hybrid attention track led by [@ChinmayK0607](https://x.com/ChinmayKak).

## Records

| PR | Record | Time |
| --- | --- | --- |
| PR `#49` (`1f0fe74`) | `3.246` val loss | about `81` minutes |
| PR `#58` (`5cb9428`) | `3.241282` val loss | `72.33` min training, `76.91` min wall |
| KDA / FlashKDA extension | `3.239565` best val loss, `3.255612` final val loss | `89.28` min training, `94.42` min wall |

## Usage

GDN reference run:

```bash
torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23
```

KDA / FlashKDA run:

```bash
FLA_FLASH_KDA=1 torchrun --standalone --nproc_per_node=8 research/hybrid_attn/train.py \
  --gdn-layers 1,3,5,6,8,10,11,13,15,16,18,20,22,23 \
  --linear-attn-type kda
```

## Brief History

1. PR `#49`: introduced the core hybrid idea. The theoretical change was to mix full softmax attention with GatedDeltaNet, so some layers use recurrent associative memory instead of recomputing all context from scratch. The negative-eigenvalue GDN state makes it easier to track changing latent state, not just accumulate information.

2. PR `#58`: kept the same GDN theory, but improved the efficiency frontier of that idea. The important point was not a new inductive bias; it was making the same recurrent-memory-plus-softmax-correction story cheaper to run, so the track could realize more of the hybrid benefit within a practical time budget.

3. KDA / FlashKDA: upgraded the recurrent memory from a single forget coefficient per head to a per-dimension forget mechanism. The theoretical effect is a more expressive state space, where one head can preserve different subspaces for different timescales instead of forcing the whole head to forget or retain together. That improved loss, but it also increased runtime.

Best loss so far is KDA, but GDN stays the default because it is materially faster.
