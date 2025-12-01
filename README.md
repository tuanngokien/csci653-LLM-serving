# Project Proposal: Memory-Aware Request Scheduler for Mitigating Request Premption in LLM Serving

---
### Problem Statement

Large Language Model (LLM) serving systems (e.g., vLLM, TGI) maximize throughput by batching requests given the available KV-cache memory. These systems manage the Key-Value (KV) cache using paging techniques (e.g., PagedAttention) to continuously fit multiple requests with varying lengths into limited GPU memory. As requests decode and generate new tokens, their context windows grow and the memory requirement expands. When the aggregate growth exceeds physical capacity, the system hits an Out-Of-Memory (OOM) state.

A common strategy is **recomputation**: a request is evicted, its KV-cache memory freed, and later its state is recomputed by encoding/prefilling the entire token sequence so far. This can keep the GPU at high utilization (near 100%), but a significant portion of GPU compute is then spent re-calculating the same token history repeatedly rather than generating new tokens. Because the prefill phase of Transformer inference scales roughly quadratically with context length, recomputing long-context requests is especially expensive.

This problem arises under existing schedulers:

- **First-Come-First-Served (FCFS) schedulers** typically admit new requests whenever *immediate* memory is available. They fail to account for the *future memory growth* of the currently running batch as decoding proceeds, so the aggregate KV usage can still exceed GPU capacity and trigger OOM and recomputation.

- **Advanced schedulers with priority-based policies** (e.g., Shortest Job First, Shortest Predicted Remaining Processing Time) improve mean throughput and latency by mixing extremely short and long jobs in the same batch. However, they primarily (i) reorder requests to favor short jobs (using response-length prediction), which can cause early-arriving long jobs to be repeatedly delayed; (ii) usually do not jointly optimize request admission considering *future* KV-cache usage; and (iii) treat preemption as a cheap pause/resume operation rather than a costly “evict + recompute” event in systems.

As a result, even with sophisticated scheduling, LLM serving stacks can still suffer from frequent, expensive preemptions that waste a substantial fraction of available FLOPs on redundant recomputation instead of serving new user tokens.

## Contributions

GPU memory used by the KV cache at any time is largely determined by two components:  
- KV entries for prompts of *newly admitted requests*, and  
- Growing KV entries for decoded tokens of *already running requests*.   

The size of KV entries for the former is deterministic and fully known at arrival (it depends only on input prompt length). In contrast, the KV growth of the latter is typically unknown. By using a response-length predictor, we can turn this “unknown future” into an explicit forecast. For each active request, we can estimate its future token budget and thus its future KV-cache footprint over time. This allows:

- Plan admission and batching decisions under *both current and predicted future* KV usage,  
- Proactively avoid OOM that would trigger expensive evictions and recomputation, and  
- When preemption is unavoidable, select victims based on their *predicted recomputation cost per unit of freed KV memory*, rather than purely on arrival time or predicted completion time.


### Specific Objectives

1. **Recomputation Overhead Analysis** 
   - Extend a production-grade LLM serving stack (e.g., vLLM) with instrumentation to track:
     - per-request `preemption_frequency` (how many times each request is evicted),
     - per-request `recompute_tokens` (how many tokens are reprocessed after eviction), and
     - estimated `wasted_flops` due to recomputation, based on model shape and sequence lengths.
   - Evaluation with synthetic mixed workloads (short/long-context requests with varying arrival rates) that trigger OOM request preemption.
   - Quantitatively characterize how much GPU time and FLOPs are spent on recomputation under baseline schedulers (e.g., FCFS).
      
2. **Memory-Aware Scheduler** 
   - Use response-length prediction to **control which and how many requests are admitted concurrently**, based on both current and *predicted future* KV-cache usage.
   - Propose a preemption-aware dynamic batching algorithm that:
     - Accounts for expected KV usage over time,
     - Forms batches that avoid placing too many high-risk (long, memory-heavy) requests together
   - Evaluation that such admission and batching policies can reduce OOM in many workloads, thereby mitigating its recomputation cost.

3. **Cost-Aware Victim Selection**
   - Develop a cost model that estimates, for each active request, the compute (i.e., FLOPs) required to restore its state if evicted.
   - Propose and implement preemption policies that:
      - Prefer evicting requests with low recompute cost per MB of KV freed,
      - Incorporate request progress to protect near-completion sequences,
   - Compare these policies to FCFS baseline, where the most recently admitted requests are evicted first.

4. **End-to-End Evaluation**
   - Evaluate all proposed policies on realistic traces and stress tests, reporting:
     - `TTFT` (time to first token before initial output appears),
     - `TPOT` (time per output token during decoding),
     - ``GPU_utilization`` (fraction of time the GPU is actively computing),
     - ``recompute_fraction`` (fraction of total FLOPs spent on recomputing evicted KV state).
---

