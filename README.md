# Project: Mitigating Request Premption in LLM Serving

---
### Problem Statement

Large Language Model (LLM) serving systems (e.g., vLLM) maximize throughput by batching requests regarding the available KV Cache memory. Existing schedulers typically use a LIFO (Last-In-First-Out) policy, and when GPU memory is exhausted, the scheduler must preempt running requests. A common strategy is **Recomputation**, where a request is evicted, its memory freed, and it is later restarted from scratch. This means while the GPU runs at 100% utilization, but a significant portion of compute (FLOPS) is spent re-calculating the same token history repeatedly, rather than generating new tokens. As LLM context windows grow (e.g., 128k+ tokens), the cost of recomputing a request becomes quadratic in the context length. Preempting a long-context request to save memory can result in seconds or minutes of wasted compute time to restore its state. Solving this is critical for the **energy efficiency** and **latency** of production LLM services.

### Specific Objectives

1. **Recomputation Analysis**  
   Instrument in vLLM's scheduler to track:
     - `premption_frequency` — frequency of repeated preemptions for the same request ID
     - `wasted_flops` — redundant computations due to recomputation  
   
   The analysis can be done by creating a synthetic workload of mixed short-context and long-context requests to trigger the out-of-memory mode in the vLLM scheduler.
2. **Scheduler Development**
   Design and simulate two novel preemption algorithms:
   - Cost-Aware Victim Selection: Prefer preempting requests with short contexts (cheap to recompute) over long-context requests.
  
3. **Evaluation**  
   Compare the throughput of the proposed policies against the baseline LIFO scheduler.
---

