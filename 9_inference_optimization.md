# Chapter 9: Inference Optimization

## Understanding Inference

- Computational Bottlenecks
    1. Compute-bound: time-to-complete is determined by the computation needed for the tasks. E.g., password decryption.
    2. Memory-bandwidth-bound: determined by data transfer rate within the system

- LLM Inference:
    1. **Prefill**: Process input tokens in parallel: compute-bound
    2. **Decode**: Process output tokens in parallel: memory-bandwidth-bound
    - Prefill and decode are often decoupled in production with separate machines.

- Online API: Faster and more expensive
- Batch API: Cheaper and slower. More optimized and usually runs on batches on cheaper hardware (clear 24-hour turnaround time)

- Streaming: reduces time to wait until first token, 🔴 but you can't score a response and it can show bad things (but you can censor anything if it shows up during streaming )

### Inference Performance Metrics
- **Latency** measures the time from when users send a query until they receive the complete response
- If Streaming:
    - **Time to first token**: corresponds to the duration of the prefill step, depends on the input’s length
    - **Time per output token** (tokens / s): it's fast if > Human reading speed (for chats)
    - **Time between tokens and inter-token latency**: time between tokens (TBT), inter-token latency (ITL)
    -  total latency will equal TTFT + TPOT × (number of output tokens)
    - You can handle the tradeoff between TTFT and TPOT by assigning more compute to prefill and less to decode.
    - Some teams use the metric **time to publish** to make it explicit that it measures time to the first token users see (e.g., reasoning models)
- Because latency is a distribution, the average can be misleading, look at latency in percentiles
- Throughput measures the number of output tokens per second an inference service can generate across all users and requests (Token / s / user) or number of completed requests per time.
- Goodput measures the number of requests per second that satisfies the SLO, software-level objective.

### Utilization Metrics
- Utilization metrics measure how efficiently a resource is being used
- GPU utilization is misleading, it represents the percentage of time during
which the GPU is actively processing tasks. For example, if you run inference on a GPU cluster for 10 hours, and the GPUs are actively processing tasks for 5 of those hours, your GPU utilization would be 50.
- Simply:  In nvidia-smi’s definition of utilization, this GPU can report 100% utilization even if it’s only doing one operation per second 
- The real metric is **MFU (Model FLOP/s Utilization)**: out of all the operations a machine is capable of computing, is how many it’s doing in a given time
    - MFU is the ratio of the observed throughput (tokens/s) relative to the theoretical maximum throughput of a system operating at peak FLOP/s.
- **MBU (Model Bandwidth Utilization)** measures the percentage of achievable memory bandwidth used. If the chip’s peak bandwidth is 1 TB/s and your inference uses only 500 GB/s, your MBU is 50%.
    - (parameter count × bytes/param × tokens/s) / (theoretical bandwidth)
    - Example: 7B model, FP16, 100 tokens/s = 7B x 2 x 100 = 140GB/s
- MFU during prefilling is typically higher than MFU during decoding.

### Accelerators
- training demands much more memory due to backpropagation and is generally more difficult to perform in lower precision
- training usually emphasizes throughput, whereas inference aims to minimize latency.
- Two metrics:
    1. FLOP/s
    2. Memory size & Bandwidth
- Accelerators typically specify their power consumption under **maximum power draw** or a proxy metric TDP (thermal design power):

## Inference Optimization
- Model (arrow), Hardware (archer), or service level (overall process, bow, aiming, etc.)

### Model Optimization
- Pruning has 2 meanings:
    1. remove entire nodes of a neural network, which means changing its architecture and reducing its number of parameters
    2. find parameters least useful to predictions and set them to zero, makes the model more sparse (less space and computation)
- Pruned models can be used as-is or be further finetuned to adjust the remaining parameters and restore any performance degradation caused by the pruning process.
-  pruning is less common. It’s harder to do, and there are better options
- Quantization is far more common.

#### Decoding Bottleneck
- **Speculative decoding**: uses a faster but less powerful model to generate a sequence of tokens, which are then verified by the target model. If all draft sequences are rejected. (Take: **Verification is parallelizable**)
- **Inference with reference**: if the model wants to fix a code (with minor changes) or cite a given document. Instead of generating them, copy them from the reference directly.
- **Parallel Decoding**:  some techniques aim to break the sequential dependency. Given an existing sequence of tokens x<sub>1</sub>, x<sub>2</sub>,…,x<sub>t</sub>, these techniques attempt to generate x<sub>t+1</sub>, x<sub>t+2</sub>,…,x<sub>t+k</sub> simultaneously (e.g., it generates x<sub>t+2</sub> before x<sub>t+1</sub>)
    - Medusa uses a tree-based attention mechanism to verify and integrate tokens. Each Medusa head produces several options for each position. These options are then organized into a tree-like structure to select the most promising combination.

#### Attention Optimization
- **KV Cache**: A KV cache is used only during inference, not training. During training, **because all tokens in a sequence are known in advance**, next token generation can be computed all at once instead of sequentially, as during inference. Therefore, there’s no need for a KV cache.
- Memory needed for KV Cache: 2 × BatchSize × SequenceLength × num_layers × model_dim × Memory needed to cache numbers (FP16 or FP32)
    - Example: Llama2 13B, 40 layers, 5,120 model dim, batch of 32, seqlength of 2048 and full precision (FP32) = `2 × 32 × 2,048 × 40 × 5,120 × 2 = 54 GB`
- **local windowed attention** attends only to a fixed size window of nearby tokens, can be interleaved with global attention, with local attention capturing nearby context; the global attention captures task-specific information across the document
- **cross-layer attention**: shares key and value vectors across adjacent layers. Having three layers sharing the same key-value vectors means reducing the KV cache three times.
- **multi-query attention**: hares key-value vectors across query heads.
- **Grouped-query attention**: divides heads into groups.
- Writing kernels for attention: e.g., Flash Attention
    - **Vectorization**: Given a loop or a nested loop, instead of processing one data element at a time, simultaneously execute multiple data elements that are contiguous in memory.
    - **Parallelization**: Divide an input array (or n-dimensional array) into independent chunks that can be processed simultaneously on different cores or threads
    - **Order Fusion**: Combine multiple operators into a single pass to avoid redundant memory access.
    - Kernels are optimized for a hardware architecture. This means that whenever a new hardware architecture is introduced, new kernels need to be developed

### Inference Service Optimization
- Batching: service might receive multiple requests simultaneously. Instead of processing each request separately, batching the requests that arrive around the same time together can significantly reduce the service’s throughput. (Many cars ➡️ Bus)
    1. **Static Batching**: The service groups a fixed number of inputs together in a batch. It’s like a bus that waits until every seat is filled before departing. 🔴 all requests must wait until the batch is full.
    2. **Dynamic Batching**: sets a maximum time window
    -  all batch requests have to be completed before their responses are returned (10 tokens response needs to wait for 1000 tokens response to be finished)
    3. **Continuous Batching / in-flight batching**: allows responses in a batch to be returned to users as soon as they are completed.
    After a request in a batch is completed and its response returned, the service can add another request into the batch in its place, making the batching continuous (Like a bus, after dropping off a passenger, the bus can pick up another passenger)

- **Decoupling prefill and decode** (mentioned before), it requires transferring intermediate states from
prefill instances to decode instances
- **Prompt Caching**: stores these overlapping segments for reuse
    - a common overlapping text segment in different prompts is the system prompt
    - useful for queries that involve long documents (e.g., users uploading the same document or book)

#### Model Parallelism
- **tensor parallelism**: shards individual parameter tensors (e.g., weight matrices) across multiple devices so that each device holds only a fraction (1/N) of a large tensor
- **pipeline parallelism**: splits the model into multiple stages, each stage is processed by a different device, but 🔴 increases the total latency for each request due to extra communication between pipeline stages
- **Context parallelism**: first half of the input is processed on machine 1 and the second half on machine 2.
- **Sequence parallelism**: attention might be processed on machine 1 while feedforward is processed on machine 2.