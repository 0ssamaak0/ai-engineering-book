# Chapter 7: Fine Tuning
- Prompting, Agents, RAG: Using instructions, context, and tools. While fine tuning adjusting its weights.
- Can improve domain-specific capabilities e.g., coding or medical QA, safety etc.
- most often used to improve instruction-following.
- start with a base model that has some, but not all, of the capabilities you need.
- Ideally, much of what the model needs to learn is already present in the base model, and finetuning just **refines the model’s behavior.**
- Fine tuning is not the only way of transfer learning. **Feature based transfer** is another way when you extract the features e.g., adding classification head to BERT (Freezing all layers and train the head only)
- Before Fine tuning for task specific, fine tune with self supervision using cheap task related data: e.g., before fine tunign with legal QAs, fine tune on raw legal docs.
- Long-context finetuning: finetune a model to extend its context length, includes adjusting the positional encoding


## Should I fine tune?
- To get better structured output e.g., JSON and YAML
- for less common tasks e.g., SQL of certain dialect
- bias mitigation
- 🔴 Fine tuning the model on a specific task can degrade the performance on other tasks (**Alignment tax**)
    - You may consider using different models for different tasks if you can't fine tune a model to be good at all tasks.
- Fine tuning is not the first thing you should try: expensive, requires data and maintance.
- Fine tuned models usually requires serving it, you need to host it yourself usually.
- Fine tuning can help optimize token usage (before prompt caching was introduced) you can save more input tokens by use the examples to fine tune the model yourself.
- RAG or FT first? if the system lacks information, use RAG first. if system has behavioral issues or not following instrutions, use FT: **In short, finetuning is for form, and RAG is for facts.**


## Memory Math
- Inference: N x M x 1.2 (N: # Params, M: # bits, 12.: for key-value vectors)
- Training: model weights + activations + gradients + optimizer state
- During backprop: each trainable param requires one value for gradient + [0: SGD, 1: momentum, 2: Adam] values for optimizer states:
    - Example: 13B param, 2 bits per param, using adam: Gradient + Optimizer memory = 13B * 2 * (1 + 2) = 78GB Memory
- Gradient Checkpointing / Activation Recomputation: Instead of storing activations, you can recompute them during backprop. (Less Memory, but more timr)
- BF16 can represent more range than FP16, but it has less precision.
- **Quantization** formally is reducing to integer formats, and **reduced precision** is reducing to floating point formats. But it's common to use **quantization** for everything now.
- Training Quantization: 1. to make it model perform better in inference. 2. to reduce training cost.
- QAT: simulate the low precision behavior during training, but it doesn't reduce training time (it can even increase it since the simulation of low precision takes some time)
- Lower Precision training: helps in both issues, but harder to do. The intensive operations are done in FP16 while the loss computation and weight updates are done in FP32.

## PEFT
-  a technique is considered PEFT if it can achieve performance close to that of full finetuning while using several orders of magnitude fewer trainable parameters.
- Introduced in 2019: inserting additional parameters into the model in the right places, you can achieve strong finetuning performance using a small number of trainable parameters. They inserted 2 adapters into each transformer block of a BERT model.
- 🔴 Increases inference time.
- PEFT categories:
    1. adapter based: like the mentioned above and LoRA
    2. Soft Prompts: adding special trainable embeddings.

### Soft Prompts:
- Hard prompts: the human readable, static, non trainable tokens.
- Soft Prompts: embeddings added somewhere in the model.
- Prefix Tuning: soft prompt tokens to the input of every transformer layer.
- Prompt Tuning: only the embedding input.

### LoRA
- LoRA doesn't increase inference, the modules can be merged back to the original layers
- LoRA Decomposes the weight matrix into the product of two smaller matrices.
- Given a weight matrix W (n x m) ➡️ `W = A (n x r) @ B (r x m)` (r is the LoRA rank)
- Now define the weight is `W_new = W + ΔW` where `ΔW = A @ B` (you can use a hparam alpha to scale the LoRA weight)
- Update only the parameters of A and B.

> why is parameter efficiency possible at all? If a model requires a lot of parameters to learn certain behaviors during pre- training, shouldn’t it also require a lot of parameters to change its behaviors during finetuning?

> same question can be raised for data. If a model requires a lot of data to learn a behavior, shouldn’t it also require a lot of data to meaningfully change this behavior? How is it possible that you need millions or billions of examples to pre-train a model, but only a few hundreds or thousands of examples to finetune it?

> Many papers have argued that while LLMs have many parameters, they have very low intrinsic dimensions; see Li et al. (2018); Aghajanyan et al. (2020); and Hu et al. (2021). They showed that pre-training implicitly minimizes the model’s intrinsic dimension. Surprisingly, larger models tend to have lower intrinsic dimensions after pre-training. This suggests that pre-training acts as a compression framework for downstream tasks. In other words, the better trained an LLM is, the easier it is to finetune the model using a small number of trainable parameters and a small amount of data.

** Intrinsic Dimension **: the minimum number of independent parameters needed to describe the data, effectively capturing its underlying structure (in optimization theory)

#### LoRA Configurations
- 2 Questions: What matrices to apply lora &  rank of each factorization.
- 4 Weights Matrices to apply lora: Query, Key, Value matrices and the output projection matrix.
- If you are in tight memory, apply lora to query and value matrices.
- small r, such as between 4 and 64, is usually sufficient for many use cases. Higher r is usually not needed.
- LoRA Equation: $W' = W + \frac{\alpha}{r} \cdot W_{ab}$

#### Serving LoRA
1. Merge LoRA Weights to the original weights to create `W'` (no extra latency)
2. Keep` W, A, and B` separate during serving. The process of merging A and B back to W happens during inference, (adds extra latency) 🟢 Useful in serving multiple LoRA models and uses far less storage.

#### QLoRA
- QLoRA stores the model’s weights in 4 bits but dequantizes (converts) them back into BF16 when computing the forward and backward pass.
- Uses NF4 (Normal Float 4) which is based on the insight that pretrained weights follow a N(0, 1) distribution.
- Uses Paged Optimizers automatically to transfer data between CPU and GPU when GPU is out of memory.
- A 65B-parameter model can be finetuned on a single 48 GB GPU.
- 🔴 NF4 is expensive (quantization and dequantization requires extra computation)

## Model Merging
- If you want to fine tune for multitask:
    1. Simultanous finetuning: create a dataset of all examples and fine tune on all tasks at once
    2. Sequential finetuning: fine tune on each task one by one
- Model merging is one way of fedarated learning
- started with model ensemble methods

### Merging Approaches
#### 1. Summing
- adding the weight values of constituent models together
1. Linear Combination: weighted average (can be equal weights): model soups
    - it’s more common to merge models by linearly combining specific components, such as their adapters.
    - Using fine tuned vectors, you can use `task vectors`
2. SLREP: you can think of each component (vector) to be merged as a point on a sphere. To merge two vectors, you first draw the shortest path between these two points along the sphere’s surface. The merged vector of these two vectors is a point along their shortest path.
    - SLERP can be applied on 2 vectors only, if you want more you must merge them sequentially.
- TIES & DARE

#### 2. Layer Stacking / franken‐merging
- Use layers from different models
- Can be used in training MoE models: you take a pre-trained model and make multiple copies of certain layers or modules. A router is then added to send each input to the most suitable copy
- **Model upscaling**: the study of how to create larger models using fewer resources
    - **Depthwise scaling** : SORAL 10.7B from one 7B parameter model. Merge 2 copies of these models by summing some layers and stacking the result. Then further train the upscaled model.

#### 3. Concatenation
- you can also concatenate them. The merged component’s number of parameters will be the sum of the number of parameters from all constituent components.
- 🔴 doesn't reduce the memory footprint.

## Fine Tuning Tactics
- Try to fine tune with a middling model first,
- Try with different models to see the costs and performance trade offs.
- You can FT a strog model and use it to generate more training data.
- If you have small dataset, full finetuning may not exceed LoRA
- Take into account how many fine tuned models you want to deliver and how to serve them (maybe use adapters?)
- Hyperparameters:
    - A common practice is to take the learning rate at the end of the pre-training phase and multiply it with a constant between 0.1 and 1.
    - don't use small batch sizes (e.g., <8) if you have memory limitation. Use gradient accumulation.
    - 1-2 Epochs for large datasets (millions) and 4-10 epochs for small datasets (thousands)
- For instruction fine tuning, the example is (prompt, response) pair: response tokens should contribute more to the model’s loss during training than prompt tokens.
- **prompt_loss_weight (PLW)**  is how much prompt contributes to the loss. (100%: same contribution as response). It's usaully set to 10% or even lower (OpenAI is using 0.01)