[Very useful notes](https://mlops.systems/posts/2025-01-19-notes-on-ai-engineering-chapter-1.html)

# Chapter 1
Why LLMs Exploded? Self Supervised Learning

## Foundation Models
- The transition from task-specific models to general-purpose models.
- Trained on massive, diverse datasets using self-supervised learning.
- Shows abilities not explicitly programmed or anticipated.

## Three Layers of the AI Stack
- Application development
- Model development
- Infrastructure

## Comparison to ML
- Without foundation models, you have to train your own models for your applications. With AI engineering, you use a model someone else has trained for you.
- AI engineering works with models that are bigger, consume more compute resources, and incur higher latency than traditional ML engineering. This means that there’s more pressure for efficient training and inference optimization. 
- AI engineering works with models that can produce open-ended outputs. Open ended outputs give models the flexibility to be used for more tasks, but they are also harder to evaluate.

# Chapter 2
## Training Data
### Multilingual
- Half of common crawl is English
- Q: Why not to translate everything to English, solve and then translate back?. A: Expensive, information loss in translation (e.g., subjects)
- Median token length in English is 7, Hindi is 32.

### Size of Data:
- Training samples e.g., (image, text) pairs
- Not representative for LMs since sample can be a sentence or wikipedia page. We use # Tokens instead (Trillions)

## Modeling
### 3 Numbers signal a model's scale
- Number of parameters, which is a proxy for the model’s learning capacity. (Billion parameters)
- Number of tokens a model was trained on, which is a proxy for how much a model learned (Trillion Tokens)
- Number of FLOPs, which is a proxy for the training cost (Very high orders of magnitude e.g., 3.8 ×10^25 FLOPs for Llama 3.1 405B, GPT-3-175B using 3.14 × 10^23 FLOPs)

### Chinchilla Scaling Law
- Given a fixed compute budget, calculate the optimal model size and dataset size.
- the number of training tokens to be approximately **20 times the model size**. This means that a 3B-parameter model needs approximately 60B training tokens ➡️ Doubling the model size requires doubling the number of training tokens.

### Scaling extrapolation / Hyperparameter transferring
- study the impact of hyperparameters on models of different sizes, usually much smaller than the target model size
- then extrapolate how these hyperparameters would work on the target model size

### Bottlenecks
- Data availability
- Electricity

## Post Training

### Concerns
- Pretraining: Token level quality
- Post Training: Entire response quality

### Stages
- SFT on high quality instructions data
- Preference Fine Tuning: RLHF, DPO, etc.
- Some people use the term **instruction finetuning** to refer to **supervised finetuning**
- while some other people use this term to refer to both **supervised finetuning** and **preference finetuning**. 
- To avoid ambiguity, the author will avoid the term **instruction finetuning** in this book.

### Cost
- Very low compared to pretraining (InstructGPT used 2% of total compute)

### RLHF
1. Train a reward model that scores the foundation model’s outputs (Given pair of (prompt, response), the reward model scores the response)
2. Optimize the foundation model to generate responses for which the reward model will give maximal scores (PPO)

### Best of N
- Some companies skip RL and use the reward model only.
- They generate multiple outputs from the model and use the reward model to select the best one.

## Sampling
Done in my notes

### Test Time Compute
- Generate multiple outputs and select the best one based on the average negative logprob of the whole response.
- We can use voting for Math or MCQ questions
- Robust model: doesn’t dramatically change its outputs with small variations in the input.


### Structured Outputs
- post processing: models usually repeat their failures e.g., missing close brackets (remember claude {{ issue in plotly }})

### Inconsistency
- Same input, different outputs: You can fix the sampling variables. But not guaranteed 100% if you are using API (Host machine itself can introduce some randomness)
- Slightly different input, drastically different outputs: prompting and memory (Later)

### Hallucination
- mentioned in 2016
- 1- Snowballing hallucinations: a language model hallucinates because it can’t differentiate between the data it’s given and the data it generates
- 2- mismatch between the model’s internal knowledge and the labeler’s internal knowledge


# Chapter 3: Evaluation Methodology

## Why Evaluation is Hard?
- Models became better
- Open ended, no Ground Truth
- Foundation models became black boxed (either closed source or devs have no expertise

## Language Modeling Metrics
- Cross Entropy, Perplexity, BPC, BPB
- You can compute the others if you have one and the necessary info
- They can be used for any model that generates sequence of tokens (even non textual tokens)

### Entropy
- You create a language to describe the position in a square
- Entropy of a language is 1 bit: Upper or Lower
- Entropy of a language is 2 bits: Upper left, Upper right, Lower left, Lower right
- Lower entropy means more predictable and vice versa

## Exact Evaluation

### Functional Correctness
Does it do the job?
- Examples: Code Generation, Game bots (like tetris)

### Similarity Measurements Against Reference Data
- Each example in the reference data follows the format (input, reference responses).
- Matching? 
- Evaluator
- exact match
- lexiacl similarity (overlapping, fuzzy matching, n-gram matching like BLEU)
semantic similarity.

## LLM as a Judge
1. Evaluate the quality of a response by itself, given the original question: Given the following question and answer ...
2. generated response to a reference response to evaluate whether the generated response is the same as the reference response: Given the following question, reference answer, and generated answer
3. two generated responses and determine which one is better or predict
which one users will likely prefer: Given the following question and two answers, evaluate which answer is better (Good for generating preference data)
- Issues in latency and cost
- Issues in bias: model favors itself, favors first answer, favors lengthier answers (even with factual errors)

### Specialized LLMs (As Judges)
- Reward model (used in RLHF, Given a pair of (prompt, response) and produces score between 0 and 1)
- Reference-based judge: BLEURT (Sellam et al., 2020) takes in a (candidate response, reference response) pair and outputs a similarity score between the candidate and reference response
- Preference model: A preference model takes in (prompt, response 1, response 2) and outputs which one is prefered by users. e.g., PandaLM

# Chapter 4: Evaluate AI Systems

## Evaluation Criteria
- You don't know how your application is being used.
- Evaluation-driven development: defining criteria before building.
- For example: Recommender systems, increase in engagement.
- Criteria (5 points): domain-specific, generation, instruction following, cost, and latency.

> Chapter 3: what criteria a given approach can evaluate. 
> This Chapter: given a criterion, what approaches can you use to evaluate it?

### Domain-Specific Capability
- constrained by model conifug (size, training data, etc.)
- you can rely on domain specific benchmarks, either public or private.
- benchmarks with close-ended outputs such as MCQ is common
- Can vary with small changes, adding extra space between question and answer can change the model's answer.
- MCQs test the ability to differentiate good responses from bad responses (classification), which is different from the ability to generate good responses.

### Generation Capability
- Fluency: grammatically correct and natural sounding
- Coherence: measures how well-structured the whole text is (does it follow a logical structure?)
- Othet task specific metrics: Faithfulness in translation, relevance in summarization, etc.
- Factual Consistency
    - Local: Given Context (e.g., summarization, respect company policy in chatbots)
    - Global: Open Knowledge Base (e.g., fact checking)
    - Problem: What is the fact? e.g., breakfast is the most important meal of the day.
    - **SAFE**: decompose response into individual statements (and make them self contained) then check each statement.
    - If using a context you can think of the task as *textual entailment* task (entailment, contradiction, neutral)
- Safety

### Instruction Following
- Example: prompt asks for sentiment analysis, the model generates emotions outside the list.
- Example: structured output (JSON, regex, SQL)
- IFEval Benchmark (Google): 25 types of instructions that can be automatically verified, such as keyword inclusion, length constraints, number of bullet points, and JSON format.  
- INFOBench Benchmark: follow content constraints (such as “discuss only climate change”), linguistic guide lines (such as “use Victorian English”), and style rules (such as “use a respectful tone”)
- You should curate your own benchmark to evaluate your model’s capability to follow your instructions using your own criteria. If you are using YAML you should include YAML in the benchmark. Take inspiration from each benchmarks too!
- Role playing: Make sure your model stays in the character.

### Cost and Latency
- Latency: Time to first token, Time per token, time between tokens, etc.
- Latency doesn't depend only on the model. Since more tokens increase the total latency, prompting can make a big difference in latency.

## Model Selection
- Selection process: 
    1. best achievable performance
    2. best performance (based on your budget)
- hard attributes: can't change, made by model provider (e.g., API latency, license) or company policy (privacy)
- soft attributes: can change like accuracy, fact consistency, latency for hosted models. etc. (can be improved by prompting, splitting the task, etc.) Balance between being optimistic and realistic.
- Workflow for model selection (iterative):
    1. Filter by hard attributes
    2. Use benchmarks and arena to narrow the list
    3. Experiment with different prompts and models (use your own evaluation criteria)
    4. Continuous monitoring (detect failures and feedbacks)

### Open or Closed?
- What is open? Weights only or weights and data?
- "Open Weight" is used for models that don’t come with open data
- "Open Model" is used for models that come with open data
- License:
    - Allows Commercial Use? with any restrictions? e.g., Llama < 700M users only
    - Can I use the model outputs to train other models? (Model Distillation)
- open vs closed:
    1. Data Privacy: e.g., leaking company secrets.
    2. Data lineage and copyright
    3. Performance
    4. Functionality: e.g., scalability, RAG, agents, structured outputs, logprobs, etc.
    5. Cost: API vs Engineering
    6. Control and access: e.g., rate limits, terms, model tweaking, model being deprecated or changed (GPT4o issue in April), banning your country.
    7. On device deployment: no internet, privacy, etc. (Apple example)

### Benchmarks
- Aggregation:
    1. What benchmarks to include
    2. How to aggregate their results?
#### What benchmarks to include
- Public leaderboards:
    - Aggregated performance but it has issues
    1. Using only subset of benchmarks
    2. selection of benchmarks is not always transparent
- Huggingface Leaderboard (Deprecated)
    - ARC-C: Complex grade school level science questions
    - MMLU: 57 subjects, math, history, CS, law
    - HellaSwag: predict the completion of a sentence or a scene in a story or video
    - GSM-8K: Math (replaced later by MATH lvl 5)
#### How to aggregate their results?
    - Huggingface was just averaging
    - HELM uses mean win reate: the fraction of times a model obtains a better score than another model, averaged across scenarios
    - Note that the units are different: accuracy, BLEU, F1, etc.
#### Data Contamination (PHI)
- Intentional: Cheating
- Non Intentional: From just scraping
- Intentional for good reason: The High quality benchmark will improve the model
- Handling it: N-gram overlapping (expensive) or perplexity (lower perplexity means contamination)
- My question: Machine Unlearning?

## Evaluation Pipeline
- When creating the evaluation guideline, it’s important to define not only what the application should do, but also what it shouldn’t do
- A correct response is not always a good response
- Create Scoring Rubrics (binary, out of 5, etc.)
- Use logprobs when available
- Annotate evaluation data.
- A useful rule is that for every 3× decrease in score difference, the number of samples needed increases 10× (e.g., 30% 10, 10% 100, 3% 1000, 1% 10000 etc.)
- pipeline should be reliable (if run twice, same results)
- How correlated are your metrics?
- cost and latency of the evaluation pipeline


# Chapter 5: Prompt Engineering
## Prompt structure
- Task Description, Example (One shot here), the task itself.
- measure robustness: randomly perturbing the prompts to see how the output changes
- For Better models, the improvement of few shot learning is limited compared to 0 shot learning (because the model is better in understanding and following instructions)
- Under the hood, the system prompt and the user prompt are concatenated into a single final prompt before being fed into the model
- But it comes first, maybe post training make models give more attention to it (Open AI)

## Prompting Best Practices
- For bad models, people often use something like 3000$ tip or writing Q instead of Question. This is became kinda outdated for good models.
1. **Explain what do you want**: If you want the model to score, tell him the scoring system (e.g., out of 5)
2. **Add persona to the model**: be specific, if you tell him he is first grade teacher, he may give better score for simple essays.
3. **Provide Examples**: 
4. **Specify the output format**: 
5. **answer using only the provided context**
6. **Break Complex Tasks into Subtasks**: Prompt Decomposition benefits:
    - Monitoring: You can monitor not just the final output but also all intermediate outputs.
    - Debugging: isolate the step that is having trouble and fix it independently
    - Parallelization: execute independent steps in parallel to save time
    - Effort: Easier to write simple prompts.
    - 🔴 Downside: More Latency and cost ➡️ but you can use faster and more cheap model for some easy steps!
- Chain of thought (Reasoning models weren't released yet)
- Prompt Optimization: DSPY, Open-prompt, promptbreeder, TextGrad
- You can use LLM to optimize the prompt
- 🔴 Issues of prompt engineering tools: 
    1. hidden API calls (extra costs)
    2. may has some errors and typos
    3. May change without warning
- Separate prompts from code, you can even make a class for the prompt and add metadata for it (e.g., model, temperature, date_created, etc.) (.prompt format)

## Defense against prompt attacks:
- Prompt Extraction: either for replication or exploitation
- Jailbreaking & Injection
- Information extraction: revealing training data
- PAIR: Uses AI powered attacks
- Indirect injection: Leaving the injection in something the model can retrieve e.g., public space like github repo or reddit. (Kinda data poisoning)
- To test against attacks: violation rate and false refusal rate.

# Chapter 6: RAG and Agents
## RAG
- retrieve-then-generate: 2017, RAG term in 2020
- Context construction for foundation models is equivalent to feature engineering for classical ML models: Give the model the necessary information
- early days: RAG to overcome limited context but this is incorrect: 1. we need always more context 2. we need to focus on what matters.
- Components: Retriever and generator (separate models, but the whole RAG System can be trained end to end)
- Two main functions:
    1. Indexing: processing the data to be retrieved later
    2. Quering: Sending the query to retrieve the most relevant data
- Retrieval algorithms:
    - Not unique to RAG, being used in search engines, recommendation systems, etc.
- Metrics:
    - Context Precision: how many of the retrieved documents are relevant to the query
    - Context Recall: how many of the relevant documents are retrieved (quite hard since you need to annotate all documents in your dataset.)
- Take care of indexing and querying trade offs. slower and larger index will result in better query results.

### Sparse Retrieval (Term Based):
- Based on keyword occurrence
- Two problems
    1. 🔴 many documents may have the term ➡️ take the ones with highest frequency (TF)
    2. 🔴 You need to focus on the terms that are most relevant to the query (not words like as, the, it, etc.)
    - "the more documents contain a term, the less informative this term is" ➡️ IDF
    - TF-IDF: TF * IDF
- BM25: TF-IDF + document length normalization
- Tokenization: most common n-gram (to avoid splitting word like hot dog into hot and dog) Built in nltk, spacy an sklearn

### Dense Retrieval (Embedding Based):
- Vector Search Algorithms:
    - LHS
    - HNSW
    - Product Quantization
    - IVF
    - Annoy
- Slow, but the latency usually comes from the generation, not the retrieval.

### Hybrid Retrieval:
- Use fast (usually sparse) retrieval first, then use a more precise (usually dense) retrieval to get the most relevant documents. (called a reranker)

### Retrieval Optimization:
- Chunking Strategy: How your data should be indexed depends on how you intend to retrieve it later.
    - simple: equal fixed length (words, chars, etc.)
    - recursive: document ➡️ sections ➡️ paragraphs ➡️ sentences ➡️ words .. etc.
    - Creative strategies for special documents e.g., code, QA docs (Q and A are a chunk)
    - Take care of overlapping, you must ensure the information is included in at least one chunk.
    - Chunk size < min(LLM context window, embedding context window)
    - 🔴 Smaller chunk sizes can also increase computational overhead. 
    > 🔴 Small chunk sizes, however, can cause the loss of important information. Imagine a document that contains important information about the topic X throughout the document, but X is only mentioned in the first half. If you split this document into two chunks, the second half of the document might not be retrieved, and the model won’t be able to use its information.
    - Reranking: More crucial in search, but still important in RAG
    - **Query rewriting** is also known as **query reformulation**, **query normalization**, and sometimes **query expansion**:
    - You rewrite the query to be more specific and to be relevant to the data. e.g., use any previous context that might help, convert subjects like he or she to actual names, etc.
    - We can use AI models for it
    - **Contextual Retrieval**: Augment the chunk with relevant context 
        - metadata: tags, keywords, captions for images
        - entities (e.g., EADDRNOTAVAIL (99))
        - you can also augment the article with related questions (e.g., how to reset password: augment it with "I forgot my password" question)
        - Some chunks may don't have the necessary context, you can add the original document title and summary (Anthropic did this)

### Evaluating Retrieval (Questions)
- What retrieval mechanisms does it support? Does it support hybrid search?
- If it’s a vector database, what embedding models and vector search algorithms does it support?
- Scalable?
- Time to index, how much data can be added or removed at once?
- query latency
- pricing?

### Multimodal RAG
- For images, you can use something like CLIP to embed the image and then use the embedding to retrieve the most relevant images.
- for tabular data: text to SQL, execute the SQL, generate a response based on the result.

## Agents
- anything that can perceive its environment and act upon that environment (using tools that it has access to)
- agents reason about the tools being used, and can determine of the information is enough or not.

### Tools
- Not necessary, but it helps agent to preceive the env (read-only tools) and act upon it (write tools)
- Types of Tools:
    1. Knowledge Augmentation: like web browsing or access to any kind of data.
    2. Capability extension: calculator, time zone converter, unit converter, **Code Interpreter**, etc.
    3. Write Actions: automation like sending emails, creating files, etc. (Treat with caution)

### Planning:
- planning should be decoupled from execution to save unnecessary API calls and time. if the plan is validated, we can execute it.
- System Components (3 Agents):
    - Generate plan
    - validate plan
    - execute plan: can be sequential or parallel or using if else or sticking to a loop.
- Planning, at its core, is a search problem: You search among diff paths, predict the outcomes and pick the best one.
    - backtracking: return to the previous point and try a different path.
- Function calling (maybe less relevant now, but still important)
- Fine tuning should provide high level natural language instructions (instead of calling specific functions)
- Reflection is very important: after the user query, the plan, each execution step, and the final output. (ReAct, Reflexion)
- Ablation study: Which tools can be removed?
- If there's any tool the agent always fail to use (you can fine tune on it or add specific instructions for it)

#### Failures:
- Invalid tool: agent tries to use a tool that is not available.
- valid tool with invalid parameters: agent tries to use a tool with invalid parameters.
- valid tool and parameters, but incorrect parameter values
- Analyze agent faliures using the dataset format: (task, tool inventory) review the types of failures for each task.

## Memory
1. Intenral Knowledge: from training data
2. Short term memory: from the current conversation (context like chat history, previous messages, etc.) limited by model context window
3. Long term memory: External data sources (RAG ststem)
- Augmentation with memory systems:
    1. Manage information overflow within a session: Store it in external sources. 
    2. Presistent memory between sessions: (e.g., chatgpt memory for each user)
    3. Boost consistency: when using previous conversations (if I said some opinion or preference I should be consistent with it)
    4. Maintain data structural integrity: use a memory that keeps tables as tables since the model may treat it as just text.
- Memory system is responsible for: Memory management (What to store) and Memory retrieval (what relevant information to retrieve, similar to RAG)
    - Memory management: Add and Delete
        - FIFO: forget the oldest information, keep last N messages (like langchain)
        - More advanced like removing redundancy by summarizing the conversation
        - You can use reflection: after each action, should I insert this information into my memory?
        - How to handle contradicting information?
    - Memory retrieval: Similar to RAG.




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

**Intrinsic Dimension**: the minimum number of independent parameters needed to describe the data, effectively capturing its underlying structure (in optimization theory)

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

# Chapter 8: Dataset Engineering
- Goal: create a dataset that allows you to train the best model
    - What data we need? How Much? What is high quality data?
- different training phases aim to teach the model different capabilities ➡️ require datasets with different attributes
    - Pretraining data: quantity is measured in number of tokens
    - SFT data: number of examples.

## Model Centric AI Vs Data Centric AI
- Model Centric AI: Improve the model itself (arch, size, training techniques, etc.): e.g., training all models on imagenet
- Data Centric AI: new data processing techniques to train the same models

## Data Curation
- **Data curation**: the process of organizing, managing, and maintaining data throughout its lifecycle to ensure it is accurate, accessible, and useful for analysis or research.
- Data you need depends on the task:
    - Self supervised fine tunine: sequences of data
    - instruction fine tuning: (instruction, response) format
    - Preference fine tuning: (instruction, winning response, losing response) format
    - reward model: same preference fine tuning, or annotate them to give scores so it's ((instruction, response), score) format
- it’s even more challenging if you want to teach models complex behaviors: chain of thought, tool use, etc.
- sometimes the human annotators are different e.g., (GUI vs API)
- Conversational data:
    - single turn: respond to an individual instruction
    - multi turn: how to solve task
- removal of unwanted behaviors: e.g., the model produces unsolicited rewriting of the statement.
- if training is cooking: data is the ingredients.

### Data Quality
- 10K carefully crafted instructions are superior to hundreds of thousands of noisy instructions
- definition:
    - short: helps you do your job
    - long:
        1. **Relevant**: training examples should be relevant to the task you’re training the model to do.
        2. **Aligned with task requirements**: e.g., creative, factual, requires some justification, etc.
        3. **Consistent**: different annotators should agree on the same example (e.g., score)
        4. **Correctly Formatted**: e.g., whitespaces, new lines, etc.
        5. **Unique**: duplications can introduce biases and cause data contamination
        6. **Compliant**: With any policies (E.g., no PII data)

### Data Coverage
- Handle all uage patterns
    - long instructions and references Vs short instructions
    - Typos Vs Correct spelling
    - different programming languages
- Math and Code improves model capabilities, but The percentage of code and math data during preference finetuning is much smaller (12.82% combined),

#### How to decide the right data mix?
- reflect the real world usage patterns
- Meta uses something like **Scaling extraploation**: Train small models with different data mixes, then use the best model to train a larger model with the best data mix.
- LIMA: Less Is More for Alignment:
    - Traing 3 7B models on 3 different datasets:
        1. High quality, not diverse
        2. Low quality, diverse
        3. High quality, diverse

### Data Quantity:
- if I have too much data, should I pretrain or fine tune?
    - It can be better to fine tune
    - in some cases, fine tuning is worse due to **ossification**, where pre-training can ossify (i.e., freeze) the model weights so that they don’t adapt as well to the finetuning data

### How much data do I need? Deciding factors
1. Finetuning technique: Tens of thousands: Full fine tuning, hundreds of few thousands: PEFT (e.g., LoRA)
2. Task complexity: e.g., classification vs QA in specialized domain
3. Base model's performance: closer the base model is to the desirable performance, the fewer examples are needed to get there
- Generally: 
    - large amount of data: full finetuning with smaller models
    - small amount of data: PEFT with larger models
- You can always start finetuning with worse data then go to the better: selfsupervised ➡️ supervised, less relevant data ➡️ more relevant data, synthetic data ➡️ real data
- Fine tune the model with subset of your dataset(25%, 50%, etc.), and plot the performance scale (When will I reach plateu)
- Data Diversity:
    - task type: QA, classification, etc.
    - topic: finance, health, etc.
    - output format: JSON, HTML, etc.
- balance between data budget and compute budget

### Data Acquisition & Annotation
- The most important source of data, however, is typically data from your own application.: Application data is ideal because it’s perfectly relevant and aligned with your task. 
- Check availble data and market places (Kaggle, huggingface, govs, open data network) You can also mix them up.
- Annotation is challenging not just because of the annotation process ** d**
- Can a response be correct but unhelpful? What’s the difference between responses that deserve a score of 3 and 4?

## Data Augmentation and Synthesis
- Data augmentation creates new data from existing data (which is real)
- Data synthesis creates new data from scratch (which is synthetic)

### Why Data Synthesis?
- Improve golden data ratio: Quantity, coverage, quality
- Mitigate privacy concerns, or distill models

## Traditional Data Synthesis
- Rule based data synthesis: generate documents that follow a specific structure, such as invoices, resumes, tax forms, bank statements, event agendas.
- Or Augmentation (Generate new data from existing data): Crop, rotate, insert or separate a word etc.
- can be used for bias correction (replace he and she)
- **Perturbation**: Add some noise to the data to make the model more robust.

## AI-Powered Data Synthesis
- Simulation of Calling APIs
- use AI to translate data in high-resource languages (more available online) into low-resource languages to help train models in low-resource languages.
- You can verify the quality of translations with back-translation
- You can also translate programming languages
- synthetic data is intentionally included much more often in post-training than in pre-training **WHY?** since the goal of pretraining is to increase the model’s knowledge, and while AI can synthesize existing knowledge in different formats, it’s harder to synthesize new knowledge. But note that internet now is flooded with AI generated data so it takes its way to the pretraining data.
- **Instruction data synthesis**: you can start with a list of topics, keywords, and/or the instruction types you want in your dataset. and for each item on this list, generate a certain number of instructions. You can even use set of templates.
- **FT to understand longer context** (e.g., 8k to 128k tokens): 
    1. Split long documents into shorter chunks < 8k tokens
    2. For each short cunck, generate several (Q, A) pairs
    3. For each QA Pair, use the original long doc (maybe > 8k but < 128k tokens) as the context. This trains the model to use the extended context.
- **Verification**:
    - people tend to synthesize data they can verify
    - Code is quite easy to verify
    - Some tasks are more difficult to verify, but you can even use AI to verify them
    - Be creative: 
        - if you want synthetic data to mimic real data, its quality can be measured by how difficult it is to distinguish between the two
        - if you want the synthetic data to resemble high-quality academic work, you could train a classifier to predict whether a generated paper would be accepted at a prestigious conference like NeurIPS
- Limitations:
    - 🔴 Quality Ctrl
    - 🔴 Superficial imitation: e.g., mimicking math reasoning without real math knowledge
    - 🔴 **Model Collapse**: Training on too much synthetic data can make the model forget
    - 🔴 **Obscure data lineage**: if you are using a model trained on copyrighted it will find its way into your model. Similar of data contamination and benchmark issues.

## Model Distillation
- The student model can be trained from scratch like DistilBERT or finetuned from a pre-trained model like Alpaca 
- Some models license prohibit distillation.

## Data Processing
- Plot the distribution of tokens (to see what tokens are common), input lengths, response lengths, etc. Does the data use any special tokens? 
- What is duplicate? Whole document, quote?
    - Pairwise comparison: exact match, n-gram, fuzzy, semantic, etc.
    - hashing: Check only inside the bucket
    - dimensionality reduction: then apply pairwise comparison

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

# Chapter 10: AI Engineering Architecture and User Feedback

## AI Engineering Architecture
- simplest form, your application receives a query and sends it to the model.
- Now, improve the architecture step by step as following:

### 1. Enhance context input
- Relevant documents, external tools, web search, etc.
- Context construction is like feature engineering for foundation models. It gives the model the necessary information to produce an output.

### 2. Put in Gaurdrails
- Third-party APIs can reduce the guardrails you need to implement since API providers typically provide many guardrails out of the box for you
- self-hosting means that you don’t need to send requests externally, which reduces the
need for many types of input guardrails.

- Protection against 2 types of risks:
1. Leaking private information: input guardrails
- Examples
    - employee copies company or user's data into a prompt
    - developer puts company policy in the system prompt
    - tool retrieves private information and adds it to the context
- Detect any sensitive information and 
    1. block the whole query
    2. mask it with placeholder [PHONE NUMBER] (you can revert it back after parsing the output)
2. Executing bad prompts (like harmful code): output guardrails

### 3. Model Router and Gateway

#### Router:
- Instead of using one model for all queries, you can have different solutions for different types of queries: maybe better performance - save costs
- A router typically consists of an intent classifier that predicts what the user is trying to do.
- Prevents unrelated queries ➡️ reduces unnecessary API Calls
- Usually very small models (GPT2, LLama 7B or even smaller)

#### Gateway:
- A model gateway is an intermediate layer that allows your organization to interface with different models in a unified and secure manner
- unified interface to different models (APIs, self hosted, etc.) simply, a unified wrapper
- something like MCP

### 4. Reduce Latency with Caching
1. Exact Caching: cached items are used only when these exact items are requested.
    - if something is done before like summarize a product, search in a vector database, etc.
    - Can be implemented in memory or a database
    - eviction policy is crucial to manage the cache size and maintain performance: FIFO, LRU, LFU, etc.
    - 🔴 Caching, when not properly handled, can cause data leaks
2. Semantic Caching: only if semantically similar
    - Semantic caching works only if you have a reliable way of determining if two queries
    are similar.
    - Its success relies on high-quality embeddings, functional vector search, and a reliable similarity metric. 
    - 🔴 semantic cache can be time-consuming and compute-intensive

### 5. Add Agent Patterns
- after the system generates an output, it might determine that it hasn’t accomplished the task and that it needs to perform another retrieval to gather more information.

### Monitoring and Observability
**Monitoring**: Monitor the external outputs to figure out when something goes wrong
**Observability**: Monitor the internal state of the system to figure out when something goes wrong (logs, metrics, etc.)
- Metrics
    1. Mean time to detect (MTTD)
    2. Mean time to respond (MTTR)
    3. Change failure rate (CFR): how often the system fails after a change
- User behaviors as metrics
    - How often do users stop a generation halfway?
    - average number of turns per conversation?
    - average number of tokens per input? Are users using your application for more complex tasks, or are they learning to be more concise with their prompts 
- **It’s useful to measure how these metrics correlate to each other**
- Drift Detection
    - System Prompt Changes
    - User Behavior Changes
    - Model Changes

## User Feedback
- Explicit: ratings, thumbs up/down, etc.
- Implicit: Buying a recommended product, clicking on a link, etc.
- Use feedbacks for: Evaluation, Development (train future models), Personalization (for each user)
- Natural Language Feedback:
    - Early termination
    - Error correction: "No, I meant ... "
    - Complaints: "I'm not happy with the result"
    - Regeneration (from the UI)
    - Conversation length: number of turns