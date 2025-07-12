# Chapter 2: Understanding Foundation Models
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
- the number of training tokens to be approximately 20 times the model size. This means that a 3B-parameter model needs approximately 60B training tokens ➡️ Doubling the model size requires doubling the number of training tokens.

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