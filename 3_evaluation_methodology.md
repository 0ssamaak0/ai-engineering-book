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