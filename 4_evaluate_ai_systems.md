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
    - SAFE: decompose response into individual statements (and make them self contained) then check each statement.
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
