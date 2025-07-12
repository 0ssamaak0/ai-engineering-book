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