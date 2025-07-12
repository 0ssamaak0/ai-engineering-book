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