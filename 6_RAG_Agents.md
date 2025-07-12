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