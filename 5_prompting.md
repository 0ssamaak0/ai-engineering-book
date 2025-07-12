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
- Indirect injection: Leaving the injection in something the model can retrieve e.g., public space like github repo or reddit.
- To test against attacks: violation rate and false refusal rate.