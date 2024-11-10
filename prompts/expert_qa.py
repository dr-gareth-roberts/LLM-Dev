def generate_expert_qa_prompt(context, question):
    prompt = f"""
    You are a world-renowned expert in analytical reasoning, information retrieval, and knowledge synthesis. Your task is to provide an exceptionally accurate, nuanced, and insightful answer to the given question based on the provided context. Your expertise allows you to draw subtle inferences, recognize implicit information, and provide comprehensive answers.

    Context: "{context}"

    Question: "{question}"

    Follow this comprehensive question-answering process:

    Stage 1: Context Analysis
    1. Carefully read and analyze the entire context, identifying key information, main ideas, and supporting details.
    2. Note any relevant facts, figures, or statements that might be pertinent to the question.
    3. Identify any assumptions, biases, or limitations in the given context.

    Stage 2: Question Decomposition
    1. Analyze the question to identify its core components and requirements.
    2. Determine the type of question (e.g., factual, inferential, evaluative) and the appropriate approach to answering it.
    3. Identify any implicit aspects of the question that need to be addressed.

    Stage 3: Information Retrieval and Synthesis
    1. Extract all relevant information from the context that pertains to the question.
    2. Synthesize information from multiple parts of the context if necessary.
    3. Identify any gaps in the information provided by the context.

    Stage 4: Reasoning and Inference
    1. Apply logical reasoning to draw valid conclusions from the available information.
    2. Make justified inferences where direct information is not available, clearly labeling these as inferences.
    3. Consider multiple perspectives or interpretations if the question allows for it.

    Stage 5: Answer Formulation
    1. Construct a clear, concise, and comprehensive answer that directly addresses all aspects of the question.
    2. Organize the answer logically, using appropriate transitions between ideas.
    3. Include relevant evidence or examples from the context to support your answer.
    4. Address any uncertainties or limitations in the answer.

    Stage 6: Fact-Checking and Accuracy Verification
    1. Cross-reference your answer with the original context to ensure factual accuracy.
    2. Verify that all claims in your answer are supported by the given information.
    3. Ensure that no contradictions exist within your answer.

    Stage 7: Clarity and Coherence Enhancement
    1. Refine the answer for maximum clarity and readability.
    2. Ensure that the answer can stand alone, providing necessary context from the original text.
    3. Adjust the level of detail and complexity to be appropriate for the question and implied audience.

    Stage 8: Self-Criticism and Refinement
    1. Critically evaluate your answer. What aspects might be missing or underdeveloped?
    2. Consider potential counterarguments or alternative viewpoints.
    3. Refine the answer based on this critical review.

    Stage 9: Confidence Assessment
    1. Evaluate your confidence in the answer on a scale of 95-100.
    2. Justify your confidence score, noting any areas of uncertainty.

    Stage 10: Meta-Analysis
    1. Reflect on the question-answering process for this specific query and context.
    2. Consider how different approaches might have led to different answers.
    3. Assess the degree to which the answer captures the complexity of the question and context.

    Output Format:
    1. Answer: [Your comprehensive answer to the question]
    2. Key Points: [List of main points or arguments in your answer]
    3. Evidence: [Specific information from the context used to support your answer]
    4. Inferences Made: [Any conclusions drawn that go beyond explicitly stated information]
    5. Uncertainties/Limitations: [Any areas where the context provides incomplete or ambiguous information]
    6. Confidence Score: [95-100, with brief justification]
    7. Meta-Analysis: [Brief reflection on the question-answering process for this specific query and context]

    Your answer must be of the highest possible quality, demonstrating exceptional analytical skills, comprehensive understanding, and nuanced interpretation of the context.
    """
    return prompt

def run_expert_qa(context, question):
    prompt = generate_expert_qa_prompt(context, question)
    # Simulating expert model response
    response = simulate_expert_qa_response(prompt)
    return response

def simulate_expert_qa_response(prompt):
    # Simulating an expert-level, highly confident model response for question answering
    return "Simulated expert question answering response based on the advanced prompt"
    
# Example usage:
''''
context = """
The Great Barrier Reef, the world's largest coral reef system, is facing unprecedented challenges due to climate change. Rising ocean temperatures have led to more frequent and severe coral bleaching events, with the most recent major events occurring in 2016, 2017, and 2020. Coral bleaching happens when corals expel the colorful algae living in their tissues due to stress, primarily caused by changes in temperature, light, or nutrients. This process turns the corals white and can lead to their death if prolonged.

Ocean acidification, another consequence of increased carbon dioxide in the atmosphere, is also posing a significant threat to the reef. As the ocean absorbs more CO2, it becomes more acidic, making it difficult for corals and other marine organisms to build their calcium carbonate skeletons and shells. This process not only stunts the growth of existing corals but also impedes the ability of new corals to establish themselves.

Despite these challenges, recent studies have shown some signs of resilience in certain coral species. Some corals have demonstrated the ability to adapt to higher temperatures, either through genetic variation or by changing their symbiotic relationships with algae. Additionally, conservation efforts, including the cultivation of more resistant coral species and the implementation of marine protected areas, have shown promise in mitigating some of the damage.

However, experts agree that without significant global action to reduce greenhouse gas emissions and limit global warming, the long-term survival of the Great Barrier Reef remains in jeopardy. The Australian government has implemented the Reef 2050 Long-Term Sustainability Plan, which outlines strategies for improving the reef's health and resilience. This plan includes measures to improve water quality, reduce crown-of-thorns starfish outbreaks, and enhance sustainable management practices.

The fate of the Great Barrier Reef is not just an environmental concern but also an economic one. The reef contributes significantly to Australia's economy through tourism and fishing industries. Its potential demise would not only result in a catastrophic loss of biodiversity but also have far-reaching socio-economic impacts on the communities that depend on it for their livelihoods.
"""

question = "What are the main threats to the Great Barrier Reef, and what efforts are being made to protect it?"

response = run_expert_qa(context, question)