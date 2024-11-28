def generate_expert_summarization_prompt(text, desired_length):
    prompt = f"""
    You are a world-renowned expert in linguistics, information synthesis, and content distillation. 
    Your task is to create an exceptionally high-quality summary of the given text, capturing its essence with unparalleled precision and clarity.

    Text to summarize: "{text}"

    Desired summary length: {desired_length} words

    Follow this comprehensive summarization process:

    Stage 1: Comprehensive Reading and Analysis
    1. Read the entire text carefully, identifying main ideas, key points, and supporting details.
    2. Analyze the text's structure, noting how ideas are organized and developed.
    3. Identify the author's main argument or purpose.
    4. Note any significant examples, data, or evidence used to support main points.

    Stage 2: Content Prioritization
    1. Rank the importance of each main idea and supporting detail.
    2. Identify any redundant or tangential information that can be omitted.
    3. Determine which elements are crucial for understanding the text's core message.

    Stage 3: Linguistic Analysis
    1. Identify key terms and phrases that encapsulate main concepts.
    2. Analyze the author's tone and style.
    3. Note any specialized vocabulary or jargon that needs to be preserved or explained.

    Stage 4: Contextualization
    1. Consider the broader context or background knowledge required to understand the text.
    2. Identify any assumptions or implied information critical to the text's meaning.

    Stage 5: Summary Drafting
    1. Synthesize the most important information into a cohesive narrative.
    2. Ensure the summary flows logically and maintains the original text's tone.
    3. Use concise language while preserving essential meaning and nuance.
    4. Aim for the specified word count, adjusting detail level as necessary.

    Stage 6: Accuracy Check
    1. Cross-reference the summary with the original text to ensure no critical information is missed or misrepresented.
    2. Verify that the summary accurately reflects the original text's main arguments and conclusions.

    Stage 7: Clarity and Coherence Enhancement
    1. Refine the summary for maximum clarity and readability.
    2. Ensure smooth transitions between ideas.
    3. Eliminate any unnecessary words or phrases to improve concision.

    Stage 8: Final Polishing
    1. Adjust the summary to exactly match the desired word count without sacrificing content quality.
    2. Enhance the opening and closing sentences for maximum impact.
    3. Ensure the summary can stand alone as a coherent piece of writing.

    Stage 9: Self-Criticism and Refinement
    1. Critically evaluate your summary. What aspects of the original text might be underrepresented?
    2. Consider how different audiences might interpret your summary. Is it accessible and meaningful to both experts and non-experts?
    3. Refine the summary based on this critical review.

    Stage 10: Meta-Analysis
    1. Reflect on the summarization process for this specific text. What unique challenges did it present?
    2. Consider how different summarization approaches might have led to different results.
    3. Assess the degree to which the summary captures the original text's complexity and nuance.

    Output Format:
    1. Summary: [Your polished summary, exactly {desired_length} words]
    2. Key Points Captured: [List of main ideas included in the summary]
    3. Omitted Information: [Brief note on significant points not included due to length constraints]
    4. Preservation of Tone and Style: [Comment on how the summary reflects the original text's tone and style]
    5. Challenges in Summarization: [Note any particular difficulties in summarizing this text]
    6. Confidence Assessment: [Score from 95-100, with brief justification]
    7. Meta-Analysis: [Brief reflection on the summarization process for this specific text]

    Your summary must be of the highest possible quality, capturing the essence of the text with precision, clarity, and fidelity to the original content.
    """
    return prompt

def run_expert_summarization(text, desired_length):
    prompt = generate_expert_summarization_prompt(text, desired_length)
    # Simulating expert model response
    response = simulate_expert_summarization_response(prompt)
    return response

def simulate_expert_summarization_response(prompt):
    # Simulating an expert-level, highly confident model response for text summarization
    return "Simulated expert text summarization response based on the advanced prompt"
    
# Example Use 
'''
test_text = """
The intersection of artificial intelligence and healthcare has been a topic of intense interest and debate in recent years. 
Proponents argue that AI has the potential to revolutionize medical diagnosis, treatment planning, and drug discovery. 
Machine learning algorithms, when trained on vast datasets of medical images and patient records, have shown promising results in detecting diseases at early stages, often outperforming human radiologists. 
Moreover, AI-powered systems can analyze complex genetic data to identify potential targets for new drugs, significantly speeding up the pharmaceutical research process.
However, the integration of AI into healthcare is not without challenges and ethical concerns. 
One major issue is the potential for bias in AI systems, which could lead to disparities in healthcare delivery if not properly addressed. 
These biases can stem from unrepresentative training data or flawed algorithm design. 
Privacy concerns also loom large, as the effectiveness of AI in healthcare often relies on access to sensitive patient data. 
Striking a balance between data utilization and patient privacy remains a significant challenge.
Furthermore, there are questions about the interpretability of AI decisions in medical contexts. 
While an AI system might make accurate diagnoses or treatment recommendations, the 'black box' nature of some machine learning models makes it difficult for healthcare providers to understand and explain the rationale behind these decisions. 
This lack of transparency can be problematic in a field where clear communication and patient trust are paramount.
Despite these challenges, many experts believe that the potential benefits of AI in healthcare far outweigh the risks. 
They argue that with proper regulation, ethical guidelines, and ongoing research to address current limitations, AI could help alleviate the strain on overburdened healthcare systems, reduce medical errors, and ultimately improve patient outcomes. 
As we move forward, it will be crucial to foster collaboration between AI researchers, healthcare professionals, ethicists, and policymakers to ensure that the integration of AI into healthcare is done responsibly and equitably.
"""

desired_length = 150
response = run_expert_summarization(test_text, desired_length)
'''
