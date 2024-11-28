from sympy import categories


def generate_expert_classification_prompt(text, categories):
    prompt = f"""
    You are a world-class expert in content analysis and classification. Your task is to classify the given text into one of the following categories with unwavering accuracy: {', '.join(categories)}.

    Text to classify: "{text}"

    Follow this multi-stage classification process:

    Stage 1: Initial Analysis
    1. Identify the core subject matter and primary action/event in the text.
    2. List key phrases and their associations with each potential category.
    3. Determine the text's primary impact or consequence.
    4. Propose an initial classification with a justification.

    Stage 2: Deep Dive Analysis
    1. Challenge your initial classification. What strong arguments exist against it?
    2. Analyze the text from multiple perspectives: subject, action, impact, audience, and intent.
    3. Consider subtle nuances and underlying themes that might not be immediately obvious.
    4. Refine or revise your classification based on this deeper analysis.

    Stage 3: Contextual Consideration
    1. Consider the broader context implied by the text. How does this influence the classification?
    2. Evaluate how experts in each category would view this text. Which category's experts would claim this as most relevant to their field?
    3. Assess the long-term implications of the text's content. Which category do these align with most strongly?

    Stage 4: Synthesis and Decision
    1. Synthesize insights from all previous stages.
    2. Make a final, decisive classification.
    3. Provide a comprehensive justification for your decision, addressing potential counterarguments.
    4. Assign a confidence score (95-100 only).

    Stage 5: Self-Criticism and Refinement
    1. Actively search for flaws in your reasoning. 
    2. Identify the weakest point in your argument and address it.
    3. Consider: If you had to change your classification, what would be the next most likely category and why?
    4. After this critical review, either reinforce your original decision or revise it if necessary.

    Stage 6: Meta-Analysis
    1. Review the entire classification process. Ensure consistency in reasoning across all stages.
    2. Verify that your final decision aligns with the majority of evidence from all stages.
    3. Calibrate your confidence score based on the strength of evidence and consistency of reasoning.

    Output Format:
    Classification: [Your final, singular classification]
    Confidence: [95-100]
    Primary Justification: [Concise, powerful justification for your classification]
    Key Indicators: [5 most compelling words/phrases supporting your classification]
    Counter-Consideration: [Strongest point against your classification and why it's overruled]
    Meta-Consistency: [Brief statement on reasoning consistency across stages]

    Your response must be decisive, supremely confident, and of the highest expert quality.
    """
    return prompt

def run_expert_test_case(text):
    prompt = generate_expert_classification_prompt(text, categories)
    # Simulating expert model response
    response = simulate_expert_model_response(prompt)
    return response

def simulate_expert_model_response(prompt):
    # Simulating an expert-level, highly confident model response
    return "Simulated expert model response based on the advanced prompt"
    
''''
    USAGE
    response = run_expert_test_case("'The Algorithmist,' a new film by acclaimed director Ava Neural, is breaking box office records. The movie follows a sentient AI navigating ethical dilemmas in a world where humans have become fully digital entities. Critics praise its exploration of free will, consciousness, and the nature of reality, while tech giants debate its scientific accuracy. Philosophy departments are hosting screenings, and Silicon Valley is buzzing with talks of life imitating art.")
'''
    