def generate_expert_sentiment_analysis_prompt(text):
    prompt = f"""
    You are a world-renowned expert in linguistic analysis, psychology, and sentiment detection. Your task is to perform an exhaustive sentiment analysis on the given text, providing unparalleled insight into its emotional tone, underlying attitudes, and implicit biases.

    Text for analysis: "{text}"

    Follow this comprehensive sentiment analysis process:

    Stage 1: Initial Sentiment Assessment
    1. Carefully read the entire text, noting your immediate emotional response.
    2. Identify explicit sentiment indicators (e.g., emotion words, tone indicators).
    3. Propose an initial sentiment classification (Positive, Negative, Neutral, or Mixed) with a justification.

    Stage 2: Linguistic Deep Dive
    1. Analyze lexical choices, focusing on connotations and emotional weight of key terms.
    2. Examine syntactic structures and their impact on sentiment (e.g., passive voice, rhetorical questions).
    3. Identify and interpret figurative language (e.g., metaphors, similes, irony) and its sentiment implications.
    4. Assess the impact of intensifiers, hedges, and qualifiers on sentiment strength.

    Stage 3: Contextual and Pragmatic Analysis
    1. Consider the broader context implied by the text. How does this influence sentiment interpretation?
    2. Analyze pragmatic features (e.g., presuppositions, implicatures) and their sentiment effects.
    3. Identify any cultural or domain-specific references that might affect sentiment interpretation.

    Stage 4: Subjectivity and Objectivity Assessment
    1. Differentiate between factual statements and opinion-based content.
    2. Analyze the author's level of certainty and commitment to expressed views.
    3. Identify any attempts at objective presentation and their impact on overall sentiment.

    Stage 5: Multi-dimensional Sentiment Mapping
    1. Plot the sentiment along multiple dimensions (e.g., Positive-Negative, Arousal-Calm, Dominant-Submissive).
    2. Identify any shifts in sentiment throughout the text, noting potential causes.
    3. Assess the overall sentiment trajectory and its implications.

    Stage 6: Target-specific Sentiment Analysis
    1. Identify all sentiment targets (e.g., people, objects, ideas) mentioned in the text.
    2. Analyze sentiment specifically associated with each target.
    3. Compare and contrast sentiments towards different targets.

    Stage 7: Implicit Bias and Undertone Analysis
    1. Look for subtle indicators of bias or unexpressed attitudes.
    2. Analyze potential discrepancies between explicit and implicit sentiment.
    3. Consider how different audiences might interpret the sentiment differently.

    Stage 8: Confidence Assessment and Self-Criticism
    1. Assign a confidence score (95-100) to your overall sentiment analysis.
    2. Identify aspects of the text that were particularly challenging for sentiment analysis.
    3. Consider alternative interpretations. How might other experts disagree with your analysis?
    4. Refine your analysis based on this critical review.

    Stage 9: Meta-Analysis
    1. Reflect on the overall sentiment analysis process for this specific text.
    2. Consider how different sentiment analysis systems or human annotators might interpret this text.
    3. Discuss any unique challenges this text posed for sentiment analysis and how you addressed them.

    Output Format:
    1. Overall Sentiment: [Classification with fine-grained description]
    2. Confidence: [95-100]
    3. Sentiment Breakdown:
       - Explicit Sentiment: [Description with key indicators]
       - Implicit Sentiment: [Description with subtle cues]
       - Target-specific Sentiments: [List of targets and associated sentiments]
    4. Linguistic Analysis: [Key linguistic features affecting sentiment]
    5. Contextual Factors: [Relevant contextual elements influencing sentiment interpretation]
    6. Sentiment Trajectory: [Description of how sentiment evolves through the text]
    7. Multi-dimensional Analysis: [Plot on Positive-Negative, Arousal-Calm, Dominant-Submissive scales]
    8. Challenging Aspects: [Difficult elements for sentiment analysis in this text]
    9. Potential Alternative Interpretations: [Brief description of how the sentiment could be interpreted differently]
    10. Meta-Analysis: [Reflection on the sentiment analysis process for this specific text]

    Your analysis must be exhaustive, nuanced, and of the highest expert quality. Capture ALL aspects of sentiment, no matter how subtle or complex.
    """
    return prompt

def run_expert_sentiment_analysis(text):
    prompt = generate_expert_sentiment_analysis_prompt(text)
    # Simulating expert model response
    response = simulate_expert_sentiment_analysis_response(prompt)
    return response

def simulate_expert_sentiment_analysis_response(prompt):
    # Simulating an expert-level, highly confident model response for sentiment analysis
    return "Simulated expert sentiment analysis response based on the advanced prompt"
    
''''
EXAMPLE USE

test_text = """
The government's new environmental policy, unveiled yesterday, has been met with a mix of cautious optimism and skepticism. Proponents argue that it's a step in the right direction, praising its ambitious targets for reducing carbon emissions. "It's about time we took decisive action," said Sarah Green, an environmental activist. However, critics point out potential loopholes and the lack of concrete enforcement mechanisms. Industry leaders warn of potential job losses, while some economists suggest the policy might actually stimulate green job growth. As citizens, we're left wondering: is this truly the environmental breakthrough we've been waiting for, or just another political maneuver?
"""

response = run_expert_sentiment_analysis(test_text)

=====================OUTPUT======================
1. Overall Sentiment: Mixed with a slight negative lean
The text presents a balanced view of a new environmental policy, acknowledging both positive and negative perspectives. 
However, the concluding question introduces a subtle negative bias, suggesting skepticism towards the policy's effectiveness.

2. Confidence: 97

3. Sentiment Breakdown:
   - Explicit Sentiment: Balanced with both positive ("cautious optimism", "step in the right direction", "ambitious targets") and negative ("skepticism", "loopholes", "lack of concrete enforcement") elements.
   - Implicit Sentiment: Slightly negative, implied by the structure of the text which ends with a skeptical question, leaving readers with a final impression of doubt.
   - Target-specific Sentiments:
     * Environmental Policy: Mixed (positive for ambition, negative for potential ineffectiveness)
     * Government: Slightly negative (implied criticism for potential "political maneuver")
     * Proponents: Positive
     * Critics: Negative towards the policy
     * Industry Leaders: Negative (concerned about job losses)
     * Economists: Cautiously positive (potential for green job growth)
     * Citizens: Uncertain, leaning negative (implied by the final questioning tone)

4. Linguistic Analysis:
   - Use of contrastive conjunctions ("However", "while") to balance perspectives
   - Quotation marks around "It's about time we took decisive action" emphasize the positive view
   - Use of questioning at the end introduces doubt and shifts the overall tone
   - Hedging language ("cautious optimism", "potential loopholes") indicates uncertainty

5. Contextual Factors:
   - The text assumes background knowledge of environmental policies and their typical challenges
   - The mention of job losses vs. green job growth reflects broader economic concerns related to environmental policies

6. Sentiment Trajectory:
   Starts neutral → Moves to positive → Shifts to negative → Ends with uncertainty leaning negative

7. Multi-dimensional Analysis:
   - Positive-Negative: Slightly negative (+40 on a -100 to +100 scale)
   - Arousal-Calm: Moderate arousal (+30 on a -100 to +100 scale)
   - Dominant-Submissive: Slightly submissive (-20 on a -100 to +100 scale)

8. Challenging Aspects:
   - Balancing multiple perspectives and stakeholder sentiments
   - Interpreting the impact of the final rhetorical question on overall sentiment
   - Distinguishing between reported sentiments and the author's implicit sentiment

9. Potential Alternative Interpretations:
   - The text could be seen as purely neutral, presenting balanced viewpoints without bias
   - The concluding question could be interpreted as promoting critical thinking rather than implying skepticism

10. Meta-Analysis:
    This text presents a complex sentiment analysis challenge due to its multi-stakeholder perspectives and the subtle shift in tone throughout. 
    The use of reported speech and the final rhetorical question add layers of complexity, requiring careful consideration of explicit statements versus implicit sentiment. 
    The analysis process involved weighing the impact of linguistic features against the overall structure and concluding tone of the text.

This sentiment analysis demonstrates the text's complexity, balancing multiple viewpoints while subtly leaning towards a skeptical, slightly negative overall sentiment. 
The high confidence score reflects the comprehensive analysis process, while acknowledging the potential for alternative interpretations.
'''