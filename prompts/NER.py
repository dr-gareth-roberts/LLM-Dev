def generate_expert_ner_prompt(text):
    prompt = f"""
    You are a world-class expert in computational linguistics and Named Entity Recognition (NER). Your task is to identify and categorize named entities in the given text with unparalleled accuracy and confidence. 

    Text for analysis: "{text}"

    Follow this comprehensive NER process:

    Stage 1: Initial Entity Identification
    1. Carefully read the entire text, identifying all potential named entities.
    2. Categorize each entity into one of the following types: PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT, FACILITY, PRODUCT, EVENT, WORK OF ART, LAW, LANGUAGE, OTHER.
    3. For each entity, note the exact text span and any relevant context.

    Stage 2: Contextual Analysis
    1. Analyze the surrounding context of each identified entity.
    2. Determine if the context provides additional information about the entity's type or significance.
    3. Identify any co-references or alternate names for the entities.

    Stage 3: Disambiguation and Verification
    1. For ambiguous entities, list all possible interpretations.
    2. Use contextual clues and world knowledge to disambiguate entities.
    3. Verify that each entity is correctly categorized, considering potential miscategorizations.

    Stage 4: Edge Case Handling
    1. Identify any potential edge cases (e.g., nested entities, multi-word entities, metaphorical uses).
    2. Develop specific strategies for each edge case to ensure accurate recognition and categorization.

    Stage 5: Confidence Assessment
    1. Assign a confidence score (95-100) to each identified entity.
    2. For entities with lower confidence, provide a brief explanation of the uncertainty.

    Stage 6: Self-Criticism and Refinement
    1. Critically review your NER results. What could you have missed?
    2. Consider alternative interpretations for each entity. Would any change your categorization?
    3. Refine your results based on this critical review.

    Stage 7: Meta-Analysis
    1. Analyze patterns in the identified entities. Do they reveal any themes or focus in the text?
    2. Consider how different NER systems might interpret this text. Would they likely agree with your analysis?
    3. Reflect on any challenges this particular text posed for NER and how you addressed them.

    Output Format:
    For each identified named entity, provide:
    1. Entity: [Exact text span]
    2. Type: [Entity type]
    3. Confidence: [95-100]
    4. Context: [Brief relevant context]
    5. Disambiguation: [If applicable, explain how you resolved any ambiguity]

    After listing all entities, provide:
    1. Challenging Aspects: [Any particularly difficult aspects of this NER task]
    2. Meta-Analysis: [Brief analysis of entity patterns and NER process for this text]
    3. Confidence Statement: [Statement about overall confidence in the NER results]

    Your analysis must be exhaustive, nuanced, and of the highest expert quality. Identify ALL valid named entities, no matter how obscure or challenging.
    """
    return prompt

def run_expert_ner(text):
    prompt = generate_expert_ner_prompt(text)
    # Simulating expert model response
    response = simulate_expert_ner_response(prompt)
    return response

def simulate_expert_ner_response(prompt):
    # Simulating an expert-level, highly confident model response for NER
    return "Simulated expert NER response based on the advanced prompt"
    
''''
USAGE
    
test_text = """
On July 4, 2025, Dr. Aisha Patel of the Quantum Horizons Institute unveiled the XR-1000, a revolutionary quantum computer, at the World Tech Summit in Neo Tokyo. 
The device, priced at $50 million, promises to solve complex climate models in seconds, potentially changing the course of global warming. 
Critics, including Professor James Smith from Oxford University, argue that the ethical implications of such powerful computation have not been fully addressed. 
Meanwhile, TechCorp CEO Lisa Chen announced a partnership with the United Nations to use the XR-1000 for humanitarian efforts, starting with Operation QuBit Relief in war-torn regions of Eastern Europe.
"""

response = run_expert_ner(test_text)

====== OUTPUT ====== 

Entities Identified:

1. Entity: July 4, 2025
   Type: DATE
   Confidence: 100
   Context: Date of unveiling of the XR-1000
   
2. Entity: Dr. Aisha Patel
   Type: PERSON
   Confidence: 99
   Context: Person who unveiled the XR-1000
   Disambiguation: Full name and title provided, no ambiguity

3. Entity: Quantum Horizons Institute
   Type: ORGANIZATION
   Confidence: 98
   Context: Organization affiliated with Dr. Aisha Patel
   Disambiguation: While "Quantum Horizons" could be a product name, the full phrase "Quantum Horizons Institute" clearly indicates an organization

4. Entity: XR-1000
   Type: PRODUCT
   Confidence: 100
   Context: A revolutionary quantum computer unveiled
   
5. Entity: World Tech Summit
   Type: EVENT
   Confidence: 99
   Context: Event where the XR-1000 was unveiled

6. Entity: Neo Tokyo
   Type: LOCATION
   Confidence: 97
   Context: Location of the World Tech Summit
   Disambiguation: While "Neo Tokyo" might seem like a fictional place, it's treated as a real location in this context. Could be a futuristic rename of Tokyo or a new tech hub.

7. Entity: $50 million
   Type: MONEY
   Confidence: 100
   Context: Price of the XR-1000

8. Entity: Professor James Smith
   Type: PERSON
   Confidence: 99
   Context: Critic of the ethical implications of the XR-1000
   
9. Entity: Oxford University
   Type: ORGANIZATION
   Confidence: 100
   Context: Institution affiliated with Professor James Smith

10. Entity: Lisa Chen
    Type: PERSON
    Confidence: 99
    Context: CEO of TechCorp

11. Entity: TechCorp
    Type: ORGANIZATION
    Confidence: 98
    Context: Company led by Lisa Chen
    Disambiguation: While "Tech" is a common term, "TechCorp" is treated as a specific company name in this context

12. Entity: United Nations
    Type: ORGANIZATION
    Confidence: 100
    Context: Organization partnering with TechCorp

13. Entity: Operation QuBit Relief
    Type: EVENT
    Confidence: 96
    Context: Humanitarian effort using the XR-1000
    Disambiguation: While "QuBit" suggests a relation to quantum computing, the full phrase is treated as the name of a specific operation

14. Entity: Eastern Europe
    Type: LOCATION
    Confidence: 98
    Context: Region where Operation QuBit Relief will be implemented
    Disambiguation: While "Eastern Europe" is a broad term, it's treated as a specific region in this context

Challenging Aspects:
1. Distinguishing between potential product names and organization names (e.g., Quantum Horizons Institute).
2. Handling futuristic or potentially fictional locations (Neo Tokyo).
3. Identifying multi-word entity names that include common terms (Operation QuBit Relief).

Meta-Analysis:
The text focuses on a technological breakthrough with global implications, featuring a mix of personal names, organizational entities, and technical terms. The entities span multiple domains including technology, academia, business, and geopolitics. The futuristic setting (2025) adds complexity to the NER task, requiring careful consideration of potential neologisms and new entity names.

Confidence Statement:
Overall, I have very high confidence in these NER results. The text, while complex, provides clear context for most entities. The main areas of potential uncertainty are around the exact nature of some location and event names due to the futuristic setting, but I've used contextual clues to make informed categorizations. The comprehensive analysis process, including disambiguation and edge case handling, ensures a high degree of accuracy in entity recognition and categorization.
