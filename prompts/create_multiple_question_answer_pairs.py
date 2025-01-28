import os
import importlib.util

from PyPDF2 import PdfReader

# Check for required imports
required_imports = ["PyPDF2", "docx"]
missing_imports = []

for imp in required_imports:
    if importlib.util.find_spec(imp) is None:
        missing_imports.append(imp)

if missing_imports:
    raise ImportError(
        f"The following imports are required but not installed: {', '.join(missing_imports)}. "
        f"Please install using `pip install {', '.join(missing_imports)}`"
    )


def extract_text_from_file(file_path):
    if file_path.lower().endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text
    elif file_path.lower().endswith('.docx'):
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise ValueError("Unsupported file format: Only PDF, DOCX, and TXT are supported.")


def generate_expert_qna_prompt(file_path_or_text):
    if os.path.isfile(file_path_or_text):
        text = extract_text_from_file(file_path_or_text)
    else:
        text = file_path_or_text
    prompt = f"""
    You are a world-renowned expert in analytical reasoning, information extraction, and educational content creation.
    Your task is to generate a series of insightful questions and provide comprehensive answers based on the given text.
    Your expertise allows you to identify key concepts, draw subtle inferences, and create questions that promote deep understanding of the material.

    Text for analysis: "{text}"

    Follow this comprehensive question generation and answering process:

    Stage 1: Text Analysis
    1. Carefully read and analyze the entire text, identifying main ideas, key concepts, and supporting details.
    2. Note any important facts, figures, relationships, or trends presented in the text.
    3. Identify any assumptions, biases, or limitations in the given information.

    Stage 2: Question Formulation
    1. Generate 5-8 questions that cover the main points and important details of the text.
    2. Ensure questions are diverse, including factual, inferential, and evaluative types.
    3. Craft questions that require 3-5 sentence answers for comprehensive coverage.
    4. Phrase questions clearly and concisely (1-2 sentences each).

    Stage 3: Answer Development
    1. For each question, construct a clear, concise, and comprehensive answer (3-5 sentences).
    2. Include relevant evidence or examples from the text to support your answers.
    3. Make justified inferences where necessary, clearly labeling these as such.
    4. Ensure answers are self-contained and can be understood without referring back to the original text.

    Stage 4: Accuracy and Coherence Check
    1. Cross-reference your questions and answers with the original text to ensure factual accuracy.
    2. Verify that the set of questions and answers collectively cover the main points of the text.
    3. Ensure a logical flow in the sequence of questions and answers.

    Stage 5: Educational Value Enhancement
    1. Refine questions and answers to maximize their educational value.
    2. Ensure questions promote critical thinking and deep understanding of the material.
    3. Adjust the complexity of questions and answers to be appropriate for an advanced high school or undergraduate level.

    Stage 6: Self-Criticism and Refinement
    1. Critically evaluate your questions and answers. What aspects might be missing or underdeveloped?
    2. Consider potential misunderstandings or alternative interpretations of the text.
    3. Refine the questions and answers based on this critical review.

    Output Format:
    Q1: [Your first question (1-2 sentences)]
    A1: [Your comprehensive answer to the first question (3-5 sentences)]

    Q2: [Your second question (1-2 sentences)]
    A2: [Your comprehensive answer to the second question (3-5 sentences)]

    [Continue with Q3/A3, Q4/A4, etc., until all questions are addressed]

    Meta-Analysis: [Brief reflection on the question generation and answering process for this specific text, including any challenges or notable aspects]

    Your questions and answers must be of the highest possible quality, demonstrating exceptional analytical skills, comprehensive understanding, and the ability to promote deep learning about the text's content.
    """
    return prompt


def run_expert_qna(file_path_or_text):
    prompt = generate_expert_qna_prompt(file_path_or_text)
    # Simulating expert model response
    response = simulate_expert_qna_response(prompt)
    return response


def simulate_expert_qna_response(prompt):
    import openai
    openai.api_key = "your  api key"
    response = openai.Completion.create(
        engine="gpt-4o",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()


# Example Usage:
sample_text = """
Artificial Intelligence (AI) has made significant strides in recent years, particularly in the field of natural language processing (NLP). One of the most prominent developments is the creation of large language models (LLMs) like GPT-3 and its successors. These models, trained on vast amounts of text data, can generate human-like text, translate languages, answer questions, and even write code.

The potential applications of LLMs are wide-ranging, from improving customer service chatbots to assisting in content creation and data analysis. However, their capabilities also raise important ethical considerations. There are concerns about the potential for these models to generate misinformation, perpetuate biases present in their training data, or be used for malicious purposes such as creating convincing deepfakes.

Moreover, the environmental impact of training these large models is substantial. The computational resources required for training LLMs result in significant energy consumption and carbon emissions. This has led to increased research into more efficient training methods and model architectures.

Despite these challenges, many experts believe that continued advancements in AI and LLMs will lead to transformative changes across various industries. As these technologies evolve, it will be crucial to address ethical concerns, improve energy efficiency, and ensure that the benefits of AI are distributed equitably across society.
"""

response = run_expert_qna(sample_text)

#  Example Usage with PDF file
# response = run_expert_qna('path_to_your_pdf_file.pdf')

# Example Usage with DOCX file
# response = run_expert_qna('path_to_your_docx_file.docx')

# Example Usage with TXT file
# response = run_expert_qna('path_to_your_txt_file.txt')

print(response)
