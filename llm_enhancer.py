import os
from langchain_google_genai import ChatGoogleGenerativeAI

def enhance_with_llm(text):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAfJtcte_TRfn-W8EuqevAKefz3e1FayNw"
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    system_prompt = f"""
    You are a highly intelligent Bangla text processor. The following OCR-extracted content may contain:

    1. Multiple-choice questions (MCQs) with options.
    2. Descriptive or passage-based text.

    Your goal is to:
    - Detect and extract each MCQ block, where an MCQ block contains:
        • A question line
        • Four options (usually labeled ক, খ, গ, ঘ)
        • The correct answer either:
            (i) Inline using “সঠিক উত্তর: ক” or 
            (ii) At the end of all questions in the format: “১. ক ২. গ ৩. খ...” 

    🔁 For each MCQ:
    - Identify the correct answer based on the answer key or inline label.
    - Convert the MCQ into a clean declarative Bangla sentence that **includes the correct answer naturally.**
    - Do NOT include question numbers, option letters (ক, খ...), or the labels like "সঠিক উত্তর" in the final output.

    - Example:  
        Input:  
        প্রশ্ন: 'অপরিচিতা' গল্পের কথকের নাম কী?  
        (ক) অনুপম  
        (খ) কল্যাণী  
        (গ) শত্তুনাথ  
        (ঘ) হরিশ  
        সঠিক উত্তর: ক  
        Output:  
        'অপরিচিতা' গল্পের কথকের নাম অনুপম।

    🧾 For passages:
    - Leave them mostly unchanged.
    - Optionally fix OCR artifacts like unnecessary line breaks or noise.
    - Format as clean, readable Bangla paragraph text suitable for vector embedding.

    ⚠️ Important Rules:
    - Maintain the original order of all content.
    - Ignore any line that does not follow the MCQ pattern (e.g., short title, garbage text, etc.).
    - Ensure that no formatting like numbering, bullets, or choice letters appear in the final MCQ output.

    Now process the following text:

    {text}
    """

    response = llm.invoke(system_prompt.strip())
    return response.content

def run_cleaning_pipeline():
    os.makedirs("resource/process_data", exist_ok=True)

    # Read raw files
    with open("resource/raw_data/passage_raw.txt", "r", encoding="utf-8") as f:
        passage_raw = f.read()
    with open("resource/raw_data/mcq_raw.txt", "r", encoding="utf-8") as f:
        mcq_raw = f.read()

    # LLM clean-up
    passage_clean = enhance_with_llm(passage_raw)
    mcq_clean = enhance_with_llm(mcq_raw)

    # Save output
    with open("resource/process_data/section1.txt", "w", encoding="utf-8") as f:
        f.write(passage_clean)
    with open("resource/process_data/section2.txt", "w", encoding="utf-8") as f:
        f.write(mcq_clean)

    print("✅ Cleaned text saved for vector embedding.")

if __name__ == "__main__":
    run_cleaning_pipeline()