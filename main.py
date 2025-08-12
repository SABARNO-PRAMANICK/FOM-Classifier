import sys
from groq import Groq  

def classify_with_llm(text: str) -> str:
    try:
        client = Groq()  
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Classify the question as 'factual', 'opinion', or 'math'. Respond with only the label."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        return f"Error: {e}"

def respond(label: str, text: str, use_llm: bool = False) -> str:
    base = f"{label.capitalize()} question detected. "
    if label == "math":
        base += "Outline steps and final answer."
    elif label == "opinion":
        base += "Share balanced perspective and recommendation."
    else:
        base += "Provide concise, sourced facts."
    
    if use_llm:
        try:
            client = Groq()
            llm_resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": f"Respond to this {label} question: {text}"},
                          {"role": "user", "content": text}]
            ).choices[0].message.content
            return f"Groq LLM: {llm_resp}"
        except Exception as e:
            return base + f" (Groq LLM error: {e})"
    return base

def main():
    use_llm_resp = "--llm" in sys.argv
    query = " ".join(a for a in sys.argv[1:] if a != "--llm") or input("Enter question: ").strip()
    label = classify_with_llm(query)
    message = respond(label, query, use_llm_resp)
    print(f"label: {label}\nresponse: {message}")

if __name__ == "__main__":
    main()
