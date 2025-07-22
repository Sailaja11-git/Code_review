import re
import traceback
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def generate_review(code):
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    prompt = f"""
Review the following Python code:

```python
{code}

You are an expert Python code reviewer. Review the following code thoroughly:

1. Detect and explain any syntax errors with line numbers.
2. Explain the purpose and logic of the code.
3. Identify bugs, logical errors, and edge cases that may break the code.
4. Suggest improvements related to:
   - Readability
   - Performance
   - Style (PEP8)
   - Error handling
   - Input validation
5. Add proper docstrings to all functions and classes.
6. Rewrite the code with all suggested improvements and fixes.
7. Generate Python unit tests using `unittest` to cover normal and edge cases.

Return your response in the following format:

---
**Syntax Errors (if any):**

[List syntax issues with line numbers and descriptions]

---
**Code Explanation:**

[Describe what the code does]

---
**Issues / Bugs:**

[List all problems found]

---
**Suggestions & Improvements:**

[Recommend fixes or enhancements]

---
**Improved Code:**

```python
# [Cleaned, improved, fully functional code]

End your response clearly without repeating sections or extraneous text.
"""
    response = generator(prompt, max_new_tokens=2048, do_sample=False)[0]["generated_text"]
    return response.replace(prompt, "").strip()
def check_syntax(code):
    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"{e.__class__.__name__}: {e}"


def write_report(output_path, content):
    with open(output_path, "w") as f:
        f.write(content)

def main():
    code = '''
def mul(a, b);
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a/b
'''

    syntax_ok, syntax_err = check_syntax(code)

    try:
        review_text = generate_review(code)
        review_text = clean_review_text(review_text)
    except Exception:
        tb = traceback.format_exc()
        review_text = f"Code review failed during AI analysis.\n\n```\n{tb}\n```"

    if not syntax_ok:
        content = (
            f"# Syntax Error Detected\n\n"
            f"**Code with syntax error:**\n\n```python\n{code}\n```\n\n"
            f"**Error details:**\n\n```\n{syntax_err}\n```\n\n"
            f"---\n\n"
            f"## AI Review Output (including syntax errors):\n\n"
            f"{review_text}"
        )
    else:
        content = (
            f"# No Syntax Errors Detected\n\n"
            f"## AI Review:\n\n"
            f"{review_text}"
        )

    write_report("review_report.md", content)
    print("Review completed. See 'review_report.md'")


def clean_review_text(text):
    # Remove multiple blank lines to just one
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Detect repeated large blocks and keep only the first
    # A heuristic: split text into paragraphs and keep unique ones
    paragraphs = text.split('\n\n')
    seen = set()
    unique_paragraphs = []
    for p in paragraphs:
        if p.strip() not in seen:
            unique_paragraphs.append(p)
            seen.add(p.strip())
    cleaned_text = '\n\n'.join(unique_paragraphs)

    return cleaned_text.strip()

if __name__ == "__main__":
    main()
