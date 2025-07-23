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

   Please provide your review in the following structured format with explicit section headers:

Explanation what input code does:

Bugs or issues:
-Check for logical errors and exception handling
-Detect mismatch between function names, variable names, and the actual operations they perform.
-Flag misleading names or comments
-Highlight any potential runtime errors

Improvements (style, readability, performance), including adding proper docstrings:

Improved Code (complete corrected code snippet):

Explanation after improved code :

Unit Tests (example test cases):
Please generate at least 5 diverse test cases covering:
- Normal input values
- Edge cases (e.g., zero, negative numbers)
- Invalid input handling
- Floating point behavior
- Exception conditions
- Include at least 3 test cases
- Cover normal, edge, and exception cases


Include them using Python's `unittest` framework or simple `assert` statements.

End your response clearly without repeating sections or extraneous text.
"""
    response = generator(prompt, max_new_tokens=1024, do_sample=False)[0]["generated_text"]
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
def div(a, b);
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
