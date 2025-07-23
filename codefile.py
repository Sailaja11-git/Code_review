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
Please follow this strict format in your response:

Check if the code has any syntax errors or structural issues. If yes, point them out clearly and continue reviewing the code assuming those are fixed.

Clearly list all bugs or logical issues found in the code (e.g., wrong operations, misleading names, poor exception handling, etc.).

Suggest improvements for readability, maintainability, or performance, including:

Adding docstrings

Using better variable or function names

Adding type hints

Any refactor recommendations

Then provide the Improved Code: corrected and complete, with all improvements applied (including docstrings).

Then provide Explanation of the corrected/improved code to help the developer understand.

Then provide at least 5 unit tests in Python using unittest (or assert) to cover:

Normal input

Edge cases

Invalid inputs

Floating-point behavior

Error/exception scenarios

Don't skip bugs if any are found. If the code is perfect, say so and continue with explanations and tests.

End your response cleanly without repeating anything or adding extra headings.

Start now.
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
    return a*b
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
