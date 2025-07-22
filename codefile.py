import re
import sys
import traceback
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def sanitize_review_output(text):
    # Remove echoed code block
    text = re.sub(r"```python[\s\S]+?```", "", text)

    # Remove repeated section headers (e.g., multiple ### Suggestions)
    seen_sections = set()
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        if line.strip().startswith("###"):
            if line.strip() in seen_sections:
                continue  # skip duplicate section
            seen_sections.add(line.strip())
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def generate_review(code):
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    prompt = f"""
    ### Review the following Python code:
    ```python
    {code}
    ```

    ### Tasks:

    1. Explain what the code does.
    2. Identify any bugs or potential issues.
    3. Suggest improvements (style, readability, performance).
    4. Generate unit tests for this code.

    ### Review:

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
    # Hardcoded code snippet to review
    code = '''
def multiply(a, b);
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a*b
'''

    syntax_ok, syntax_err = check_syntax(code)

    try:
        review_text = sanitize_review_output(generate_review(code))
    except Exception:
        tb = traceback.format_exc()
        review_text = f"Code review failed during AI analysis.\n\n```\n{tb}\n```"

    if not syntax_ok:
        content = (
            f"#  Syntax Error Detected\n\n"
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


if __name__ == "__main__":
    main()
