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
````

Please provide your review in the following structured format using hyphenated bullet points under each heading:

---

# Syntax Error Detected (if any)

* If there is a syntax error, provide:

  * The **exact error message**
  * The **line of code** causing the issue
  * A short **explanation** of what went wrong

---

## Explanation of what the input code does:

* Describe what the function is *intended* to do based on its name and docstring (if present).
* Explain what the function actually *does* based on the code logic.

---

## Bugs or Issues:

* Point out any **syntax or logical errors**
* Mention if the function name does not match what the function actually does.
* Highlight any **misleading names, poor variable naming, or comments**
* Flag **any potential runtime errors or unhandled edge cases**

---

## Improvements (style, readability, performance), including adding proper docstrings:

* Suggest improvements for:

  * Code clarity
  * Naming conventions
  * Exception handling
  * Code efficiency
  * Docstring quality

---

## Improved Code (complete corrected and logically consistent code snippet):

* Provide the full improved version of the function with:

  * Corrected logic
  * Proper variable names
  * Docstrings explaining the function’s purpose, parameters, and return value
  * Any necessary type checks and exception handling

---

## Explanation after Improved Code:

* Explain how the corrected code:

  * Fixes the original problems
  * Better aligns with its intended behavior
  * Is more robust, readable, or maintainable


Unit Tests (example test cases):
Please generate at least 5 diverse test cases covering:

Normal input values

Edge cases (e.g., zero, negative numbers)

Invalid input handling

Floating point behavior

Exception conditions

Include at least 3 test cases

Cover normal, edge, and exception cases

Include them using Python's unittest framework or simple assert statements.

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
