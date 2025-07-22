import sys
import traceback
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def generate_review(code):
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    prompt = f"### Review the following Python code and provide detailed suggestions, bug fixes, and improvements:\n\n```python\n{code}\n```\n\n### Review:"
    response = generator(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
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
def greet(name):
    print("Hello, " + name + "!")

greet("World")
'''

    syntax_ok, syntax_err = check_syntax(code)

    if not syntax_ok:
        content = f"# Syntax Error Detected\n\n```\n{syntax_err}\n```\n\nAI review skipped due to syntax errors."
        write_report("review_report.md", content)
        sys.exit(1)

    try:
        review_text = generate_review(code)
        content = f"# AI Code Review Report\n\nNo syntax errors detected in the hardcoded code snippet.\n\n## 🔍 AI Review:\n\n{review_text}"
        write_report("review_report.md", content)
        print("Review completed. See 'review_report.md'")
    except Exception:
        tb = traceback.format_exc()
        content = f"# Code Review Failed during AI analysis\n\n```\n{tb}\n```"
        write_report("review_report.md", content)
        print(" Code review failed. See 'review_report.md' for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
