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
    try:
        code_path = sys.argv[sys.argv.index("--path") + 1]
        output_path = sys.argv[sys.argv.index("--output") + 1]
    except Exception as e:
        # If args missing, write error report and exit
        write_report("review_report.md", f"# ❌ Code Review Failed\n\nMissing or invalid arguments:\n```\n{e}\n```")
        sys.exit(1)

    try:
        with open("Code.py", "r") as f:
            code = f.read()
    except Exception as e:
        write_report(output_path, f"# ❌ Code Review Failed\n\nCould not read file `{"Code.py"}`:\n```\n{e}\n```")
        sys.exit(1)

    syntax_ok, syntax_err = check_syntax(code)

    if not syntax_ok:
        # Write syntax error report, no AI review
        content = f"# ❌ Syntax Error Detected\n\n```\n{syntax_err}\n```\n\nAI review skipped due to syntax errors."
        write_report(output_path, content)
        return

    try:
        review_text = generate_review(code)
        content = f"# 🔍 AI Code Review Report\n\n✅ No syntax errors detected.\n\n## AI Review:\n\n{review_text}"
        write_report(output_path, content)
    except Exception:
        # Catch any AI/model errors, write them in report
        tb = traceback.format_exc()
        content = f"# ❌ Code Review Failed during AI review\n\n```\n{tb}\n```"
        write_report(output_path, content)
        sys.exit(1)


if __name__ == "__main__":
    main()
