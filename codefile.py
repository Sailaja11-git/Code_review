import sys
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def generate_review(code):
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"  # Make sure it's downloaded or cached
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    prompt = f"### Review the following Python code and provide detailed suggestions, bug fixes, "f"and improvements:\n\n```python\n{code}\n```\n\n### Review:"
    response = generator(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]

    # Clean output (optional)
    return response.replace(prompt, "").strip()


def write_review_report(output_path, review):
    with open(output_path, "w") as f:
        f.write("# 🔍 AI Code Review Report\n\n")
        f.write(review)


if __name__ == "__main__":
    try:
        code_path = sys.argv[sys.argv.index("--path") + 1]
        output_path = sys.argv[sys.argv.index("--output") + 1]
        code = open("Code.py").read

        review = generate_review(code)
        write_review_report(output_path, review)

    except Exception as e:
        with open("review_report.md", "w") as f:
            f.write("# Code Review Failed\n\n")
            f.write("```\n")
            f.write(str(e))
            f.write("\n```")
        sys.exit(1)
