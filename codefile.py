import sys
import traceback

try:
    code_path = sys.argv[sys.argv.index("--path") + 1]
    output_path = sys.argv[sys.argv.index("--output") + 1]
    code = open("Code.py").read()

    # For now, simulate a review (replace with LLM later)
    review = f"# Code Review for `{code_path}`\n\n"
    review += " Syntax seems valid.\n\n"
    review += "️ No deep review yet.\n"

    with open(output_path, "w") as f:
        f.write(review)

except Exception as e:
    with open("review_report.md", "w") as f:
        f.write("Code Review Failed\n\n")
        f.write("```\n")
        traceback.print_exc(file=f)
        f.write("```")
    sys.exit(1)
