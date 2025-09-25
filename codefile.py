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


Explanation of what the input code does:

-Describe clearly what the function claims to do based on its name and docstring (if any).
-Then describe what it actually does based on the code logic.

Bugs or issues:
-Check for logical errors and exception handling
- If function name and operation mismatch, correct both consistently in improved code.
-Detect mismatch between function names, variable names, and the actual operations they perform.
-Flag misleading names or comments
-Highlight any potential runtime errors

Improvements (style, readability, performance), including adding proper docstrings:

Improved Code (complete corrected and logically consistent code snippet):

Explanation after improved code:
-Explain how the corrected code now behaves properly and aligns with its purpose.

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
import time
from enmutils.lib import log
from enmutils_int.lib.profile_flows.common_flows.common_flow import GenericFlow
from enmutils_int.lib.em import get_profile_users_nodes, retrieve_poids, validate_sessions, task_set
from enmutils_int.lib.services.deployment_info_helper_methods import is_scs_deployment, is_transport_network
 
 
class EM01Flow(GenericFlow):
 
    SUCCESSFUL_NODES = []
    POIDS = []
 
    def execute_flow(self):
        """
        Executes the flow for the profile
        """
        self.state = "RUNNING"
        users, configured_nodes = get_profile_users_nodes(self)
        if is_scs_deployment() and is_transport_network():
            users = users[:18]
            log.logger.debug("SCS + transport: limiting users to 18")
 
        log.logger.debug("Users:{0}, Nodes:{1}".format(
            len(users), [node.node_name for node in configured_nodes]
        ))
 
        self.download_tls_certs(users)
 
        while self.keep_running():
            self.sleep_until_time()
            try:
                if len(self.POIDS) < len(users):
                    self.POIDS.extend(retrieve_poids(self, configured_nodes))
 
                user_nodes = list(zip(users, set(self.POIDS)))
                for i in range(0, len(user_nodes), self.PARALLEL_SESSIONS_LAUNCH):
                    self.create_and_execute_threads(
                        user_nodes[i:i + self.PARALLEL_SESSIONS_LAUNCH],
                        len(user_nodes),
                        func_ref=task_set,
                        args=[self]
                    )
                    log.logger.debug("Completed opening {0} sessions".format(
                        min(i + self.PARALLEL_SESSIONS_LAUNCH, len(user_nodes))
                    ))
 
                time.sleep(600)
                validate_sessions(users)
            except Exception as e:
                self.add_error_as_exception(e)
            finally:
                try:
                    for user in users:
                        user.remove_session()
                except Exception as e:
                    self.add_error_as_exception(e)
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
