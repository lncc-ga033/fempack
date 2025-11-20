import json
import re


def sanitize_id(name):
    return re.sub(r"[^a-zA-Z0-9]", "_", name).lower().strip("_")


with open(".github/classroom/autograding.json", "r") as f:
    data = json.load(f)

tests = data["tests"]

yaml_content = """name: Autograding Tests

on:
  - push
  - workflow_dispatch

permissions:
  checks: write
  actions: read
  contents: read
  pull-requests: write

jobs:
  run-autograding-tests:
    runs-on: ubuntu-latest
    if: github.actor != 'github-classroom[bot]'
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install Dependencies
      run: pip install -e '.[dev]'

"""

runner_ids = []
env_vars = {}

for test in tests:
    test_name = test["name"]
    test_id = sanitize_id(test_name)
    runner_ids.append(test_id)

    # Escape single quotes in run command if necessary, though YAML block scalar handles most
    run_cmd = test["run"]
    setup_cmd = test["setup"]
    points = test["points"]
    timeout = test.get("timeout", 10)

    yaml_content += f"""    - name: {test_name}
      id: {test_id}
      uses: classroom-resources/autograding-command-grader@v1
      with:
        test-name: {test_name}
        command: {run_cmd}
        timeout: {timeout}
        max-score: {points}
      continue-on-error: true

"""
    env_key = test_id.replace("-", "_").upper()
    env_vars[f"{env_key}_RESULTS"] = "${{steps." + test_id + ".outputs.result}}"

yaml_content += """    - name: Autograding Reporter
      id: autograding-reporter
      uses: classroom-resources/autograding-grading-reporter@v1
      if: always()
      continue-on-error: true
      env:
"""

for key, value in env_vars.items():
    yaml_content += f'        {key}: "{value}"\n'

yaml_content += "      with:\n"
yaml_content += f"        runners: '{','.join(runner_ids)}'\n"

yaml_content += """
    - name: Post Autograding Results
      uses: actions/github-script@v6
      if: always()
      with:
        script: |
          const points = "${{ steps.autograding-reporter.outputs.points }}";
          const availablePoints = "${{ steps.autograding-reporter.outputs.available-points }}";

          const finalPoints = (points && points !== "undefined") ? points : "0";
          const finalAvailablePoints = (availablePoints && availablePoints !== "undefined") ? availablePoints : "0";

          // Find the PR associated with this push
          const prs = await github.rest.pulls.list({
            owner: context.repo.owner,
            repo: context.repo.repo,
            head: `${context.repo.owner}:${context.ref.replace('refs/heads/', '')}`,
            state: 'open'
          });

          if (prs.data.length > 0) {
            const pr = prs.data[0];
            await github.rest.issues.createComment({
              issue_number: pr.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `Autograding points: ${finalPoints}/${finalAvailablePoints}`
            });
          }
"""

print(yaml_content)
