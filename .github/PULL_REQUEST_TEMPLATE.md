1. Do not use AI, such as GPT, to help write your contribution.
   The licensing implications are unclear.

2. Read our
   [contribution guidelines](https://scikit-image.org/docs/dev/development/contribute.html)
   if you have any questions.

3. Use `pre-commit` to check and format code.

   Make it run automatically after each commit using:

   ```
   pip install pre-commit  # install the package
   pre-commit install  # add git hook that runs checks after each commit
   ```

   Or, run it manually: `pre-commit run -a`

## Description

<!-- If this is a bug-fix or enhancement, provide the issue it closes. -->
<!-- If this is a new feature, reference what paper it implements. -->

## Checklist

<!-- Before PRs can be merged, they should provide: -->

- [A descriptive commit message](https://vxlabs.com/software-development-handbook/#good-commit-messages).
- [Unit tests](https://scikit-image.org/docs/stable/user_guide/install.html#testing)
- [Docstrings for all functions](https://github.com/numpy/numpy/blob/master/doc/example.py)
- A gallery example in `./doc/examples` for new features.

## For reviewers

- PR title is concise yet descriptive.
- Label "type: API" is applied for API changes.
- Run benchmarks by adding the `run-benchmark` label (output in "Actions" tab).
