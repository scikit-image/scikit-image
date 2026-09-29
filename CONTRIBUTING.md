(howto-contribute)=

# How to contribute to scikit-image

Developing open source software as part of a community is fun, and
often quite educational!

We coordinate our work using GitHub, where you can find lists of [open
issues](https://github.com/scikit-image/scikit-image/issues?q=is%3Aopen)
and [new feature requests](https://github.com/scikit-image/scikit-image/labels/%3Apray%3A%20Feature%20request).

To follow along with discussions, or to get in touch with the
developer team, please join us on the [scikit-image developer forum](https://discuss.scientific-python.org/c/contributor/skimage) and
the [Zulip chat](https://skimage.zulipchat.com/).

Please post questions to these public forums (rather than contacting
developers directly); that way, everyone can benefit from the answers,
and developers can answer according to their availability. Don't feel
shy, the team is very friendly!

```{contents}
:local: true
```

## Development process

The following is a brief overview about how changes to source code and documentation
can be contributed to scikit-image.

1. If you are a first-time contributor:

   - Go to [https://github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image) and click the
     "fork" button to create your own copy of the project.

   - [Set up GitHub SSH authentication](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh).

   - Clone (download) the repository with the project source on your local computer:

     ```
     git clone --origin upstream git@github.com:scikit-image/scikit-image
     ```

   - Change into the root directory of the cloned repository:

     ```
     cd scikit-image
     ```

   - Add your fork as a
     [remote repository](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes)
     that you will interact with.

     Assuming a GitHub username of `codemonkey`:

     ```
     git remote add codemonkey git@github.com:codemonkey/scikit-image
     git fetch codemonkey
     ```

   - You now have two remote repositories:

     - `upstream`, which refers to the `scikit-image` project repository, and
     - `codemonkey`, which refers to your personal fork.

   - Next, {ref}`set up your build environment <build-env-setup>`.

   - Finally, we recommend that you use our pre-commit hook, which runs code
     checkers and formatters each time you do a `git commit`:

     ```
     pip install pre-commit
     pre-commit install
     ```

2. Develop your contribution:

   - Pull the latest changes from the project:

     ```
     git switch main
     git fetch upstream main
     git merge upstream/main
     ```

   - Create a branch for the feature you want to work on. Use a sensible name,
     such as 'transform-speedups':

     ```
     git switch -c transform-speedups
     ```

   - Commit locally as you progress (with `git add` and `git commit`).
     Please write [good commit messages](https://vxlabs.com/software-development-handbook/#good-commit-messages).

   - It is a good idea to read our {ref}`guidelines` at this point.
     While we don't require a contribution to meet every guideline from the
     start, they will come up during review.

3. To submit your contribution:

   - Push your changes back to your fork on GitHub:

     ```
     git push codemonkey transform-speedups
     ```

     A message will be displayed with a URL to open in your browser to create a
     pull request (PR).

   - Before submitting the pull request:

     - Use a concise, descriptive title
     - Describe and link relevant context in the description
     - Disclose all _generative_ tools (AI, LLMs, agents) that you used, see our
       {ref}`ai-policy` for details.

   :::{tip}
   If you get stuck, reach out to us on
   [our Zulip chat](https://skimage.zulipchat.com/).
   :::

4. Review process:

   - Reviewers (the other developers and interested community members) will
     write inline and/or general comments on your pull request (PR) to help
     you improve its implementation, documentation, and style. Every single
     developer working on the project has their code reviewed, and we've come
     to see it as a friendly conversation from which we all learn and the
     overall code quality benefits. Therefore, please don't let the review
     discourage you from contributing: its only aim is to improve the quality
     of the project, not to criticize (we are, after all, very grateful for the
     time you're putting in!).

   - To update your pull request, make your changes on your local repository
     and commit. As soon as those changes are pushed up (to the same branch as
     before) the pull request will update automatically.

   - Continuous integration (CI) services are triggered after each
     pull request submission to build the package, run unit tests, and
     check the coding style and formatting of your branch. The tests
     must pass before your PR can be merged. If CI fails, you can find
     out why by clicking on the "failed" icon (red cross) and
     inspecting the build and test logs.

     :::{note}
     PR labeling

     CI will always fail on new PRs, until a maintainer adds a
     suitable category label.
     :::

   - A pull request must be approved by two core team members before merging.

(documenting-changes)=

5. Document changes

   If your change introduces a deprecation, add a reminder to `TODO.txt`
   for the team to remove the deprecated functionality in the future.

   scikit-image uses [changelist](https://github.com/scientific-python/changelist)
   to generate a list of release notes automatically from pull requests. By
   default, changelist will use the title of a pull request and its GitHub
   labels to sort it into the appropriate section. However, for more complex
   changes, we encourage you to describe them in more detail using the
   `release-note` code block within the pull request description; e.g.:

   ````
   ```release-note
   Remove the deprecated function `skimage.color.blue`. Blend
   `skimage.color.cyan` and `skimage.color.magenta` instead.
   ```
   ````

   You can refer to {doc}`/release_notes/index` for examples and to
   [changelist's documentation](https://github.com/scientific-python/changelist)
   for more details.

:::{note}
To reviewers: if it is not obvious from the PR description, make sure that
the reason and context for a change are described in the merge message.
:::

## Divergence between `upstream main` and your feature branch

If GitHub indicates that the branch of your PR can no longer
be merged automatically, merge the main branch into yours:

```
git fetch upstream main
git merge upstream/main
```

If any conflicts occur, they need to be [fixed before continuing](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line).

[We recommend](https://github.com/stefanv/git-tools?tab=readme-ov-file#conflict-diff-display) setting:

```
git config --global merge.conflictstyle zdiff3
```

to make conflict markers easier to read.

An alternative to merging is to rebase your branch—but we squash and merge all
PRs anyway, so we don't mind merge commits.

(ai-policy)=

## AI Policy

scikit-image is technically complex, key research infrastructure;
therefore, we place a high premium on correctness, and on avoiding
technical complexity that may affect maintainability.

To preserve our culture of collaboration, we (the human maintainers)
need to know that you (the human author) fully understand the code you
have submitted, and its implications for the code-base.

If you want to make use of LLMs in a significant way and have your
contributions reviewed, safest is to **check in with us first** to
discuss your strategy.

### AI do's and don'ts

Do not use AI for:

1. Creating novel code (algorithmic additions, writing tests, etc.).[^why-no-ai-tests]
2. Writing pull request descriptions, or commenting on issues or PRs.

How you may use AI:

1. For exploring the codebase, for iterating on hand-written code, and for fixing trivial bugs.
2. To automate mechanical tasks. E.g., if you discover that you need to add a decorator across the code-base.
3. For infrastructure code, such as CI, as long as the changes are easy to review.

[^why-no-ai-tests]:
    The decision about _what to test_ relies on
    understanding _how_ code could give the wrong answer. It is
    important to think about algorithm edge cases: for example, if you
    implement `sin(x)/x`, you know that you need to be particularly
    careful around 0; or for `sqrt(x)` what happens when input is
    negative. We do not want to test simply for the sake of coverage
    either (e.g., trivial input parameter verification); so, while AI
    will happily add a ton of tests, those may not be the _right_
    tests. That said, tests contain a lot of scaffolding,
    and there's no problem using AI to help with creating that
    structure, or with fixing broken tests.

### AI requirements

If you use AI as above, you must:

1. **Always declare tool usage**: say specifically what part of the task you used AI for in the PR description.
2. **Write PR descriptions and comments by hand.** If you do quote AI text, do so sparingly, and indicate where you're doing so, e.g. by writing `:robot: _AI text below_ :robot:`.
   You can also use a `<details>...</details>` block to "fold up" that text by default.
3. As far as possible, **separate AI-generated code from human-generated** code with individual commits.
   Tag AI commits with an `Assisted-by:` tag as recommended by the [Linux kernel](https://docs.kernel.org/process/coding-assistants.html).
   (Commits where AI was predominantly used to tidy up hand-written code or language don't need to carry the tag.)
4. **AI-generated code must be trivial to review:** the reviewer must be able to review the AI-generated code with little background in image-processing or experience working on the code-base, and see that the changes are correct.
   Note that _tests, specifically, are rarely trivial_ [^why-no-ai-tests].
5. You must **take responsibility for copyright** of the AI-generated code; the simplest and best way to do this, is to make sure the code changes are trivial, in the sense that they cannot reasonably be done differently to solve the given problem.
6. **Expect the team to ask questions** about your work - and you must answer these questions yourself, without deferring to AI.

The landscape around AI is changing quickly, and we will continue to update this policy as informed by our current experience.

### Core maintainers requirements

Core maintainers may use their own judgment on when and how to use AI,
including for tasks outside the "how you may use AI" list above. They
must still follow the requirements section: declaring tool usage,
tagging commits, etc.
