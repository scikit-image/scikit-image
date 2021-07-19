:orphan:

If automated deployment of release notes fails you can follow the manual
instructions in this document.

- Identify if the Notes artifact is currently available for the failed
“release” workflow; If that is the case, you can manually download the
artifacts and extract the release file, you can then follow the process of
creating a release on Github and submit the release notes as its contents;

- If the Notes artifacts are currently not available because the CI action
failed or is not available you can run the following steps on your local
machine.

```
cd <path/to/scikit-image/repository>
pip install -r requirements/_release_docs.txt
python tools/aggregate_contributors .
towncrier build --version=<your.release.version>
cat "reviewers_and_authors.txt" >>   \
    doc/release/release_<your release version>.rst
```
