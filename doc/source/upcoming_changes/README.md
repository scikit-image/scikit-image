# Pending release notes

This project uses Towncrier for aggregating release notes.
A description of the automated aggregation process is described in RELEASE.txt.

### Adding news fragments

All non trivial changes must be accompanied by a [news fragment
entry](https://github.com/twisted/towncrier#news-fragments). The fragment file
name should be “doc/source/upcoming_changes/<PR number>.<type>.rst” where
"type" can be one of  api, improvements, deprecated, process, feature, bugfix,
doc, (or trivial if you do not wish the changes included in the release notes.

Example: 5456.bugfix.rst

In brief, the contents of the rst file should be brief, in sentence case, in an
imperative tone, and targetted to end users. See [this
link](https://pip.pypa.io/en/stable/development/contributing/?highlight=towncrier#news-entries)
for a more detailed description.


### Manual aggregation of release notes

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
