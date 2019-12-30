#!/bin/bash
if [[ $TRAVIS_PULL_REQUEST == false && $TRAVIS_BRANCH == "master" &&
      $BUILD_DOCS == 1 && $DEPLOY_DOCS == 1 ]]
then
    # See https://help.github.com/articles/creating-an-access-token-for-command-line-use/ for how to generate a token
    # See https://docs.travis-ci.com/user/encryption-keys/ for how to generate 
    # a secure variable on Travis
    echo "-- pushing docs --"

    (
    git config --global user.email "travis@travis-ci.com"
    git config --global user.name "Travis Bot"

    # build docs a second time to fix links to Javascript
    (cd doc && make html)

    git clone --quiet --branch=gh-pages https://${GH_REF} doc_build
    cd doc_build

    git rm -r dev
    cp -r ../doc/build/html dev
    git add dev

    git commit -m "Deployed to GitHub Pages"
    git push --force --quiet "https://${GH_TOKEN}@${GH_REF}" gh-pages > /dev/null 2>&1
    )
else
    echo "-- will only push docs from master --"
fi
