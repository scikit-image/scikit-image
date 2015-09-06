#!/bin/bash
if [[ $TRAVIS_PULL_REQUEST == false && $TRAVIS_BRANCH == "master" ]]
then
    echo "-- pushing docs --"

    (
    git config user.email "travis@travis-ci.com"
    git config user.name "Travis Bot"

    git clone --quiet --branch=gh-pages https://${GH_REF} doc_build
    rm -r doc_build/dev
    cp -r doc/build/html doc_build/dev

    cd doc_build
    git add dev
    git commit -m "Deployed to GitHub Pages"
    git push --force --quiet "https://${GHTOKEN}@${GH_REF}" master:gh-pages > /dev/null 2>&1
    )
else
    echo "-- will only push docs from master --"
fi
