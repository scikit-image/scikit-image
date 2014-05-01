#!/usr/bin/env bash
BRANCH=gh-pages
TARGET_REPO=sharky93/docs.git
DOCS_OUTPUT_FOLDER=build/html

if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
    echo -e "Starting to deploy to Github Pages\n"
    if [ "$TRAVIS" == "true" ]; then
        git config --global user.email "rishabhr123@gmail.com"
        git config --global user.name "sharky93"
    fi
    # using token clone gh-pages branch
    git clone --quiet --branch=$BRANCH https://${GH_TOKEN}@github.com/$TARGET_REPO built_website > /dev/null
    # go into directory and copy data we're interested in to that directory
    
    # find the version number from setup.py
    input="../setup.py"
    while read key sep value
    do
     if [[ "$key" == "VERSION" ]]
        then
            if [[ $value = *dev* ]]
                then
                    tag="dev"
                else
                    # rename e.g. 0.9.0 to 0.9.x
                    tag=${value:1:-1}
            fi
     fi
    done < "$input"

    cd built_website/$tag

    rsync -rv --exclude=.git  ../../$DOCS_OUTPUT_FOLDER/* .
    # add, commit and push files
    git add -f .
    git commit -m "Travis build $TRAVIS_BUILD_NUMBER pushed to Github Pages"
    git push -fq origin $BRANCH > /dev/null
    echo -e "Deploy completed\n"
fi
