# Can sub out for any other docker image with necessary python version
FROM mcr.microsoft.com/devcontainers/python

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends build-essential