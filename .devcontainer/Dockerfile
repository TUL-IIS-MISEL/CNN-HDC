FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ARG USERNAME=vscode
# Create the user
RUN useradd -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Uncomment this section to install additional\ OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends git openssh-client icu-devtools


# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY .devcontainer/environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

RUN chmod a+rwx /opt/conda

USER $USERNAME