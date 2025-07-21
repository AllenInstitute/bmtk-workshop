FROM continuumio/miniconda3:latest

RUN apt-get -y update && apt-get install -y automake        \
                                            build-essential \
                                            git \
                                            libpthread-stubs0-dev

ARG WORKSHOP_REPO=https://github.com/AllenInstitute/bmtk-workshop.git
ARG WORKSHOP_BRANCH=main

ENV BUILD_DIR=/home/build
ENV HOME_DIR=/home/shared
ENV WORK_DIR=${HOME_DIR}/workspace

RUN mkdir -p ${BUILD_DIR}
RUN mkdir -p ${HOME_DIR}
RUN mkdir -p ${WORK_DIR}


WORKDIR ${BUILD_DIR}

COPY conda_env.bmtk-py311.yml .

# Install prereqs to run jupyter
RUN conda env create -f conda_env.bmtk-py311.yml


# Disable login req. when running jupyter in docker image.
RUN mkdir -p /root/.jupyter
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py


SHELL ["conda", "run", "-n", "bmtk-py3.11", "/bin/bash", "-c"]


# Install Workshop tutorials
RUN cd ${HOME_DIR}; \
    git clone --branch ${WORKSHOP_BRANCH} ${WORKSHOP_REPO}


# Required for running nestml, requires shared library but only finds static one.
RUN ln -s $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/lib64/libpthread.so.0 $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib/libpthread.so


# Run jupyter lab server in the conda-environment when docker image is ran
EXPOSE 8888
CMD ["conda", "run", "--no-capture-output", "-n", "bmtk-py3.11", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--notebook-dir='/home/shared/bmtk-workshop'"]
