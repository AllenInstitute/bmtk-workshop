FROM continuumio/miniconda3:latest

RUN apt-get -y update && apt-get install -y automake \
                                            build-essential \
                                            git


ENV GIT_BRANCH=update/workshop-2024
ENV GIT_REPO=https://github.com/kaeldai/bmtk.git

ENV BUILD_DIR=/home/build
ENV HOME_DIR=/home/shared
ENV WORK_DIR=${HOME_DIR}/workspace

RUN mkdir -p ${BUILD_DIR}
RUN mkdir -p ${HOME_DIR}
RUN mkdir -p ${WORK_DIR}

RUN apt-get install -y libgsl-dev
RUN conda install conda-forge::nest-simulator

RUN pip install neuron

RUN conda install jupyterlab

RUN pip install scikit-image

RUN pip install neo 

RUN pip install quantities

RUN cd ${BUILD_DIR} \
    && git clone --branch ${GIT_BRANCH} ${GIT_REPO} \
    && cd bmtk \
    && python setup.py install

RUN cd ${HOME_DIR}; \
    git clone --branch updates/2024-add-tuts https://github.com/kaeldai/bmtk-workshop.git

COPY setup_env.sh ${HOME_DIR}/bmtk-workshop
RUN chmod +x ${HOME_DIR}/bmtk-workshop/setup_env.sh && cd ${HOME_DIR}/bmtk-workshop && ./setup_env.sh

COPY entry_script.sh ${BUILD_DIR}
RUN chmod +x ${BUILD_DIR}/entry_script.sh
ENTRYPOINT ["/home/build/entry_script.sh"]

