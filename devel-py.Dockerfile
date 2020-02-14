FROM quay.io/pypa/manylinux2010_x86_64

WORKDIR /usr/src/cuBERT
COPY . .

RUN yum -y install \
        cmake3 \
        && \
    yum -y clean all && \
    rm -rf /var/cache/yum/*

RUN rm -rf build && mkdir build && cd build && \
    cmake3 -DCMAKE_BUILD_TYPE=Release -DcuBERT_ENABLE_MKL_SUPPORT=ON .. && \
    make

RUN /opt/python/cp36-cp36m/bin/pip install Cython numpy
RUN /opt/python/cp27-cp27mu/bin/pip install Cython numpy

RUN cd python && \
    /opt/python/cp36-cp36m/bin/python setup.py bdist_wheel && \
    /opt/python/cp27-cp27mu/bin/python setup.py bdist_wheel
