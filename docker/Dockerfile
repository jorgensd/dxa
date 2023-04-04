FROM ghcr.io/fenics/dolfinx/dolfinx:nightly

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

WORKDIR /tmp/

COPY . ./dxa
RUN python3 -m pip install ./dxa[all]


RUN rm -rf /tmp
