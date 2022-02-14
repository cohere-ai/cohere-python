#!/bin/bash
set -ex

curl -O https://storage.googleapis.com/golang/go1.17.6.linux-amd64.tar.gz
file go1.17.6.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.17.6.linux-amd64.tar.gz

for PYVER in 3.6.15 3.7.9; do
    curl -O https://www.python.org/ftp/python/$PYVER/Python-$PYVER.tgz && \
                tar xzf Python-$PYVER.tgz && \
                cd Python-$PYVER && \
                ./configure --enable-optimizations --enable-shared && \
                make altinstall && \
                cd /
done

export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export GOBIN=$GOPATH/bin
export PATH=$PATH:$GOBIN

mkdir -p $GOPATH/src/github.com/cohere-ai
for PYBIN in /opt/python/{cp36-cp36m,cp37-cp37m,cp38-cp38,cp39-cp39,cp310-cp310}/bin; do
    export PYTHON_SYS_EXECUTABLE="$PYBIN/python"
    cd $GOPATH/src/github.com/cohere-ai && git clone https://github.com/cohere-ai/tokenizer.git && cd $GITHUB_WORKSPACE

   "${PYBIN}/python" -m pip install pybindgen

    go get golang.org/x/tools/cmd/goimports
    go get github.com/go-python/gopy
    cd $GOPATH/src/github.com/cohere-ai/tokenizer
    ~/go/bin/gopy build -output=tokenizer -vm="${PYBIN}/python" github.com/cohere-ai/tokenizer
    cd tokenizer && make build
    cp -a $GOPATH/src/github.com/cohere-ai/tokenizer/tokenizer $GITHUB_WORKSPACE/cohere-$PYBIN/tokenizer
    cd $GITHUB_WORKSPACE
    "${PYBIN}/python" setup.py bdist_wheel

    rm -rf $GOPATH/src/github.com/cohere-ai/tokenizer
    rm -rf build/*
done

# Keep only manylinux wheels
rm dist/*-linux_*

# Upload wheels
/opt/python/cp37-cp37m/bin/pip install -U awscli
/opt/python/cp37-cp37m/bin/python -m sync --exact-timestamps ./dist "s3://tokenizers-releases/python/$DIST_DIR"
