#!/bin/bash
set -ex

curl -O https://storage.googleapis.com/golang/go1.17.6.linux-amd64.tar.gz
file go1.17.6.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.17.6.linux-amd64.tar.gz

export PATH=$PATH:/usr/local/go/bin

for PYBIN in /opt/python/{cp36-cp36m,cp37-cp37m,cp38-cp38,cp39-cp39,cp310-cp310}/bin; do
    export PYTHON_SYS_EXECUTABLE="$PYBIN/python"

   "${PYBIN}/python" -m pip install pybindgen

    go get golang.org/x/tools/cmd/goimports
    go get github.com/go-python/gopy
    ls
    go mod init github.com/cohere-ai/tokenizer
    go mod tidy
    ~/go/bin/gopy build -output=tokenizer -vm=python3 github.com/cohere-ai/tokenizer
    "${PYBIN}/python" setup.py bdist_wheel

    rm -rf build/*
done

# Keep only manylinux wheels
rm dist/*-linux_*

# Upload wheels
/opt/python/cp37-cp37m/bin/pip install -U awscli
/opt/python/cp37-cp37m/bin/python -m awscli s3 sync --exact-timestamps ./dist "s3://tokenizers-releases/python/$DIST_DIR"
