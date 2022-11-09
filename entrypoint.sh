#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"

drug=$1
target=$2

if [ $# -gt 2 ] 
then
    echo "You can only enter two arguments at max. " 
fi

python3 test.py --drug "$drug" --target "$target"