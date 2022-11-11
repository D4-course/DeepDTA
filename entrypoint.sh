#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"

drug=$1
target=$2

if [ $# -gt 2 ] 
then
    echo "You can only enter two arguments at max. " 
    echo "Don't worry, I've handled it for you by passing only the first two :)"
fi

python3 deepdta/main.py --drug "$drug" --target "$target"

