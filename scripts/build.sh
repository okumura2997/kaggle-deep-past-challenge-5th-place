#!/bin/bash

docker build -f docker/Dockerfile \
    --build-arg UID=$(id -u) --build-arg USER=$USER \
    --network host --rm -t $USER/kaggle-base .