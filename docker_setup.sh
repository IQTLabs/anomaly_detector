#!/usr/bin/env bash

image_name=anomaly_image
container_name=anomaly_container

if [ -z "$(docker images -q $image_name)" ]; then
    nvidia-docker build -f Dockerfile -t $image_name ./
fi

nvidia-docker run --name $container_name \
	      -it \
	      --user $(id -u):$(id -g) \
	      $image_name
