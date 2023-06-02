#!/usr/bin/env bash

image_name=anomaly_image
container_name=anomaly_container

if [ -z "$(docker images -q $image_name)" ]; then
    nvidia-docker build -f Dockerfile -t $image_name ./ \
    --build-arg uid=$(id -u) \
    --build-arg gid=$(id -g)
fi

nvidia-docker run --name $container_name \
	      -it -p 8890:8888 --shm-size=64g \
	      -v ~/src:/home/anom/src \
	      -v ~/wdata:/home/anom/wdata \
	      -v /nfs:/nfs \
	      -v /local_data:/local_data \
	      $image_name
