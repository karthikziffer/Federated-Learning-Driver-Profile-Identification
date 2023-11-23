#!/bin/bash

sudo apt-get update -y
sudo apt-get upgrade -y
echo Done!

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

docker version

sudo docker run hello-world


