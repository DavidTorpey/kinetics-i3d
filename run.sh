#!/bin/bash

sudo rm -r /var/lib/dpkg/lock*
sudo apt-get update
sudo apt-get install -y unrar

cd /dev/shm
aws s3 cp -r s3://wits-msc/UCF101/ .
unrar x UCF101.rar
unzip UCF101TrainTestSplits-RecognitionTask.zip
tar -xzvf ucf101_I3D.tgz

cd ~/kinetics-i3d

sudo pip2 install --upgrade opencv-python tensorflow-gpu dm-sonnet tensorflow-probability-gpu
