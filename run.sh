#!/bin/bash

#sudo apt-get update
#sudo apt-get install -y unrar

cd /dev/shm
mkdir generated
aws s3 cp s3://wits-msc/UCF101/UCF101.rar .
aws s3 cp s3://wits-msc/UCF101/UCF101TrainTestSplits-RecognitionTask.zip .
aws s3 cp s3://wits-msc/UCF101/ucf101_I3D.tgz .
unrar x UCF101.rar
unzip UCF101TrainTestSplits-RecognitionTask.zip
tar -xzvf ucf101_I3D.tgz

cd ~/kinetics-i3d

touch temp.txt

sudo pip2 install --upgrade opencv-python tensorflow-gpu dm-sonnet tensorflow-probability-gpu tesnroflow-probability
sudo pip2 install opencv-python==3.4.4.19
sudo pip2 install imageio

