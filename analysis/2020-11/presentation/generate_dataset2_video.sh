#!/bin/sh

root=$(dirname $0)/../../../datasets/rc_car/Stanmer_park_dataset/0511/unwrapped_dataset2
ffmpeg -i "$root/image%d.jpg" -r 15 dataset2.mp4
