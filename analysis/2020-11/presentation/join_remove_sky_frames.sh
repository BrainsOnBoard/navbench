#!/bin/sh

ffmpeg -i unwrapped_dataset1_nosky/image%d.jpg -r 5 nosky.mp4
