ffmpeg -y -f image2 -i images/image_%03d.jpg -r 12 -vcodec libx264 -profile high -preset slow timelapse.mp4
