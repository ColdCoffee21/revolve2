
study="plasticoding_seasons"
mainpath="/storage/karine"
file="${mainpath}/${study}/analysis/video_bests.mpg";

printf " \n making video..."
screen -d -m -S videos ffmpeg -f x11grab -r 25 -i :1 -qscale 0 $file
python3 experiments/${study}/watch_robots.py
killall screen
printf " \n finished video!"