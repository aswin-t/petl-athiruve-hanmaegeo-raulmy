docker image inspect athiruve/w266:project || docker build -t athiruve/w266:project -f DockerFile .
container=$(docker run -d --gpus all --rm --net=host --ipc=host -v $PWD:/workspace athiruve/w266:project 2>&1 | xargs)
sleep 4
myurl=$(docker logs "$container" 2>&1 | grep "       http://hostname:8888")
# ipadd=$(wget -qO- https://ipecho.net/plain | grep "")
# echo "${myurl/hostname/"$ipadd"}" 
echo "${myurl/hostname/localhost}" 
