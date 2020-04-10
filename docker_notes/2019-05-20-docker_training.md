# Mateusz Filipowicz - Docker Training Day 1  <2019-05-20 Mon>

1. Problems and issues without containers.
    - Time consuming.
    - Space optimization.
    - New ship developed after the container and not the other way around.
    - containers fit multiple different means of transportation.
    - Cheap movement of products from one marketplace around the world.
    - Low cost container ships.
    - Temperature control for degradable goods.
    - Special cases: live animals, cars, technological goods. Same container for each one of them.
2. Things that containers solved or added value to (benefits).

You shouldn't run Docker on its own in a production system. We will find out why in a little while.

## How to run a simple image

`docker run hello-world` runs a docker image and creates a container. The container is given a random name, unless it is specified by the user.

In most cases you would run the container in the background to avoid spamming your console. For this purpose you should use `docker run -d hello-world`. This will run the container in the background and just print the container ID.
To see the output of a detached container you should use `docker logs <ID>`. This will show what the output would have been if you had not detached the container.

Docker images have the form `[owner/]<name>:tag`. In the case below, `disco` is the tag.

```sh
docker run -d ubuntu:disco
```

What this image does is to run `/bin/bash` and is then closed. The logs contain nothing. The container does something and then shuts down. In this case it just ran `/bin/bash`, so no output is produced.

## How to pass a command to `docker run`

We can pass an optional command by placing it after the previous string. For example, if we want to run the command `echo 'Hello Docker` we should write:

```sh
docker run -d ubuntu:disco echo 'Hello Docker'
```

Since the container is detached, the message is echoed in the logs, not in the STDOUT.

## Running docker in an interactive terminal

To run the above image in an interactive way in an allocated terminal (TTY)
`docker run -i -t ubuntu:disco` or `docker run -it ubuntu:disco`. Note that you shouldn't use `-d` here. This will produce a prompt that looks like `root@3fb51c9908ca` and the string after `@` is the container ID. You can use this then in `docker logs 3fb51c9908ca` and you will find the history.

## See all the existing containers (running or not)

To see all the containers run so far you can use `docker ps -a`. This will show the `CONTAINER ID`, the `IMAGE`, e.g. `ubuntu:disco` the entry point `COMMAND` (discussed later), when it was `CREATED`, the `STATUS`, the `PORTS` and some interesting `NAMES`. The IDs and the names are unique. The names are generated from a dictionary, but they can be specified with the `--name` option. You can reference docker imaves both with the ID and with the name.

## General format of `docker run`

General format:
`docker run [FLAGS] IMAGE[:TAG] [COMMAND] [ARGUMENTS...]`

Example of optional flags: `-d, -it, --net`. Example of image `ubuntu:disco`.

## What is a container

**Hypervisor**: is one of the key processes, used to manage resources, for example for virtual machines. There are two types:
- type 1: doesn't need an intermediate OS and runs directly on the hardware.
- type 2: allows one host OS tu support multiple guest OSs

**Virtual Machine** (not the same thing as a container): is an abstraction layer on top of the host OS. You have virtual hardware for each application, so different applications do not interfere with each other. It is memory hungry, slow boot time, performance impact.

VMs clone an OS whereas containers *share* an OS. Docker is a *container engine*

Question: when we ran `ubuntu:disco` we were using binaries and libraries, but from what? It felt like Ubuntu, but was it?

## [Exercise 4](https://stash.intranet.roche.com/stash/users/filipowm/repos/docker-training/browse/exercises/4_docker_internals.md)

Question: what is `pstree` doing, and from what point of view?

PID mydb : 6306
PPID mydb: 6289

```sh
[Q2] What was the bash process ID (PID)? Why?
PID TTY          TIME CMD
    1 pts/0    00:00:00 bash
   16 pts/0    00:00:00 ps
```

[Q3] What is the redis-server PID within container?
```sh
[root@localhost training]# nsenter --target $DBPID --mount --uts --ipc --net --pid ps aux
PID   USER     TIME  COMMAND
    1 redis     0:01 redis-server
   12 root      0:00 ps aux
```

If you kill this process with PID 1, you kill the container.

[Q4] Which namespaces are identical in web and mydb containers (pointing to same location)?
```sh
root@localhost training]# ls -lha /proc/$WEBPID/ns/
total 0
dr-x--x--x. 2 100 101 0 May 20 09:56 .
dr-xr-xr-x. 9 100 101 0 May 20 09:56 ..
lrwxrwxrwx. 1 100 101 0 May 20 09:56 ipc -> ipc:[4026532212]
lrwxrwxrwx. 1 100 101 0 May 20 09:56 mnt -> mnt:[4026532210]
lrwxrwxrwx. 1 100 101 0 May 20 09:56 net -> net:[4026532132]
lrwxrwxrwx. 1 100 101 0 May 20 09:56 pid -> pid:[4026532213]
lrwxrwxrwx. 1 100 101 0 May 20 09:56 user -> user:[4026531837]
lrwxrwxrwx. 1 100 101 0 May 20 09:56 uts -> uts:[4026532211]
```
```sh
[root@localhost training]# ls -lha /proc/$DBPID/ns/
total 0
dr-x--x--x. 2 100 101 0 May 20 09:21 .
dr-xr-xr-x. 9 100 101 0 May 20 09:21 ..
lrwxrwxrwx. 1 100 101 0 May 20 09:27 ipc -> ipc:[4026532129]
lrwxrwxrwx. 1 100 101 0 May 20 09:27 mnt -> mnt:[4026532127]
lrwxrwxrwx. 1 100 101 0 May 20 09:21 net -> net:[4026532132]
lrwxrwxrwx. 1 100 101 0 May 20 09:27 pid -> pid:[4026532130]
lrwxrwxrwx. 1 100 101 0 May 20 09:27 user -> user:[4026531837]
lrwxrwxrwx. 1 100 101 0 May 20 09:27 uts -> uts:[4026532128]
```

Here the `user` and `net` namespaces are the same. This guarantees that the network is shared between the two containers.

It is a best practice to limit the amount of memory that a container can use. By default they can always use all the memory, so, if you are using many containers, they can kill your machine. Orchestrators don't have this problem. You can overflow the memory, but you can also overflow the CPU. We will see exercises about this later.

TODO: understand what `cgroups` are.

A container are a group of processes which are using kernel features. This is why you cannot run an operating system on Docker. They share kernel features (namespaces, cgroups etc) and filesystem to pretend they are running on separate machine and separate OS.

## Some useful docker commands

```sh
docker top
docker stats
docker diff
docker inspect
```

`docker inspect` returns *a lot* of information. 
You can enter a container with `docker exect -it container /bin/bash`

## Modularity in the container world

There's something called the **Open Container Initiative**. It's goal is to provide a unified interface that can be used by different container engines. `containerd` is the main engine, used by many. Singularity is a commonly used engine for scientific and HPC. The target useage is ML/DL/AI Data Science. HPC requires high security and Singularity offers some extra security levels.
CRI-O is a lightweight, fast alternative to Docker for Kubernetes. Balena Engine is a container for embedded devices. RKT is similar to Docker, apparently more secure, but not very commonly used. In Kubernetes you can easily switch from an engine to another.

The `docker daemon` is the server, on top of which there is a REST API on top of which there is a client `docker CLI`.

### Client-Server architecture

The client would be the CLI, which interacts with the daemon and the host where the daemon is running. Images are stored on the host, and they are recipes on how to create the containers. The images, if not locally found, are downloaded from the Docker Hib. You can also use with private registries (Roche has many). The image is always fixed: once it is created, it cannot be modified.

## Stopped containers and resources usage

Stopped containers are not using resources, but they may have data, and therefore still consume memory. This is why it can be useful to remove stopped containers.
The container can be `running` or `exited`, but it can also be `paused` and it can be `created`. There are two ways of creating a container: `docker run` which creates and runs a container, and `docker create` which creates it but doesn't run it. To start it you can use `docker start`. There is a difference between `docker stop` and `docker kill`. You can `docker pause` and `docker unpause`. A paused process exists and doesn't take time to restart, while an exited process has a boot-up time.

`docker stop` stops a process gracefully. `docker kill` does it unconditionally, without caring whether there are still logged files or attached resources.

## Collecting events

To create the container, be careful about :

`docker create --name=alpine  nginx:alpine`

```sh
docker inspect -f '{{ json .State }}' <name> | python -m json.tool
To collect events of series of commands:
t0=$(date "+%Y-%m-%dT%H:%M:%S")
<series of commands>
docker events --since $t0
```

When the container is running `docker ps` and `docker ps -a` return the same output. Same when the container is paused. When the container is stopped, `docker ps` doesn't show anything but `docker ps -a` shows the process (with status EXITED).

Useful way of formatting the JSON stuff from `docker inspect`

`docker inspect -f '{{ json .State }}' alpine | python -m json.tool`
Differences between `stop` and `kill` have different exit statuses. When you stop the `SIGTERM=15`. When you kill, I suspect it should be 9, I suppose.

### Self-healing

By default containers do not restart when they die (out of memory, machine crashes and reboots). We can change this with `docker run|create --restart <policy>`. We can create a self-restarting session with `docker create --restart always nginx:alpinedocker`

Another option for self-healing are `healthchecks` i.e. periodic actions performed to make sure things are going as expected. When the container is started it is not yet immediately ready to be used. The status, while running, can be `healthy` or `unhealthy`. You can perform certain actions to check the health with `--health-cmd`. Orchestrators do this for you, so it is often not required.

TODO: run exercise 9 on your own.

## Limiting Resources

You should always limit memory with `-m` or `--memory` and CPU usage with `--cpu`. If you have one core and `--cpus=1` it will use the whole CPU.

## Passing a configuration to a container

You should not pass passwords, certificates etc to a docker container. Specify what you pass as `-e <key>=<value>`, or a file with variables `--env-file <file>`, which is also `<key>=<value>` pairs separated by newlines.

## Store data outside container - Volumes

Assume we have a docker host with a container and a filesystem on the host. With `bind mount` one can make the filesystem accessible to the container. You cannot do this on production systems. Main problem is that one could delete the files from the container. The other option is to store data in memory with `tmpfs mount` which create a temporary filesystem in memory. The third option is to use the docker engine to control part of the host filesystem so that it is not touched by other processes. This approache is called `volume`.

- `bind mount` mounts an existing host filesystem.
- `tmpfs mount` sotre data in host's memory only.
- `volume` allows creating two copies of the same container and use the same volume.

If you want to bind something from the host filesystem you should use `bind mount`. Another possibility is to copy from the host to the `volume` you create.

Create a volume called `myvol`
`docker volume create myvol`
Create two docker containers exposing two different ports but sharing the volume.
`docker run -d -p 8090:80 -v myvol:/usr/share/nginx/html nginx:alpine`
`docker run -d -p 8091:80 -v myvol:/usr/share/nginx/html nginx:alpine`
Modify the content of one of the shared files.
`docker exec 2e1d6f8aeb53 /bin/sh -c "echo EL BOMBO! > /usr/share/nginx/html/index.html"`

## Day 2

### Recap about volumes

Three types of volumes. Memory, volume and bind mount. Volumes are the preferred way of storing data.

TODO: do exercise 12.

## Networking - Drivers and Isolation

Containers are in a default *bridge network*. All containers are by default contained in the same bridge network, called `bridge` and they can communicate to each other. If we add a second docker host and we want to make the containers in the two hosts communicate, you could use an *overlay network*. They are usually used by orchestrators. You can also have the container interacting with the docker host's network. This, however, is considered unsafe. You could also have no network at all, or you could create your own bridge network. See the examples below.

## Creating a network

To create a bridge network (if the name is not given it is called `Default` by default).

```sh
docker network create <name>
```

To connect/disconnect a container to a netowrk:

```sh
docker network connect|disconnect <network_name> <container>
```

## Listing the default nets

`docker network ls` lists the active networks and their type and driver.

```sh
$ docker network ls
NETWORK ID          NAME                DRIVER              SCOPE
f71e197fee83        bridge              bridge              local
9ad1e12409c8        host                host                local
8c22f0e2a91e        none                null                local
```

TODO what is a *block size* in UNIX?

## DNS names

DNS = Domain Name Server. My computer doesn't know where google.com is. The DNS server maps the IP address to the names, and the DNS will return the IP address.

When you have a docker host you have a number of services running, possibly exposing some ports. These docker containers may be in a network called `SomeNet`. If I want to access ont container from another, since they are the same network, the y are using DNS name resolution. You can use the container name, e.g., `mydb:3306`, as the DNS name. This makes this type of connection portable. Communication is/can be bidirectional.

If I have two networks, `SomeNet` and `AnotherNet` in the same host, and I want to make them communicate. 

More in general: you create the (virtual) network (the kernel takes care of this) with `docker network create <name>`, and then you can connect/disconnext individual containers from the network with `docker network connect|disconnect`.

If we want the container to communicate with an external network, we need to *publish* a port within the container to a port visible from outside. For example, I have `mydb` in the container listening on port 3306, and this container will be accessed by some external service on port 7777. In order to do this we can do port *publishing* 7777:3306. Here I can connect to `docker_host:3306`

To connect two different subnetworks within the same docker host you should create another network. For example, let's say we have the `dummyML` container in the `MLNet` network and the `anotherML` container in the `AnotherNet` network. We can create the `YetAnotherNet` network, and connect the two containers via the *exposed* ports (more on this later). Now, `dummyML` can communicate with `anotherML` with if they are both exposing, say, port 8080.

### Publishing ports

For containers to see each other within the same network they must *expose* ports. In order to make them visible to the outside world, they must *publish* the ports.
Port publishing works only with the bridge driver. Examples:

To publish a container port to a host port

```sh
docker run -p <host_port>:<container_port>
```

To automatically *publish* all the container's *exposed* ports to random high number free ports

```sh
docker run -P
```

- **Exposing** makes ports accessible for other containers on the same docker network.
- **Publishing** makes ports accessible on the docker host. They must be unique on the host.

### Exercise 14

The images are in `filipowm/exercise-13-appX` where `X` is the number of the exercise.

Things to do for the exercise:

#### For app1

- Set the container's name
- Publish port 5000
- Configure an environment variable called APP2 = app2:5000 (the address of the other application. You should look into the container to see how this variable is used.)
- Deploy in the AppnNet network.

#### For app2

- Set name
- No port published.
- Environment variable APP3=app3:5000
- Set network to AppNet

#### For app3

- Create new directory
- Need to map <your_path> to `:/fancymodel`

**Note**: in order to make `app2` and `app3` communicate with each other we need to create an additional network (called `NewNet` here) and we need to connect these two apps to this new network. This is not explicitly mentioned in the exercise. See the code below.

#### Solution to exercise 14

```sh
# Create the network
docker network create AppNet

# Create the container for app1
docker run -d -e APP2=app2:5000 -p 8080:5000 --name app1 --network AppNet filipowm/exercise-13-app1

# Create the container for app2. No port published.
docker run -d -e APP3=app3:5000 --name app2 --network AppNet filipowm/exercise-13-app2

# Create the volume 
docker volume create fancymodel

# Create app3 and map the volume created above to the absolute path on the left
# of the : symbol
docker run -d -p 8081:5000 -v /home/vagrant/my_random_dir:/fancymodel --name app3 filipowm/exercise-13-app3

# Create another netowrk to include app2 and app3. We call this network NewNet
docker network create NewNet
docker network connect NewNet app2
docker network connect NewNet app3
```

To see what is running under the hood of `app3`:

```sh
docker exec -it app3 /bin/sh
```

This starts the shell and lets you inspect the content of the container. 

### Useful inspections

To inspect the `AppNet` network:

```sh
docker network inspect AppNet
```

To inspect the `app1` container:

```sh
docker inspect app1
```

To inspect the `fancymodel` volume

```sh
docker volume inspect fancymodel
```

To view the resources used by a container

```sh
docker stats <container>
```

To list the mapped ports

```sh
docker port <container>
```

### How to access the content of a mounted volume

In the example above, we have created `fancymodule`. To know what it contains we can use `docker volume inspect fancymodel` and this returns:

```json
[
    {
        "CreatedAt": "2019-05-21T08:32:06Z",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/fancymodel/_data",
        "Name": "fancymodel",
        "Options": {},
        "Scope": "local"
    }
]
```

The `Mountpoint` line shows where the data are actually stored: `/var/lib/docker/volumes/fancymodel/_data`

Note that we didn't explicitly expose port 5000. In this case, this was not required because the exposed ports are already specified in the dockerfile. There is an `--expose` flag when you run a container. This is why we didn't need to do it explicitly. This is a good practice, because when you are building your application you already know which port you'll be using. `--expose` only means that I have a port (say 5000) and I am running some service on it. Publishing requires the `<host>:<container>` syntax. Exposing only requires the port.

## Object Metadata

Goal: make objects findable. It is possible to attach a label to any docker object (networks, containers, images, volumes). For example:

```sh
docker run \
	   --label <key>=<value> \
	   --label-file <file>

docker volume create --label <key>=<value>
```

#### Filtering

Possible to filter objects by label with the `-f` option. It is also possible to remove images, containers etc by label. For example:

```sh
docker rmi -f $(docker images -qf 'label=app=myapp'
```

It is not possible to remove networks when containers are running inside them. You first need to disconnect any container running within the network. `docker network disconnetct` wants exactly two arguments. To remove all unused things you can use `docker prune`. This will remove unused images, containers, networks etc.

You can first remove all the containers and then use docker `system prune` to remove whatever is left. In practice:

```sh
docker rm -f $(docker ps -aq)
docker system prune -a --volumes
```

## Recap

We have seen

1. Container lifecycle (`docker inspect`, `docker events`).
2. Self healing: restart policies are always, on-failure, unless-stopped, no.
3. Limiting resources.
4. Storing data.
5. ...

## Module 3: Creating and running your own images

A `Dockerfile` is an image recipe on which you run `docker build` and this creates the image. Once the image is built, you can run `docker run` to start the container.

### Image formats

The general format is `[repository] [owner/]<name>[:tag]`. For example:
In `ubuntu:disco` `ubuntu` is the `<name>` and `disco` is the `tag`. If the tag is not specified it is `latest` by default.
In `filipowm/exercise-12`, `filipowm` is the owner and `exercise-12` is the image name.

### Official images

The `hub.docker.com` repository contains the official versions of various images. These do not have an owner, but one can create his/her own images. You can access the images with `docker pull`. You can create both public and private images. Once private, other people won't be able to find the images. Roche has several repositories. `Artifactory` is one.

When we run an image often we see the message `downloading`. What is being downloaded are the **layers**. This is like an instruction execution of the recipe. Some layers have zero size because they are just executing a command (for example the `echo` command), while others are adding files to the image, installing software to the image, etc.

These layers translate into something like the code below.

```sh
CMD ["/nginx", "-g", "daemon off"]
RUN apt-get update && apt-get install ....
COPY . /usr/bin
```

Having separate layers increases performance and the speed of building. The container reuses the layers and adds a *read/write* layer. Containers are re-using layers.

### Copy on write strategy

My container may want to create a new file. It will store it in the read/write layer. If you want to write to another layer, you can create a copy and move it into the read/write layer. This is why they are so fast: things are reused until we need to modify them. Docker `diff <container>` shows which layers have been modified or deleted.

### Docker commit (clarify)

Layers are added at each modification of the file, and this makes them grow.
We don't have only one layer because rebuilding only the layer that has been modified makes rebuilding the whole image much faster. It is possible to squash some layers, and one can play with it.

## Dockerfiles

Instructions in a docker file adds layers. It is usually a good practice to start from an official image. The one below contains Ubuntu (a small version of it?)

```sh
FROM ubuntu:latest
```

This copies the content of the current local directory into a Docker volume called `/app`.

```sh
COPY ./ /app
```

This is run when building the image.
```sh
RUN apt-get update && apt-get install vim
```

### ENTRYPOINT, CMD and RUN

- `RUN` is executed when building the container.
- `ENTRYPOINT` and `CMD` are executed when starting the container. They have PID 1 (first executed). One cannot override the `ENTRYPOINT` but can override the `CMD` when using `docker run ... [command]`.

Only layers create files or take space, the "history" layers do not.

## Building images

`docker build -t <name>[:tag] [-f <path_to_dockerfile>] <build_context>`

Here `<name>` is the image name. It can be `myimage` or `sth/myimage` and it can have a tag, e.g., `sth/myimage:1.0`. Default tag is `latest`. The path to dockerfile is optional and should be specified when the dockerfile is non-standard (custom name) or when it doesn't exist in the `<build_context>`. The build context could be any path. If I have `COPY . /` in the dockerfile and the context is `/home/app` then `COPY . /app` will copy the content of `/home/app`.

If the dockerfile is in `/home/app` and the files are in `/tmp/test` I will pass the path to the dockerfile with `-f /home/app/Dockerfile` and the context as `/tmp/test`.

```sh
dockr build -t myapp -f /home/app/Dockerfile /tmp/test
```

### Exercise 14

```sh
docker build -t giovanni_dario .
```

This build pulls the ubuntu image from the docker hub. To inspect an image 

```sh
docker image inspect ubuntu:latest
```

but, better to see the various layers, use

```sh
docker image history ubuntu:latest
```

Note that the path to the dockerfile must contain the dockerfile name. It can be either an absolute or a relative path.

We then should use the same commands with the `giovanni_dario` image.
We can run the image with `docker run giovanni_dario ls /app`

When we change the context the files that are copied are different.

### Exercise 15

Build a docker image for a Python application. A minimal Dockerfile could be

```sh
FROM python:3.6.8-slim-jessie
LABEL owner=gdario
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT python -u app.py
EXPOSE 8080
```

Another (better) possibiltiy would be

```sh
FROM python:3.6.8-slim-jessie
LABEL owner=gdario
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
EXPOSE 8080
```

To run this image type: `docker run -d -p 8080:8080 --name flasky my_flask_app`

In general, try to start from the smallest possible public image and then try to add your own. For example, you can use `alpine` or `slim` related official images. They are, by construction, very low-weight versions.

## ONBUILD

It postpone action execution. For exapmle, this code

```sh
FROM ubuntu:18.04

ONBUILD COPY ./ /app
...
```

will run only when we execute

```sh
FROM filipowm/ubuntu-base:18.04
RUN apt-get install -y pip
...image.
```

In other words, it is used for building parent 

### Exercise 16

The CMD option in the dockerfile is the command that is passed to `docker run`, and it can be overridden. For example, in the dockerifile, we have `CMD ["app.py"]` but when we run `docker run ... app.py` the content of `CMD` (`main.py`) is overridden by `apppy`. If in `docker run` you don't specify the command, it will use the content of `CMD`.
Similarly, you can override the `ENTRYPOINT` with the `--entrypoint` option.
By running `docker run -it` you can launch a terminal

## Publishing images

Where should we store our images? We can specify multiple tags for an image.

```sh
docker tag web:latest web:1.0
docker tag web:latest filipowm/web:training
docker tag app web
...
```

A registry is a place where you store images. A repository is a place within the registry where you push. You need to login with `docker login <registry_address>`. If you are not pushing to docker hub you must provide the registry address, otherwise it can be omitted.

There is a command `docker push` which is the opposite of `docker pull`. For example

```sh
docker push filipwm/web:latest
```

will push to the remote repository. Similarly you can use `docker pull`.

### Image versioning

When you want to version an image, use tags. You should never rely on the `latest` tag in a production system. Always rely on exact version numbers. Semantic versioning is also applicable, by using the syntax: `MAJOR.MINOR.PATCH[.labels]`, e.b. `7.13.2.RELEASE`, `7.14.1.ALPHA`, etc.

It's possible tu use custom names. Labels are more for metadata, for example, you can have 

```sh
LABEL project=<profject_name>
APPLICATION ...
```

- `rchdckr/<first_name>_<last_name?:latest`
- tag renamed image with tag 1.0
- Push image. It should appear in https://hub.docker.com/search?q=rchdckr&type=image
- Logging: user: ******* Password: *************

When pushing, you can see that some layers are pushed, and several are recycled.

The Roche repository is `repository.intranet.roche.com:6555`
We can use `repository.intranet.roche.com:6555/may/<first_name>_<last_name>`

## Docker file best practices

- Versioning.
- Meaningful names.
- Good/useful base parent images.
- Reusability.
- Start small with the minimal set of things that work for you, and add what you need.
- Define exposed ports.
- Documentation.
- Minimize the number of layers but still. Try to put as much as makes sense into a RUN command.

```sh
RUN apt-get update && \
	apt-get install curl && \
	cp /tmp/my_file /somewhere_else
```

## Containers worst practices a.k.a. when you shouldn't use containers

Don't use containers if:

- You have extreme security requirements.
- You have to run legacy apps (e.g. dot.net or java 1.2 - you may not find the image).
- Highly stateful applications like HDFS or Hadoop.
- Exotic libraries.
- Databases. This is more controversial.
- Licensing per core.

## Day 3

### Docker compose

In exercise 13 we had to do a number of things manually. This is error prone, but there is the `docker-compose` for this. It allows to define whole architectures as a YAML file, where you can define the various components, then you run `docker compose up`.

### Format of the YAML file

The file is separated into db service definition and volumes (top of the file below) and networks definition (bottom part).

Note that the `app` depends on the db. The app will not start up if the database is not there.

```yml
version: '3.7'
services:
	db:
		image: mysql:5.7
		volumes:
			- ...
		networks:
			= ...
	app:
		depends_on:
			- db
		image: filipowm/myapp:latest
		networks:
			- app_net
		ports:
			- "8000:80
			
volumes:
	db_data: {}
networks:
	app_net
```

### Exercise 19

Re-create the same architecture as exercise 13 but now with a `docker-compose.yml` and run it with `docker-compose up`

#### Solution

Note that I had a problem with the version: originally the version was 3.7, but it failed. The syntax below is a bit different from the recommended one. This may be due to the different version. Not completely clear.

```yml
version: "3.3"
services:
        app1:
                image: filipowm/exercise-13-app1
                networks:
                        - AppNet
                environment:
                        APP2: "app2:5000"
                ports:
                        - "8080:5000"
        app2:
                image: filipowm/exercise-13-app2
                networks:
                        - AppNet
                environment:
                        APP3: "app3:5000"
        app3:
                image: filipowm/exercise-13-app3
                networks:
                        - AppNet
                volumes:
                        - /home/vagrant/my_random_dir:/fancymodel
                ports:
                        - "8081:5000"
networks:
        AppNet:
```

Doing CTRL-C will stop the process, but will not remove the images. For this you need to do `docker-compose down`. This will remove the images **but not the volumes**. We can run the same in the background with `docker-compose up -d` (note, it's **not** `docker-compose -d up`.

Apparently you can use both these syntaxes:

```yml
networks:
	- AppNet
	
networks:
	AppNet:
```

If we change the port in `app3` and re-run `docker-compose up -d`, only `app3` will be updated.

### Container building strategies

Mat showed an example where he created a UNIX-based image, and started adding modules with `apt-get install` until he had what he needed. That's a way. Another way is to start from a large image and remove as much as you can. This seems more complex, however.

## Module 2 - Kubernetes

### Bases of modern IT

- Serverless
- Kubernetes
- Infrastructure as a code
- Hybrid and multi cloud
- Innovative SaaS

#### Serverless

In a serverless architecture, you may write some ML model, and do a "serverless deploy", which automatically creates a container, puts the code there, and exposes the needed ports. This greatly simplifies how you ship your applications. It can automatically scale, by creating new instances, when the load is too high. This is, in Mat's opinion, the best way to ship a ML model. We don't have it yet on premises, but that's what AWS (Lambda?) offers, as well as the other cloud providers. The model is expressed as a YAML or JSON file, and this makes it much easier to run on different platforms.
These systems organize the following hierarchy.

- Application (NodeJS, Java, Python etc)
- Platforms (Kubernetes etc)
- Service (Kafka, Oracle, etc)
- Infrastructure (compute, storage, entwork)

How to ensure that the cargo is delivered even if one ship sinks? This can happen for multiple reasons. The basic solution is to replicate the containers, still trying to be efficient.

### Load Balancing

Imagine we have two containers, c1 and c2, both listening on port 5000. The load balancer will take care of how to send and receive messages from the two containers. Kubernetes makes this very easy, but the application must be ready for the load-balancing.

### Container orchestration

- Provisioning and deployment of containers.
- Redundancy and availability of containers.
- Scaling up or removing containers to spread application load evenly across host infrastructure.
  - Example: we have two nodes: N1 and N2. Two containers are running on N1 and one on N2. If we need to add a new container, the balancer will inspect the resources usage. To make things a bit safer it will distribute replicas on multiple nodes. If one node dies, the container will be rescheduled on a still working node.
  - Other example: if my container is hitting 80% of CPU usage, I can tell the orchestator to create new containers, but not more then 5, for example.
- Allocation of resources between containers:
  - How many resources must the application get
  - What is the maximum resources the application can use.
- Intelligent routing:
  - I have two users: a Swiss and a German one.
  - I have a Load Balancer (LB) and a Reverse Proxy (RP).
  - This will address the Swiss user to the Swiss user and the German to the German one. Other use case is for free and paying users of a service.
- Logs from multiple containers are collected and can be easily monitored.

### General architecture of orchestrators

Similar to the one above:

- Persistent storage.
- Networking and isolation (e.g. "these two containers should not talk to each other")
- Container orchestrator
- Docker host (these could be VMs, physical servers, cloud providers). If you are using the cloud and you are running out of resources it will spawn new containers keeping in the budget limits you fixed.
- Infrastructure

Example: Two nodes N1, N2 have a container each: C1, C2. These containers need to share data. For this we need a **persistent storage** layer that can be accessed by both nodes/containers. This could be a NFS storage.

Container Orchestration is an abstraction layer put on top of infrastructure. Docker Swarm was originally developed by Docker, but it is deprecated now, and Docker itself recommends using Kubernetes. Other possibilities are Marathon (layer on top of Mesos (apparently great for data management centers and handling HDF5 (or HDF?) files).

#### Terminology
- **Pods**: usually one container per pod, but tighly connected containers could be into the same pod.
- **Replication controller**: provide a pod template for creating any number of pod copies. 
- **Services** pods may come and go, but the address of services is persistent.
- **Volume** are locations where pods can store data, both persistent and ephemeral.
- **Namespace** segments pods, rcs, volumes etc together.

### Kubernetes Architecture

There is a 
- Kubernetes Master node.
- `kube-apiserver`, a container exposing the K8 API.
- etcd is a distributed and consistent key-value store, used to store its state It stores all cluster data
- kube-scheduler: it manages how to deploy containers (i.e. how to schedule pods) It takes into account the resources required and hardware/software/policy constraints. Individual and collective resource requirements. It checks inter-workload interference.
- kube-controller-manager: responds when a node goes down. It manages the declared number of pods and check that they are running.
- [optional] cloud-controlloer-manager:  it interacts with cloud providers, schedules new nodes etc.

This whole thing is the **K8 Control Plane**.

#### Node

In the node we must have a **container engine** (not necessarily Docker, can be Cri-o, Singularity etc). It must also have a `kubelet` which talks to the engine and creates a container based on the pod specification. There is one kubelet per node, and it manages multiple pods. The `kube-proxy` handles all the networking and provides all the service abstraction. It is the most complex component. Finally we have the pod which contain a container. To all this we can add monitoring, load balancers, etc. Nodes talk to the kube-apiserver using kubelet and kube-proxy.

Let's assume we have 3 data centers. One can label a node in a DC, and specify, for example, that one node is in Basel (BAS) and needs a GPU (GPU). If I want to run pods only in Basel and Kaiseraugs, I can use these labels for this.

### K8 Ecosystem

Various cloud providers offer their own K8 versions.Google Kubernetes engine is supoerior to all the others, according to Mat. In these cases, it's the vendor who manages K8. The alternative is to set up your own K8.
On top of the platform there are a lot of services: CI/CD, telemetry, service catalog, logging, security.

### Minikube

It's a mini K8 that you can use to play with K8 on your local machine.

```sh
kubectl get pod -n kube-system
```

Will print all pods.

### K8 connection configuration

```yml
apiVersion: v1
kind: Cnfig
clusters:
- cluster:
    ...
  name: minikube
contexts:
- context:
	cluster: minikube
	user: minikube
...
```

You can easily switch contexts and switch users, for example for debugging purposes.

#### Managing objects

- Imperative commands (like in Docker. Shouldn't be used in productiona)
- Imperative with configuration files (YAML). If you make manual changes in the system, it will not know. If the system autoscales from 5 to 10 pods, and you reapply the script, you will lose those 5 additional pods.
- Declarative with configuration files. This is the same as above, but it can see when the state changes.

#### Required fields

- `apiVersion`: version of the K8 API.
- `kind`: type of object you are defining.
- `metadata`: data that help uniquely identify the object.
  - You can and *should* have labels.
- `spec`:

#### Namespaces

One can organize projects and environments based on namespaces. For example I can have a `projectA-dev` namespace, a `projectA-prod` namespace etc. The defulat namespace is called `default`.

one can override the namespaces with 
```sh
kubectl config set-context <context_name> --namespace <namespace_name>
```

#### Exercise 1

Create 3 namespaces

1. create namespace
   - using imperative command
   - using imperative object configuration
   - using declarative object congfiguration

Example of namespace file:

```yml
apiVersion: v1
kind: Namespace
metadata:
	name: my-namespace
```

You can group one or more containters with shared storage, network and specification. All containers in a pod are always co-located on the same node and they are co-scheduled. We could have one pod, `pod-A` with one container and `pod-B` with two more.

In a pod you should run *one* process per container. This makes monitoring simpler.

```yml
apiVersion: v1
kind: Pod
metadata:
    name: noginxpod
spec:
    containers:
		- image: nignx:alpine
		name: nginx
		volumeMounts:
			- montPath: /app
			name: my-volume
		env:
		- name: MY_ENV_VAR
		value: "First variable"
	resources:
		limits:
			memory: "200Mi"
			cpus: "500m"
		requests:
			memory: "100Mi"
	command: ["ls"]
	args: ["-a"]

	volumes:
		- name: my-volume
		emptyDir: {}
```

#### Exercise 2

This YAML file works. Pay attention to the indentation! The one above is incorrect.

```yml
apiVersion: v1
kind: Pod
metadata:
        name: noginxpod
spec:
        containers:
                - image: nignx:alpine
                  name: nginx
                  ports:
                        - containerPort: 80
                          hostPort: 30080
```

#### ReplicaSet

Maintains a stable set of replica Pods running at any given time. Its responsibility is scaling-up/down our pods. You usually never create them by hand, as they are abstracted from you. What you usually configure is not individual pods, but rather the **Deployment** abstraction layer as a whole.

You create a Deployment configuration file, similar th the one above, but with

```yml
kind: Deployment
metadata:
	labels:
		app: my-app
	name: my-app
spec:
	replicas: 2
	selector:
		matchLabels:
			app: my-app
	template:
		metadata:
			labels:
				app: my-app
	    spec:
			...
			This is the same as in the pod
```

The template defines what the pod is about. `matchLabels:` etc specifies which pods should be replicates. Note that the restart policies in the rule above are for the whole pod, not for the individual image.

### Core K8s commands

```sh
kubectl get
kubectl describe
kubectl delete
```

and you can use these on `pod`, `deployment`, `svc`. For example, in our case we had:

```sh
kubectl get deployment
NAME     READY   UP-TO-DATE   AVAILABLE   AGE
my-app   0/2     1            0           4m5s
mypod    1/1     1            1           36m
```

To remove these two deployments:

```sh
kubectl delete deployment my-app
kubectl delete deployment mypod
```

We can inspect the individual pods with 

```sh
kubectl get pod
```

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
        labels:
                app: my-app
        name: my-app
spec:
        replicas: 2
        selector:
                matchLabels:
                        app: my-app
        template:
                metadata:
                        name: my-app
                        labels:
                                app: my-app
                spec:
                        containers:
                                - image: makocchi/docker-nginx-hostname
                                  name: nginx
                                  ports:
                                        - containerPort: 80
                                          hostPort: 30095
                        restartPolicy: Always
```

### Services and load balancing

A service exposes our applications, and represent another layer. It could expose ports (`NodePort`), pods localized on a certain cluster (`ClusterIP`) and the load balancer (`LoadBalancer`).

### Security and RBAC basic

Imagine that we have two services svc1 and svc2. A user can interact with a single address `myapp.kube.roche.com`. This address is specified in `Ingress`, which will specify it for me. Ingress than could re-address to two different services. Ingress will act as a reverse proxy.
- Ingress
  - svc1
  - svc2

`CronJob` would be the thing to use if you want to have an automated data refresh (daily, weekly, whatever) for a ML application. You don't need to hardcode it in the app, but may be a cron job dedicated to this.

RSI has a central thing called Rancher which has access to K8s from most cloud vendors. You can ask to have access.

## Capabilities @ Roche

- Enterprise Container as a Service (using Mesos + Marathon).
  - production grade.
  - qualified.
  - no K8s (using an obsolete version of Marathon).
  - missing shared services.
  - L1/L2 data only.
- RSI Elastic Compute (K8s)
  - Not qualified.
  - Immature and unstable.
  - Some tooling, but not suffucient.
  - No shared services.
  - Only scientific workloads (and this will not change).
  - Still under development.
- gRed RAX K8s
  - Not qualified.
  - Comprehensive tooling.
  - Shared services.
  - Any workload.
  - Not yet fully ready.
  - gRed only.
- gRed HPC (Rosalind - Singularity only)
  - Not qualified.
  - No tooling?
  - No shared services.
  - Only intensive scientific workdloads.
  - Not an orchestration really.
  - gRed only?

