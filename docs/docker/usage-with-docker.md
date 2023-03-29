# Run the simulator within a docker environment

If you are on other system then linux, please use the docker for developing purposes and small (local machine) tests.
For this purpose use the follwing [Dockerfile](Dockerfile).

1. Clone the simulator repository. This will repository will be used for further development.

```shell
git clone https://github.com/It4innovations/ruth.git
```

2. Build the docker image within the folder with Dockerfile

```shell
docker build -t ruth .
```

3. Run the docker file with mounted volumes

```shell
docker run -it -v </absolute/path/to/host/ruth>:/workdir -v <absolute/path/to/host/datadir>:/data --name ruth-con ruth 
```

The following commands are performed within the docker container.

4. Install ruth.

```shell
python3 -m pip install -e .
```
The default folder in running container is `/workdir`. Which is the same one where ruth from host is mapped. The `-e` option is important as it allows us to modify the ruth source code on host machine and directly test it within container without re-installation.





