# Dockerfile for usage with arlib library

The arlib library for alternative routing written in C++ enforces the target platform to be linux. For the development purposes there is a docker file used to prepare the enviroment for testing the simulator.


- **It works only on arlib branch of ruth repository.**

- **It can be used for testing the simulator after local changes within the ruth.** `py-arlib` (python binding of C++ library) is fixed and downloaded from the repository.

You can prepare the Docker image by running the following command from the `ruth` root directory. So you clone the `ruth` repostory, enter it, and call the command.

``` shell
docker build -t ruth-with-arlib -f docker/arlib/Dockerfile .
```

Once the image is buld you can run the container. After exiting the container it is removed so next time you have a fresh environment. If you need access to a host data/folder for output, mount a volume and use it as a path for input/output paramters.

``` shell
 docker run -it --rm --name ruth-with-arlib.con ruth-with-arlib:latest
```

Now you can test simulator

``` shell
ruth-simulator --departure-time="2021-06-16 07:00:00" --k-alternatives=4 --nproc=8 --out=simulation_record.pickle --seed=7 rank-by-prob-delay /ruth/benchmarks/hello-world/vehicles-state.parquet 70 500
```

## Make changes in ruth and test them

Now any time you make a change you want to test you build an updated image with the same command.
``` shell
docker build -t ruth-with-arlib -f docker/arlib/Dockerfile .
```

As there is a previous version it just copy the local files and install ruth but all the dependencies are already installed hence the process is much faster. Then you can test it in the same way.

If you make a change in `requirements.txt`, the installation build can take more time.

If you make a change in `py-arlib` please delete the image and build it from scratch.

``` shell
docker image rm ruth-with-arlib
docker system prune
```


