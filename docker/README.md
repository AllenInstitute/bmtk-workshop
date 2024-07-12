### Building the image

```bash
$ docker build -t workshop-2024 .
```

### Running the container

```bash
$ docker run -p <client-port>:8888 workshop-2024
```


### How to use docker image to edit local version of the bmtk-workshop

1. You can either make a fork of the [updates/2024-add-tuts](https://github.com/AllenInstitute/bmtk-workshop/tree/updates/2024-add-tuts) branch
of bmtk-workshop in github into your own repo. Then clone it into your own machine
```bash
 $ git clone --branch updates/2024-add-tuts https://github.com/AllenInstitute/bmtk-workshop.git
 $ cd bmtk-workshop
 ```

2. Next step is to build the docker image. The first time you do it it may take a bit of time, be patient
```bash
 $ cd docker/
 $ docker build -t workshop-2024 .
```
This will create a docker image with NEURON, NEST, and a special [bmtk of branch](https://github.com/kaeldai/bmtk/tree/update/workshop-2024) that includes
some required changes to run new tutorials.

3. Startup a docker container using the built `workshop-2024` image 
```bash
$ cd .. # eg path to where bmtk-workshop was cloned into
$ docker run -v $(pwd):/home/shared/bmtk-workshop -p 9001:8888 workshop-2024
```
About these option
* The `-v $(pwd)::/home/shared/bmtk-workshop` says that you will share the current working director `pwd` with the docker container
* The `-p 9001:8888` means that even though inside the docker container jupyter-lab is using port `8888`, on you're computer it is using port `9001` (you can change this value to use a different port number)

4. Test this out by opening [http://127.0.0.1:9001/lab/tree/TableOfContents.ipynb] with a browser. You should see the table of contents page with links to all the new tutorials. 

**NOTE**: This should be the same files you see in you local bmtk-workshop folder, and any changes you make in the jupyter-lab instance running inside docker should also show up on you local machine. One good way of testing this is that in jupyter-lab there is a option for creating `Test File`. Go ahead and create a text file in jupyter-lab and add some text, then open up a file-explorer and verify you can see the file.

5. You can make changes as you see and when you are ready to push them back to github using a command line:
```bash
$ git add ChapterIChanged.pynb
$ git commit -m "Im making my changes yo!"
$ git push origin updates/2024-add-tuts
```

When you are done you can close docker through jupyter-lab by clicking on `File > Shut Down`. You can always restart it back up using Step #3 





