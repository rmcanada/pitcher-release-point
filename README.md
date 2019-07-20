## Install

### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+
- opencv3

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://github.com/rmcanada/pitcher-release-point.git
$ cd pitcher-release-point
$ pip3 install -r requirements.txt
$ pip3 install requests
$ pip3 install Cython
$ sudo apt-get update
$ sudo apt-get install ffmpeg
```

Build c++ library for post processing.
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

### Download Tensorflow Graph File(pb file)

Before running demo, you should download graph files. You can deploy this graph on your mobile or other platforms.

CMU's model graphs are too large for git, so I uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ chmod +x download.sh
$ ./download.sh
```

### Test Inference

You can test the scripts by running the following scripts. Change the value entered into --guid for any pitch that is in the current directory .json files.

```
$ python3 utils.py --guid 896c23c5-aa0a-473b-9f2b-859c6d88928b
$ python3 run.py --model=cmu
```
The terminal will run analysis for each frame in the video. 
Once finished, it will print out the filename of the frame it has selected. 
It has also copied the image into the current working directory and created an output.mp4 stitched together from the pitcher's pose estimations. 
It will also print out a list of candidate frames (frame number, arm length, right (567) / left (234) arm). The largest arm length is selected.

