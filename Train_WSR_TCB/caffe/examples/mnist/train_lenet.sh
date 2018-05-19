#!/usr/bin/env sh
set -e

/gdata/liuzhd/data5/HED_NEW/caffe/build/tools/caffe train --solver=/gdata/liuzhd/data5/HED_NEW/caffe/examples/mnist/lenet_solver.prototxt $@
