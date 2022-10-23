**A facial keypoint extraction network demo, built with TensorFlow 2.**

### Network Architecture
The network is a transfer learning of [`ResNet-50`](https://arxiv.org/abs/1512.03385). We added an input convolution layer and several output dense layers.

### Dataset
This demo is trained on the Kaggle [Face Images with Marked Landmark Points dataset](https://www.kaggle.com/datasets/drgilermo/face-images-with-marked-landmark-points), which predicts 15 keypoints given a grayscale facial image.

### Notice
This repo is currently only a demo, and the network is designed purely for internal research purposes. Further instructions on usages, etc. might be available in the future.
