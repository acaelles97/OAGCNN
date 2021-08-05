# OAGCNN: Object Aware Graph CNN for multiple VOS
Project done during the Introduction to Research subject at UPC, Master in Advanced Telecommunications Technologies. You can check the details of the project here: [OAGCNN_paper.pdf](https://github.com/acaelles97/OAGCNN/files/6937412/OAGCNN_paper.pdf)

# Abstract
Video object segmentation has increased in popularity since the release of Davis2016 in which a single object had to be segmented. With the release of Davis2017 and YouTube-VOS the task moved to multiple object segmentation, increasing in difficulty. In this work we focus on this scenario, presenting a novel graph convolutional neural network that has the sense of each object of the video sequence in each node, working entirely in the feature space domain. The nodes are initialized using an encoder that takes as input features from the image together with each object’s mask. After graph message passing, we use a decoder on each final node state to segment the object that node is referring to.

![OAGCNN_model_figure](https://user-images.githubusercontent.com/46324089/128324753-9c10c23a-d752-44db-b5f4-4c840d564d61.png)
