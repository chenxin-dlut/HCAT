# Nvidia AGX系列搭建深度学习环境

可以参考[agx深度学习环境搭建](https://blog.csdn.net/qq_41426807/article/details/124705416)

## 一、查看自己的AGX内置环境

```
uname -a #查看系统版本
jetson_release -v#查看jetson版本
nvcc -V #查看cuda版本
cat /usr/include/cudnn.h |grep CUDNN_MAJOR -A 2 #查看cudnn版本
```

##  二、下载自己对应版本的miniforge(anaconda的替代品)参考[下载安装miniforge](https://blog.csdn.net/weixin_46025237/article/details/121150859?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166749137016782428652881%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166749137016782428652881&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-121150859-null-null.142%5Ev63%5Econtrol,201%5Ev3%5Econtrol_2,213%5Ev1%5Econtrol&utm_term=nvidia%20%E5%AE%89%E8%A3%85miniforge&spm=1018.2226.3001.4187)

下载正确之后就可以创建虚拟环境

## 三、安装pytorch,torchvision

根据[pytorch for jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 安装自己对应版本的pytorch,并按照官网提供的安装方法进行安装

torchvision同样按照[pytorch for jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) 官网提供的方法进行安装，(版本对应pytorch!!!!!)

# 四、验证pytorch,torchvision是否安装成功

```python
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
import torchvision
print(torchvision.__version__)
```

验证成功之后就可以接着安装其他所需要的包