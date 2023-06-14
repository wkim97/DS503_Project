# DS503_Project

# Setting

- conda version : 22.9.0
- conda-build version : 3.22.0
- python version : 3.9.13.final.0
- virtual packages : __cuda=12.1=0
- platform : win-64
- user-agent : conda/22.9.0 requests/2.28.1 CPython/3.9.13 Windows/10 Windows/10.0.22621

## Packages

Check DS503_Packages.txt

# Dataset

[FashionMNIST — Torchvision 0.15 documentation (pytorch.org)](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)

[AI-Hub (aihub.or.kr), 페르소나 기반의 가상 인물 몽타주 데이터](https://aihub.or.kr/)

# How to use

## Jupyter notebook

1. Go to DS503_Project/DS503_tutorial.ipynb
2. The first cell is for setting. You don’t need to modify any codes.
    1. If you want to use Fashion_MNIST data, input ‘fashion’.
    2. If you want to use face data, input ‘face’.
3. You can use our method in the second cell.
    1. First, pick one target concept such as ‘dark short-sleeved shirt’(If you want go with Fashion_MNIST).
    2. Pick one image that is closest to the target you think of and put the number.
    3. Repeat.
    4. If the model works well, you’ll get only images that is close to your target concept.
