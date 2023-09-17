# Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need
<div align="center">

<div>
    <a href='http://www.lamda.nju.edu.cn/zhoudw' target='_blank'>Da-Wei Zhou</a><sup>1</sup>&emsp;
    <a href='http://www.lamda.nju.edu.cn/yehj' target='_blank'>Han-Jia Ye</a><sup>1</sup>&emsp;
    <a href='http://www.lamda.nju.edu.cn/zhandc' target='_blank'>De-Chuan Zhan</a><sup>1</sup>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu </a><sup>2</sup>
</div>
<div>
<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University&emsp;

<sup>2</sup>S-Lab, Nanyang Technological University&emsp;
</div>
</div>

The code repository for "[Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need](http://arxiv.org/abs/2303.07338)" in PyTorch.  If you use any content of this repo for your work, please cite the following bib entry: 

    @article{zhou2023revisiting,
        author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
        title = {Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need},
        journal = {arXiv preprint arXiv:2303.07338},
        year = {2023}
    }





## Updates
[03/2023] [arXiv](http://arxiv.org/abs/2303.07338) paper has been released.

[03/2023] Code has been released.

## Introduction
Class-incremental learning (CIL) aims to adapt to emerging new classes without forgetting old ones. Traditional CIL models are trained from scratch to continually acquire knowledge as data evolves. Recently, pre-training has achieved substantial progress, making vast pre-trained models (PTMs) accessible for CIL. Contrary to traditional methods, PTMs possess generalizable embeddings, which can be easily transferred for CIL. In this work, we revisit CIL with PTMs and argue that the core factors in CIL are adaptivity for model updating and generalizability for knowledge transferring. 1) We first reveal that frozen PTM can already provide generalizable
embeddings for CIL. Surprisingly, a simple baseline (**SimpleCIL**) which continually sets the classifiers of PTM to prototype features can beat state-of-the-art even without training on the downstream task. 2) Due to the distribution gap between pre-trained and downstream datasets, PTM can be further cultivated with adaptivity via model adaptation. We propose **ADapt And Merge** (ADAM), which aggregates the embeddings of PTM and adapted models for classifier construction. ADAM is a general framework that can be orthogonally combined with any parameter-efficient tuning method, which holds the advantages of PTM’s generalizability and adapted model’s adaptivity. 3) Additionally, we find previous benchmarks are unsuitable in the era of PTM due to data overlapping and propose four new benchmarks for assessment, namely ImageNet-A, ObjectNet, OmniBenchmark, and VTAB. Extensive experiments validate the effectiveness of ADAM with a unified and concise framework.

<div align="center">
<img src="imgs/adam.png" width="95%">

<h3>TL;DR</h3>

A simple baseline (**SimpleCIL**) beats SOTA even without training on the downstream task. **ADapt And Merge** (ADAM) extends SimpleCIL with better adaptivity and generalizability. Four new benchmarks are proposed for assessment.
</div>



## Requirements
### Environment
1. [torch 1.11.0](https://github.com/pytorch/pytorch)
2. [torchvision 0.12.0](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)


### Dataset
We provide the processed datasets as follows:
- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**:  Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ImageNet-A**:Google Drive: [link](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL)
- **OmniBenchmark**: Google Drive: [link](https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA)
- **VTAB**: Google Drive: [link](https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EQyTP1nOIH5PrfhXtpPgKQ8BlEFW2Erda1t7Kdi3Al-ePw?e=Yt4RnV)
- **ObjectNet**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) if the file is too large to download.

These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

The md5sum information can be found in this [issue](https://github.com/zhoudw-zdw/RevisitingCIL/issues/5).

You need to modify the path of the datasets in `./utils/data.py`  according to your own path.

## Running scripts
Please follow the settings in the `exps` folder to prepare your json files, and then run:

```
python main.py --config ./exps/[configname].json
```


## Acknolegment
This repo is based on [CIL_Survey](https://github.com/zhoudw-zdw/CIL_Survey) and [PyCIL](https://github.com/G-U-N/PyCIL).

The implemenations of parameter-efficient tuning methods are based on [VPT](https://github.com/sagizty/VPT), [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer), and [SSF](https://github.com/dongzelian/SSF).

## Correspondence
If you have any questions, please contact me via [email](mailto:zhoudw@lamda.nju.edu.cn) or open an [issue](https://github.com/zhoudw-zdw/RevisitingCIL/issues/new).


<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=zhoudw-zdw.RevisitingCIL&left_color=green&right_color=red)

</div>
