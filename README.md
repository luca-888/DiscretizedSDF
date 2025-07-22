# Gaussian Splatting with Discretized SDF for Relightable Assets


This repository contains the official implementation of the following paper:
> **Gaussian Splatting with Discretized SDF for Relightable Assets**<br>
> [Zuo-Liang Zhu](https://nk-cs-zzl.github.io/)<sup>1</sup>, [Jian Yang](https://scholar.google.com/citations?user=0ooNdgUAAAAJ)<sup>1</sup>, [Beibei Wang](https://wangningbei.github.io/)<sup>2</sup><br>
> <sup>1</sup>Nankai University  <sup>2</sup>Nanjing University<br>
> In ICCV 2025<br>

[[Paper](https://arxiv.org/abs/2507.15629)]
[[Project Page](https://nk-cs-zzl.github.io/projects/dsdf/index.html)]
[Video (TBD)]

DiscretizedSDF is an **efficient, robust** solution for object relighting, aiming to produce decent **decompositions of geometry, material, and lighting** for **multi-view observations**.

<p align="middle">
<img src="demo/horse_golf.gif" width="32%"/><img src="demo/angel_corridor.gif" width="32%"/><img src="demo/potion_golf.gif" width="32%"/>
</p>


## News
- **Jul. 21, 2025**: Our code is publicly available.
- **Jul. 22, 2025**: Our paper is publicly available on ArXiv.
- **Jul. 22, 2025**: Release pretrained models.


## Method Overview
![pipeline](https://nk-cs-zzl.github.io/projects/dsdf/assets/images/overview.png)

For more technical details, please refer to our paper on [arXiv](https://arxiv.org/abs/2507.15629).

## Dependencies and Installation
1. Clone repo.

   ```bash
   git clone https://github.com/NK-CS-ZZL/DiscretizedSDF.git
   cd DiscretizedSDF
   ```

2. Create Conda environment and install dependencies
   ```bash
   conda create -n dsdf python=3.10
   conda activate dsdf
   pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   git clone https://github.com/NVlabs/nvdiffrast
   pip install ./nvdiffrast
   pip install ./submodules/fused-ssim
   pip install ./submodules/diff-surfel-sdf-rasterization
   pip install ./submodules/simple-knn
   ```
    **Note that** 
    + Our code is verfied under CUDA11.8 runtime, so we recommend to use the same environment to guarantee reproductibility.
    + Please switch to the corresponding runtime if the NVCC version is higher than 11.8. 

3. Download pretrained models for demos from [Download](#Download) and place them to the `pretrained` folder

## Quick Demo
We provide a demo checkpoint and a environment map in the `demo` folder. You can simply run ``sh demo.sh`` to creating a relighting video demo in 3 minutes.


## Download

<p id="Download"></p>

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> Training Bash </th>
    <th> :link: Source </th>
    <th> :link: Checkpoint </th>
    <th> :link: Result </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Glossy Synthetic</td>
    <th> <a href="scripts/train_scripts/train_glossy.sh">train_glossy.sh</a> </th>
    <th> <a href="https://connecthkuhk-my.sharepoint.com/personal/yuanly_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuanly%5Fconnect%5Fhku%5Fhk%2FDocuments%2FNeRO&ga=1">Images</a></th>
    <th><a href="https://drive.google.com/file/d/1C_T0xZouK2TmK_sGHkumxPzYumWP9Lmx/view?usp=sharing">Google Driver</a></th>
    <th><a href="https://drive.google.com/file/d/1Q5GJJMwxTDqc-Z4pTkQNgq82luosuapC/view?usp=sharing">Google Driver</a></th>
  </tr>
  <tr>
    <td>Shiny Blender</td>
    <th> <a href="scripts/train_scripts/train_shiny.sh">train_shiny.sh</a> </th>
    <th> <a href="https://storage.googleapis.com/gresearch/refraw360/ref.zip">Images</a> / <a href="https://drive.google.com/file/d/1HGTD3uQUr8WrzRYZBagrg75_rQJmAK6S/view?usp=sharing">Point Cloud</a></th>
    <th><a href="https://drive.google.com/file/d/11nMvuUnigmUkes8mePE9tG1-c5aH25w_/view?usp=sharing">Google Driver</a></th>
    <th><a href="https://drive.google.com/file/d/1MhySgphbCNsxrwOai_hG2HaybP8fp0yc/view?usp=sharing">Google Driver</a></th>
  </tr>
  <tr>
    <td>TensoIR Synthetic</td>
    <th> <a href="scripts/train_scripts/train_tir.sh">train_tir.sh</a> </th>
    <th> <a href="https://zenodo.org/records/7880113#.ZE68FHZBz18">Images</a> / <a href="https://drive.google.com/file/d/10WLc4zk2idf4xGb6nPL43OXTTHvAXSR3/view">Env. maps</a></th>
    <th><a href="https://drive.google.com/file/d/1XiX2Pj8I1MSfZvgHJRgNdDetAwoOTrDI/view?usp=sharing">Google Driver</a></th>
    <th><a href="https://drive.google.com/file/d/1aKNI2FD6K7oeCNkPHlOLxSFFpnzmGFzF/view?usp=sharing">Google Driver</a></th>
  </tr>

</tbody>
</table>
**Update: Now you can also download our checkpoints from [HuggingFace](https://huggingface.co/lalala125/DiscreteSDF).**

## Training and Evaluation

Please refer to [develop.md](docs/development.md) to learn how to benchmark the DiscretizedSDF and how to train yourself DiscretizedSDF model from the scratch.


## Citation
   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @inproceedings{zhu_2025_dsdf,
      title={Gaussian Splatting with Discretized SDF for Relightable Assets},
      author={Zhu, Zuo-Liang and Yang, Jian and Wang, Beibei},
      booktitle={Proceedings of IEEE International Conference on Computer Vision (ICCV)},
      year={2025}
   }
   ```


## License
This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact `nkuzhuzl[AT]gmail.com`.

For commercial licensing, please contact `beibei.wang[AT]nju.edu.cn`。

## Acknowledgement

We thank [Zixiong Wang](https://www.bearprin.com/) for his suggestions during the project.

Here are some great resources we benefit from:
[GaussianShader](https://github.com/Asparagus15/GaussianShader), [2DGS](https://github.com/hbb1/2d-gaussian-splatting), [NeRO](https://github.com/liuyuan-pal/NeRO), [TensoSDF](https://github.com/Riga2/TensoSDF), [Ref-NeuS](https://github.com/EnVision-Research/Ref-NeuS)

**If you develop/use DiscretizedSDF in your projects, welcome to let us know. We will list your projects in this repository.**
