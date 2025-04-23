<div align="center">

<h1> [AAAI25 Oral] DiffCalib: Reformulating Monocular Camera Calibration as Diffusion-Based Dense Incident Map Generation</h1>


#### 🔥 Fine-tune diffusion models for camera intrinsic estimation and depth estimation simultaneously! ✈️

</div>

<div align="center">
<img width="800" alt="image" src="figs/pipeline.jpg">
</div>


##  📢 News
- 2025.4.23: DiffCalib has been published at [Proceedings of the AAAI Conference on Artificial Intelligence](https://ojs.aaai.org/index.php/AAAI/article/view/32355).
- 2025.2.15:  Congratulations! DiffCalib has been accepted as an oral presentation at AAAI 2025. AAAI 2025 received a total of 12,957 valid submissions, with 3,032 papers accepted, resulting in an acceptance rate of 23.4%. Among these, Oral papers accounted for 4.6%.
- 2024.7.25:  Release inference code and checkpoint weight of Diffcalib in the repo.
- 2024.7.25: Release [arXiv paper](https://arxiv.org/abs/2405.15619), with supplementary material.


##  🖥️ Dependencies

```bash
conda create -n diffcalib python=3.10
conda activate diffcalib
pip install -r requirements.txt
pip install -e .
```

## Data

Download the intrinsic data from [MonoCalib](https://github.com/ShngJZ/WildCamera/blob/main/asset/download_wildcamera_dataset.sh)

## 🚀 Evaluation
First, Download the [stable-diffusion-2-1](download_stable-diffusion-2-1.py) and transform it to the 8 inchannels from [modify](load_ckpt_and_modify_8in.py) which will be put in --checkpoint, put the checkpoints under ```./checkpoint/```

Then, Download the pre-trained models ```diffcalib-best.zip``` from [BaiduNetDisk](https://pan.baidu.com/s/1jy2jXHoe8IUtOUH_5mfw3g?pwd=xn5p)(Extract code：xn5p). Please unzip the package and put the checkpoints under ```./checkpoint/``` which will be put in --unet_ckpt_path.

finally, you can run the bash to evaluate our model in the benchmark.
```bash
sh scripts/run_incidence_mutidata.sh
```

## 🚀 visualization and 3D reconstruction
For depth and incident map visualization
, download shift model from [res101](https://pan.baidu.com/s/1o2oVMiLRu770Fdpa65Pdbw?pwd=g3yi) (Extract code: g3yi), ```diffcalib-pcd.zip``` from [BaiduNetDisk](https://pan.baidu.com/s/1F0uUlYDfz0ysV_KdP8g3mg?pwd=20z6)(Extract code:20z6) and install torchsparse packages as follows
```bash
sudo apt-get install libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.2.0
```
Then we can reconstruct 3D shape from a single image.
```bash
bash scripts/run_incidence_depth_pcd.sh
```

## 📖 Recommanded Works

- Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. [arXiv](https://github.com/prs-eth/marigold), [GitHub](https://github.com/prs-eth/marigold).

## 🏅 Results in Paper

### point cloud

<div align="center">
<img width="800" alt="image" src="figs/point_cloud.png">
</div>

### depth and incident map visualization

<div align="center">
<img width="800" alt="image" src="figs/visualization.png">
</div>

## 🎫 License

For non-commercial use, this code is released under the [LICENSE](LICENSE).

## 🎓 Citation
```
@article{
	He_Xu_Zhang_Chen_Cui_Guo_2025, 
	title={DiffCalib: Reformulating Monocular Camera Calibration as Diffusion-Based Dense Incident Map Generation}, 
	volume={39},
	DOI={10.1609/aaai.v39i3.32355},  
	number={3},
	journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
	author={He, Xiankang and Xu, Guangkai and Zhang, Bo and Chen, Hao and Cui, Ying and Guo, Dongyan}, 
	year={2025}, 
	month={Apr.}, 
	pages={3428-3436}
}
```
