<div align="center">

# OST

## Out-of-Sight Trajectory

### TPAMI Submission (Extended Version)

by [Haichao Zhang](https://Hai-chao-Zhang.github.io/), [Yi Xu](https://sites.google.com/view/homepage-of-yi-xu/), [Yun Fu](http://www1.ece.neu.edu/~yunfu/).  
</div>

---

**Pytorch implementation of the TPAMI extended version of ["OOSTraj: Out-of-Sight Trajectory Prediction With Vision-Positioning Denoising"](https://arxiv.org/abs/2404.02227).**

<div align="center">

<a href="">
<img width="800" alt="image" src="https://www.zhanghaichao.xyz/Out-of-SightTrajPred/assets/head.png">
</a>
</div>

---

Out-of-sight trajectory prediction is critical for real-world applications in autonomous driving, robotics, and surveillance, where agents may become occluded or move beyond the field of view. While previous methods rely on clean visual data, they fail to address challenges from noisy or partial sensor observations.

Our **TPAMI** extension builds upon the CVPR 2024 conference version and presents several key improvements:
- **Generalized prediction for diverse agent types**, including both pedestrians and vehicles.
- A **refined Vision-Positioning Denoising (VPD) framework** with improved feature fusion and uncertainty-aware projection.
- **Expanded benchmarks** on occluded trajectory prediction with comparisons to Kalman/Particle filters and deep learning models.
- **In-depth ablations** and a newly added discussion on limitations and future directions (e.g., dynamic cameras, 3D scene reconstruction).

By robustly denoising trajectories from noisy sensor data and mapping them into coherent visual space, our method significantly boosts performance under out-of-view scenarios. This work sets a new standard for out-of-sight trajectory reasoning.

---

<div align="center">
<strong>Vision-Positioning Denoising Model in OOSTraj</strong>

<a href="">
<img width="800" alt="image" src="https://www.zhanghaichao.xyz/Out-of-SightTrajPred/assets/arch.png">
</a>
</div>

---

## 🔧 Datasets

### Processed Vi-Fi and JRDB Datasets  
📂 [Google Drive (Preprocessed .pkl Files)](https://drive.google.com/drive/folders/1W6ze1z8X54kK9BOgYbXYQj_AScf79Z-q?usp=sharing)

Place the downloaded files into:
