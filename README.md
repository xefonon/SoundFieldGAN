Generative adversarial networks with physical sound field priors
================================================

This repository contains the code for the paper [Generative adversarial networks with physical sound field priors](https://pubs.aip.org/asa/jasa/article/154/2/1226/2908014) accepted for publication in The Journal of the Acoustical Society of America (2023).

Abstract
--------------------
This paper presents a deep learning-based approach for the spatio-temporal reconstruction of sound fields using Generative Adversarial Networks (GANs). The method utilises a plane wave basis and learns the underlying statistical distributions of pressure in rooms to accurately reconstruct sound fields from a limited number of measurements. The performance of the method is evaluated using two established datasets and compared to state-of-the-art methods. The results show that the model is able to achieve an improved reconstruction performance in terms of accuracy and energy retention, particularly in the high-frequency range and when extrapolating beyond the measurement region. Furthermore, the proposed method can handle a varying number of measurement positions and configurations without sacrificing performance. The results suggest that this approach provides a promising approach to sound field reconstruction using generative models that allow for a physically informed prior to acoustics problems

Usage
--------------------

To create a conda environment with all the required dependencies, execute the following command in your terminal:

```bash
conda env create -f environment.yml
```

An illustrative example of sound field inference can be found in the notebook `./notebooks/bandwidth_extension_example.ipynb`. To run this notebook, please start by executing the scripts `./data/Inference files/dl_gen_weights.py` and `./data/Inference files/dl_meshrir.py` to download the necessary files.



Citation
--------------------
If you use this work, please consider including the following citation in your research:

```
@article{karakonstantis2023generative,
  title={Generative Adversarial Networks with Physical Sound Field Priors},
  author={Karakonstantis, Xenofon and Fernandez-Grande, Efren},
  journal={The Journal of the Acoustical Society of America},
  volume={154},
  number={2},
  pages={1226--1238},
  year={2023},
  publisher={AIP Publishing}
}
```
