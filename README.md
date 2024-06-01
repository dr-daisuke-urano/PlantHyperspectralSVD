# SVD-assisted visualization and assessment of spatial hyperspectral patterns in plant leaves

This is the official implementation for Shalini Krishnamoorthi et al. (2024) [https://www.cell.com/cell-reports/home].

## Project Overview

Hyperspectral camera captures the reflectance of light with greater spectral resolutions (sub-nanometer to a few nanometers), and saves information into the data cube of x, y and λ (two dimensional images with many wavelength channels). This project describes the Python codes to visualize leaf color patterns using pseudo-color spaces built with singular vector decomposition (SVD) of the normalized hyperspectral images. The procedure consists of four steps. The first step normalizes pixel intensity at all wavelength channels using the mean reflectance near 900 nm (875 to 925 nm bands were used in this invention disclosure). The second step conducts SVD transformation and saves the first five SVD spaces. The third step generates the pseudo-colored images with density plot along the five SVD color spaces. The user of hyperspectral camera then selects and saves the SVD color space(s) that effectively represents leaf color patterns. The fourth step applies the saved SVD space(s) to hyperspectral images of other leaves. The pseudo-colored images can also be used together with spot or area detection algorithms to diagnose plant nutrient stresses. 

## Dependencies

- Python 3.6+
- PyTorch 1.4+
- NumPy
- Matplotlib
- Jax
- Optax
- GPy
- Scipy
- scikit-learn
- [TuRBO](https://github.com/uber-research/TuRBO)
- [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization)

## Usage

### Black-Box Adversarial Attack
```bash
PYTHONPATH=$PWD python run_exp.py --syn_func MNIST_Attack --dim 784 --lb -0.3 --ub 0.3 --bo_iters 2000 --gd_iters 0 --alg zord --trials 1 --save "eps(0.3)-" --gd_lr 0.5 --seed 0
PYTHONPATH=$PWD python run_exp.py --syn_func CIFAR10_Attack --dim 1024 --lb -0.2 --ub 0.2 --bo_iters 0 --gd_iters 1000 --alg zord --trials 1 --save "eps(0.2)-" --gd_lr 0.5 --seed 5
```
### Non-Differentiable Metric Optimization
```bash
# Need to choose the corresponding metric function in the functions/metrics.py before runing the following commands

PYTHONPATH=$PWD python run_exp.py --syn_func MetricOpt --dim 2187 --lb -0.2 --ub 0.2 --alg zord --gd_iters 800 --bo_iters 0 --trials 5 --gd_lr 0.01 --save results/CovType/f1score-
PYTHONPATH=$PWD python run_exp.py --syn_func MetricOpt --dim 2187 --lb -0.2 --ub 0.2 --alg zord --gd_iters 800 --bo_iters 0 --trials 5 --gd_lr 0.01 --save results/CovType/jaccard-
PYTHONPATH=$PWD python run_exp.py --syn_func MetricOpt --dim 2187 --lb -0.2 --ub 0.2 --alg zord --gd_iters 800 --bo_iters 0 --trials 5 --gd_lr 0.2 --save results/CovType/precision-
PYTHONPATH=$PWD python run_exp.py --syn_func MetricOpt --dim 2187 --lb -0.2 --ub 0.2 --alg zord --gd_iters 800 --bo_iters 0 --trials 5 --gd_lr 0.2 --save results/CovType/recall-
```

## Citation

```bibtex
@inproceedings{
    shu2023zerothorder,
    title={Zeroth-Order Optimization with Trajectory-Informed Derivative Estimation},
    author={Yao Shu and Zhongxiang Dai and Weicong Sng and Arun Verma and Patrick Jaillet and Bryan Kian Hsiang Low},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=n1bLgxHW6jW}
}
```
