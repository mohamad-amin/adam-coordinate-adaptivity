
# Adam Exploits ℓ∞-geometry of Loss Landscape via Coordinate-wise Adaptivity
This repository contains the code used to train GPT2 and evaluate different metrics on its hessian in the paper ["Adam Exploits ℓ∞-geometry of Loss Landscape
via Coordinate-wise Adaptivity"](https://arxiv.org/pdf/2410.08198). We use [nanoGPT](https://github.com/karpathy/nanoGPT)'s code as the base for training GPT2. We have used [JAX](https://github.com/jax-ml/jax) to implement the Hessian-related computations by transferring the learnt GPT2 models to a [Flax](https://github.com/google/flax) based GPT2 implementation, borrowed from [HuggingFace's transformers](https://github.com/huggingface/transformers) package. 

## Training
You can use the following command to train a GPT2 model on the OpenWebText dataset:

```
$ torchrun --standalone --nproc_per_node 8 run.py --config_path=configs/gpt2_train.json --save_dir=out_dir
```

To train a model with an orthogonally rotated loss, you can run the following command:

```
$ torchrun --standalone --nproc_per_node 8 run.py --config_path=configs/gpt2_rotated_train.json --save_dir=out_dir
```

## Evaluation

You can use the following command to estimate the 1-1 norm of the Hessian of a trained GPT2 model:
```
$ torchrun --standalone --nproc_per_node 8 run_jax.py --config_path=configs/gpt2_evaluate.json --load_dir=out_dir --save_dir=eval_dir
```
To evaluate the top eigenvalue, change `hessian.task` in `configs/gpt2_evaluate.yml` from `compute_11_norm` to `compute_eigvals`.

## A few notes
* Flash attention is disabled by default. This is to achieve better numerical precision in both training and evaluation.
* Computing Hessian-related metrics would require a significant amount of GPU RAM. For the configs available in this repo, an A100 GPU with 40GB of GPU RAM is sufficient. However, the more the merrier.

## Citation
If you find this code useful, please consider citing our paper:
```
@misc{xie2024adamexploitsellinftygeometryloss,
      title={Adam Exploits $\ell_\infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity}, 
      author={Shuo Xie and Mohamad Amin Mohamadi and Zhiyuan Li},
      year={2024},
      eprint={2410.08198},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.08198}, 
}
```
