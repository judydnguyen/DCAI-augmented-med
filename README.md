# DCAI-augmented-med
- To run the ACGAN: ```python train_acgan.py```
- To run the DCGAN: ```python train_dcgan.py```
- All models and utility functions in `src/`

- To build datasets after training GAN model: ```python build_dataset.py```
- `quantitative_eval.sh` is for measure the quality of images generated.
- Original dataset `custom_covid_dataset/` is downloaded from original repo.


# Pre-trained GANs for Skin-lession Dataset
- Download ckpts for skin-lession trained with 400 epochs here: https://vanderbilt.box.com/s/g4zsdxl3tfxzuwdudnjd1ewk4dh9huvv
- Set the proper paths to load these checkpoint on the `train_acgan_skin_baseline.py`
- Generate new samples from this generative model and conduct the corresponding evaluation