# Decoding-Brain-Activity-into-Images
3.0 Setting Environments
Create and activate conda environment named vis_dec from this env.yaml

conda env create -f env.yaml
conda activate vis_dec
3.1 FRL Phase 1
Overview
In Phase 1, we pre-train an MAE with a contrastive loss to learn fMRI representations from unlabeled fMRI data from HCP. The masking which sets a certain portion of the input data to zero targets the spatial redundancy of fMRI data. The calculation of recovering the original data from the remaining after masking suppresses noises. Optimization of the contrastive loss discerns common patterns of brain activities over individual variances.

Preparing Data
In this phase, we use fMRI samples released by HCP as pretraining data. Due to size limitations and licensing constraints, please download from the official website (https://db.humanconnectome.org/data/projects/HCP_1200), put them in the ./data/HCP directory and preprocess the data with ./data/HCP/preprocess_hcp.py. Resulting data and directory looks like:

/data
┣ 📂 HCP
┃   ┣ 📂 npz
┃   ┃   ┣ 📂 dummy_sub_01
┃   ┃   ┃   ┗ HCP_visual_voxel.npz
┃   ┃   ┣ 📂 dummy_sub_02
┃   ┃   ┃   ┗ ...
Training Model
You can run

python -m torch.distributed.launch —nproc_per_node=1  code/phase1_pretrain_contrast.py \
--output_path . \  
--contrast_loss_weight 1 \
—-batch_size 250 \
—-do_self_contrast True \
—-do_cross_contrast True \
--self_contrast_loss_weight 1 \ 
--cross_contrast_loss_weight 0.5 \
—mask_ratio 0.75 \
—num_epoch 140 
to pretrain the model by youself. do_self_contrast and do_contrast_contrast control whether or not self_contrast and contrast_contrast loss are used. self_contrast_loss_weight and cross_contrast_loss_weight denote the weight of self-contrast and cross-contrast loss in the joint loss.

You can also download our pretrained ckpt from https://1drv.ms/u/s!AlmPyF18ti-A3XmuKMPEfVNdvmsT?e=3bZ0jj

3.2 FRL Phase 2
Overview
After pre-training in Phase 1, we tune the fMRI auto-encoder with an image auto-encoder. We expect the pixel-level guidance from the image auto-encoder to support the fMRI auto-encoder in disentangling and attending to brain signals related to vision processing.

Preparing Data
We use the Generic Object Decoding (GOD) and BOLD5000 dataset in this phase. GOD is a specialized resource developed for fMRI-based decoding. It aggregates fMRI data gathered through the presentation of images from 200 representative object categories, originating from the 2011 fall release of ImageNet. The training session incorporated 1,200 images (8 per category from 150 distinct object categories). The test session included 50 images (one from each of the 50 object categories). The categories in the test session were unique from those in the training session and were introduced in a randomized sequence across runs. On five subjects the fMRI scanning was conducted. BOLD5000 is a result of an extensive slow event-related human brain fMRI study. It comprises 5,254 images, with 4,916 of them being unique. The images in BOLD5000 were selected from three popular computer vision datasets: ImageNet, COCO, and Scenes.

We provided processed versions of these datasets which can be downloaded from https://1drv.ms/u/s!AlmPyF18ti-A3Xec-3-PdsaO230u?e=ivcd7L Please download and uncompress it into the ./data. Resulting directory looks like:


┣ 📂 Kamitani
┃   ┣ 📂 npz
┃   ┃   ┗ 📜 sbj_1.npz
┃   ┃   ┗ 📜 sbj_2.npz
┃   ┃   ┗ 📜 sbj_3.npz
┃   ┃   ┗ 📜 sbj_4.npz
┃   ┃   ┗ 📜 sbj_5.npz
┃   ┃   ┗ 📜 images_256.npz
┃   ┃   ┗ 📜 imagenet_class_index.json
┃   ┃   ┗ 📜 imagenet_training_label.csv
┃   ┃   ┗ 📜 imagenet_testing_label.csv

┣ 📂 BOLD5000
┃   ┣ 📂 BOLD5000_GLMsingle_ROI_betas
┃   ┃   ┣ 📂 py
┃   ┃   ┃   ┗ CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_LHEarlyVis.npy
┃   ┃   ┃   ┗ ...
┃   ┃   ┃   ┗ CSIx_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_xx.npy
┃   ┣ 📂 BOLD5000_Stimuli
┃   ┃   ┣ 📂 Image_Labels
┃   ┃   ┣ 📂 Scene_Stimuli
┃   ┃   ┣ 📂 Stimuli_Presentation_Lists

Training Model
You can run the following commands to get the fMRI encoder that we use to produce the reported reconstruction performance on GOD subject 3 in the paper.

python -m torch.distributed.launch --nproc_per_node=4 code/phase2_finetune_cross.py \
--dataset GOD \
--pretrain_mbm_path your_pretrained_ckpt_from_phase1 \
--batch_size 4 \
--num_epoch 60 \
--fmri_decoder_layers 6 \
--img_decoder_layers 6 \
--fmri_recon_weight 0.25 \ 
--img_recon_weight 1.5 \
--output_path your_output_path \ 
--img_mask_ratio 0.5 \
--mask_ratio 0.75 
You can also download our trained ckpt from https://1drv.ms/u/s!AlmPyF18ti-A3XjJEkOfBELTl71W?e=wlihVF

3.3 Tuning LDM
Overview
Tuning Model
You can run the following commands to produce the reported reconstruction performance on GOD subject 3 in the paper.

python code/ldm_finetune.py --pretrain_mbm_path your_phase2_ckpt_path \
--num_epoch 700 \
--batch_size 8 \
--is_cross_mae \
--dataset GOD \
--kam_subs sbj_3 \
--target_sub_train_proportion 1. 
--lr 5.3e-5
