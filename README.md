# SLiT_recon

This repository demonstrates the whole reconstruction pipeline for scanning light-field tomography (SLiT) with iterative multi-view phase-space deconvolution (IMPD) algorithm guided by Multi-Conjugate Digital Adaptive Optics (MDAO)

We used the experimentally captured data of the vessel of the transgenic zebrafish (fli1a: mCherry) embryos as a demo to show the 3D reconstruction process with the high-resolution 3D volume. 

Intravital microscopy, Adaptive optics, 3D reconstruction

authors: Yifan Chen, Bo Xiong, Jiamin Wu

corresponding_contributor: Qionghai Dai

email: qhdai@tsinghua.edu.cn

# dependencies
- Cuda version: 11.1
- pip install -r requirements.txt
- Conda-Pack environment: https://doi.org/10.5281/zenodo.13370218
- data and PSF: https://doi.org/10.5281/zenodo.13367201
- results: https://doi.org/10.5281/zenodo.13384506

# run the project
Run Main_PipeLine_V31x_1.py first. Then run Main_PipeLine_V34x_1.py or Main_PipeLine_V34x_2.py to get high quality results

- Code\210629_fishR_ballG_1750_6_ReOrder_ReAlign_Cut359_Cx194\Main_PipeLine_V31x_1.py

Sequential intensity reconstruction and phase reconstruction of zebrafish and fluorescent beads under 488 excitation light, 359^3 reconstruction volume

- Code\210629_fishR_ballG_1750_6_ReOrder_ReAlign_Cut359_Cx194\Main_PipeLine_V34x_1.py

High-precision global optimized intensity reconstruction of zebrafish and fluorescent beads under 488 excitation light, 721^3 reconstruction volume

- Code\210629_fishR_ballG_1750_6_ReOrder_ReAlign_Cut359_Cx194\Main_PipeLine_V34x_2.py

High-precision global optimized intensity reconstruction of zebrafish and fluorescent beads under 561 excitation light, 721^3 reconstruction volume


# results
- Intensity reconstruction result:
.\Intensity\Phantom_AVG_N.tif (Maximum N)

- Root of rays reconstruction result:
.\WaveFront\RayInitZernikeList_N (Maximum N)

.\Code\210629_fishR_ballG_1750_6_ReOrder_ReAlign_Cut359_Cx194\Recon

```bash
PipeLineOptSync  	             # Main_PipeLine Reconstruction results
   ├── Intensity                     # IMPD Intensity matrix results
   │   ├── Phantom_N.tif             # The Nth viewpoint intensity matrix iterative result
   │   └── Phantom_AVG_M.tif         # The mean of the results of all viewpoint intensity matrices in the Mth round
   └── WaveFront                     # MDAO Phases results
        ├── dRayInit_M_N.tif         # The root matrix of the ray of the Mth round and the Nth viewpoint
        ├── RayInitZernikeList_16    # the root matrix of the rays of all viewpoints of the Mth round
        └── rest of the file         # tensorboard information
```
 rest of the files are for the testing.

- _bp is single viewpoint reconstruction results.
- _PreTrain is MDAO off results.
- _Yprojection is the Y axis projection results.
