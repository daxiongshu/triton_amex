# This repo contains the notebooks and scripts for the blog **High-speed Serving Multi-Model Credit Default Prediction with NVIDIA Triton**

Install dependencies:
- `conda create -n rapids-23.04 -c rapidsai-nightly -c conda-forge -c nvidia rapids=23.04 python=3.10 cudatoolkit=11.8`
- `conda activate rapids-23.04`
- `pip install -r requirements.txt`

Please run the notebooks in the following order:
- 0_download_data.ipynb
- 1_auto_regressive_rnn.ipynb
- 2_xgboost_end2end.ipynb
- 3_triton_inference.ipynb