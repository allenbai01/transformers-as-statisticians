# Ridge
CUDA_VISIBLE_DEVICES=0 python train.py --conf conf/gpt/ridge_1.0.yaml &
CUDA_VISIBLE_DEVICES=1 python train.py --conf conf/gpt/ridge_0.25.yaml &
CUDA_VISIBLE_DEVICES=2 python train.py --conf conf/gpt/ridge_0.5.yaml &
CUDA_VISIBLE_DEVICES=3 python train.py --conf conf/gpt/ridge_0.1.yaml &
CUDA_VISIBLE_DEVICES=4 python train.py --conf conf/gpt/ridge_mtl.yaml &

# Linear regression and classification
CUDA_VISIBLE_DEVICES=5 python train.py --conf conf/gpt/linear_regression.yaml &
CUDA_VISIBLE_DEVICES=6 python train.py --conf conf/gpt/linear_classification.yaml &
CUDA_VISIBLE_DEVICES=7 python train.py --conf conf/gpt/linear_and_logistic.yaml