export CUDA_VISIBLE_DEVICES=0

echo "Start running TTS pipeline"

python -m data_gen.tts.runs.preprocess --config egs/datasets/audio/mix/base_text2mel.yaml
echo "Preprocess done!"

N_PROC=16 python -m data_gen.tts.runs.train_mfa_align --config egs/datasets/audio/mix/base_text2mel.yaml
echo "Train MFA align done!"

python -m data_gen.tts.runs.binarize --config egs/datasets/audio/mix/base_text2mel.yaml
echo "Binarize done!"
