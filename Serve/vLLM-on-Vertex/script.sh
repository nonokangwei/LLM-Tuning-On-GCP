python3 /root/scripts/launcher.py --model_gcs_uri=gs://llama2ft-project-kangwe-poc-unique/llama2-7b-hf --tensor-parallel-size=2 --swap-space=16 --port=7080 --host=0.0.0.0

python -m vllm.entrypoints.api_server --model=gs://vertex-model-garden-public-us-central1/llama2/llama2-7b-hf --tensor-parallel-size=2 --swap-space=16

python -m vllm.entrypoints.api_server --tensor-parallel-size=2 --swap-space=16 --model=/gcs-mount/llama2-7b-hf 