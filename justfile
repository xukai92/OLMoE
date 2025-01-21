set dotenv-load

pre-process:
    #!/usr/bin/env bash
    for split in train validation test; do
        dolma tokens \
            --documents /home/lab/moe/data/raw/minipile/${split}.jsonl \
            --destination /home/lab/moe/data/preprocessed/minipile/${split} \
            --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
            --max_size '2_147_483_648' \
            --seed 0 \
            --tokenizer.eos_token_id 50279 \
            --tokenizer.pad_token_id 1 \
            --processes 128
    done

pre-train:
    #!/usr/bin/env bash
    torchrun --nnodes 1:1 --node_rank 0 --nproc-per-node 8 --rdzv_id=12347 --rdzv_backend=c10d --rdzv_conf='read_timeout=420' --rdzv_endpoint=localhost:29400 /home/lab/moe/OLMo/scripts/train.py \
        /home/lab/moe/OLMoE/configs/OLMoE-small.yml \
        --run_name=olmoe-small-variational-dev \
        --save-overwrite \
        --fsdp.sharding_strategy=FULL_SHARD \
        --device_train_microbatch_size=4 \
        --canceled_check_interval=9999999

convert name-slash-step:
    #!/usr/bin/env bash
    python /home/lab/moe/OLMo-unshard/scripts/unshard.py /home/lab/moe/OLMoE/runs/{{name-slash-step}} /home/lab/moe/OLMoE/runs/{{name-slash-step}}-unsharded --model-only
    cp /home/lab/.cache/huggingface/hub/models--allenai--gpt-neox-olmo-dolma-v1_5/blobs/d06ef09037384992a31352810f7500d69c902195 /home/lab/moe/OLMoE/runs/{{name-slash-step}}-unsharded/tokenizer.json
    python /home/lab/moe/transformers/src/transformers/models/olmoe/convert_olmoe_weights_to_hf.py --input_dir /home/lab/moe/OLMoE/runs/{{name-slash-step}}-unsharded --tokenizer_json_path /home/lab/moe/OLMoE/runs/{{name-slash-step}}-unsharded/tokenizer.json --output_dir /home/lab/moe/artifacts/olmoe-small-dev
