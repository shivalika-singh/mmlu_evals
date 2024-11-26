MODEL_CKPT=$1
EVAL_DIR=$2
USE_SINGLE_PROCESS=$3

USE_SINGLE_PROCESSES=${USE_SINGLE_PROCESS} ## This uses data parallel, update this as "true" for large models that do not fit a single GPU 

# if [ "${USE_SINGLE_PROCESSES}" = "true" ]; then
if [ "${MODEL_CKPT}" != "google/gemma-2-9b-it" ]; then
    
    for i in en ar es fr hi ru de id it ja ko pt zh yo nl ro uk vi tr pl fa cs he el ms fil te si ne ky sv lt sr mg so ha am sn ig ny bn sw;do 
        lm_eval \
            --model hf \
            --model_args pretrained="${MODEL_CKPT}",add_bos_token=True,dtype="float16",parallelize=True \
            --tasks mmlu_CS_${i} \
            --batch_size 2 \
            --num_fewshot 5 \
            --log_samples \
            --output_path ${EVAL_DIR}/mmlu_CS/${i};                                                                       
    done
else
    for i in en ar es fr hi ru de id it ja ko pt zh yo nl ro uk vi tr pl fa cs he el ms fil te si ne ky sv lt sr mg so ha am sn ig ny bn sw;do 
        accelerate launch -m lm_eval \
            --model hf \
            --model_args pretrained="${MODEL_CKPT}",add_bos_token=True,dtype="float16" \
            --tasks mmlu_CS_${i} \
            --batch_size 2 \
            --num_fewshot 5 \
            --log_samples \
            --output_path ${EVAL_DIR}/mmlu_CS/${i};                                                                       
    done
fi