#!/bin/bash

# This code might cause zombies. If you run it, you might have to kill the processes manually.

MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct


# specify your directory
DIR_NAME=lcb_apr24_monitor
# DIR_NAME=lcb_apr9_single
export VLLM_URL="http://t-100.cs.tau.ac.il:8100/v1/"

ROLES="['PythonAssistant', 'AlgorithmDeveloper', 'ComputerScientist', 'Programmer']"
# ROLES="['PythonAssistant']"
JUDGES="['Passer', 'Tester', 'Reflector', 'Ranker']"

mkdir ${DIR_NAME}_PythonAssistant_AlgorithmDeveloper_ComputerScientist_Programmer/
for part in {0..3}
do
    EXP_NAME="llmlp_lcb_${part}"
    # run python script in background
    export OUTPUT_NAME=${DIR_NAME}_PythonAssistant_AlgorithmDeveloper_ComputerScientist_Programmer/terminal_${EXP_NAME}.txt
    # python llmlp_listwise_liveCodeBench.py "$part" "$EXP_NAME" "$MODEL" "$DIR_NAME" "$ROLES" "$JUDGES"
    python llmlp_listwise_liveCodeBench.py "$part" "$EXP_NAME" "$MODEL" "$DIR_NAME" "$ROLES" "$JUDGES" > $OUTPUT_NAME &
done

wait
echo "All done"


# DIR_NAME=${DIR_NAME}_PythonAssistant_AlgorithmDeveloper_ComputerScientist_Programmer
# rm ${DIR_NAME}/llmlp_human_eval.jsonl
# for i in {0..0}
# do
#     for part in {0..3}
#     do
#         EXP_NAME="llmlp_human_eval_${part}"
#         cat ${DIR_NAME}/${EXP_NAME}_443.jsonl >> ${DIR_NAME}/llmlp_human_eval.jsonl
#     done
#     evaluate_functional_correctness ${DIR_NAME}/llmlp_human_eval.jsonl > ${DIR_NAME}/llmlp_human_eval.txt

# done
