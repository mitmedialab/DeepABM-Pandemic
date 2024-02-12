DOSAGE_VALUE=08

nohup python -u main2.py --params Data/generated_params-expvac-2_scale1.yaml --seed 1234 --num_runs 10 --results_file_postfix NoDenYesQNoVacEFFScale1_wpoc_${DOSAGE_VALUE} > logs/NoDenYesQNoVacS1_${DOSAGE_VALUE}_wpoc.txt &
nohup python -u main2.py --params Data/generated_params-expvac-2_scale2.yaml --seed 1234 --num_runs 10 --results_file_postfix NoDenYesQNoVacEFFScale2_wpoc_${DOSAGE_VALUE} > logs/NoDenYesQNoVacS2_${DOSAGE_VALUE}_wpoc.txt &
nohup python -u main2.py --params Data/generated_params-expvac-2_scale4.yaml --seed 1234 --num_runs 10 --results_file_postfix NoDenYesQNoVacEFFScale4_wpoc_${DOSAGE_VALUE} > logs/NoDenYesQNoVacS4_${DOSAGE_VALUE}_wpoc.txt &
nohup python -u main2.py --params Data/generated_params-expvac-2_scale8.yaml --seed 1234 --num_runs 10 --results_file_postfix NoDenYesQNoVacEFFScale8_wpoc_${DOSAGE_VALUE} > logs/NoDenYesQNoVacS8_${DOSAGE_VALUE}_wpoc.txt &

