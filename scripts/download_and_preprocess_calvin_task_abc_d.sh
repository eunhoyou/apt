export PROJECT_ROOT='/workspace/ActionGPT/'
export OUTPUT_ROOT='/data/'

echo "***********************************************************"
if [ ! -d ${OUTPUT_ROOT} ]; then
   mkdir -p ${OUTPUT_ROOT}
fi

cd ${OUTPUT_ROOT}
dataset_path="${OUTPUT_ROOT}/task_ABC_D"
# if [ -d ${dataset_path} ]; then
#    	echo "${dataset_path} exists."
# else
#    echo "Downloading CALVIN task_ABC_D ..."
#    # aria2c -x 16 -s 16 http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
#    unzip -o task_ABC_D.1.zip
#    echo "saved folder: ${OUTPUT_ROOT}/task_ABC_D"
# fi

echo "***********************************************************"
lmdb_path="${OUTPUT_ROOT}/lmdb_datasets/task_ABC_D/"
cd ${PROJECT_ROOT}/data_preprocessing
if [ -d ${lmdb_path} ]; then
   echo "${lmdb_path} exists."
else
   echo "Outputting CALVIN task_ABC_D lmdb dataset to ${lmdb_path} ..."
   python3 -u calvin_to_lmdb.py \
   --input_dir ${OUTPUT_ROOT}/task_ABC_D \
   --output_dir ${lmdb_path}
fi

<<COMMENT
cd ${PROJECT_ROOT}/scripts/
nohup bash download_and_preprocess_calvin_task_abc_d.sh > download_and_preprocess_calvin_task_abc_d.log 2>&1 &
tail -f download_and_preprocess_calvin_task_abc_d.log
COMMENT