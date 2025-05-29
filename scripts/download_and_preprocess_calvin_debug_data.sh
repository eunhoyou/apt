export PROJECT_ROOT='/workspace/ActionGPT/'
export OUTPUT_ROOT='/data/'

echo "***********************************************************"
if [ ! -d ${OUTPUT_ROOT} ]; then
   mkdir -p ${OUTPUT_ROOT}
fi

cd ${OUTPUT_ROOT}
dataset_path="${OUTPUT_ROOT}/debug"
if [ -d ${dataset_path} ]; then
   	echo "${dataset_path} exists."
else
   echo "Downloading CALVIN debug dataset..."
   wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
   unzip calvin_debug_dataset.zip
   echo "saved folder: ${OUTPUT_ROOT}/calvin_debug_dataset"
fi

echo "***********************************************************"
lmdb_path="${OUTPUT_ROOT}/lmdb_datasets/calvin_debug/"
cd ${PROJECT_ROOT}/data_preprocessing
if [ -d ${lmdb_path} ]; then
   echo "${lmdb_path} exists."
else
   echo "Outputting CALVIN debug dataset to LMDB format at ${lmdb_path}..."
   python3 -u calvin_to_lmdb.py \
   --input_dir ${OUTPUT_ROOT}/calvin_debug_dataset \
   --output_dir ${lmdb_path}
fi

<<COMMENT
cd ${PROJECT_ROOT}/scripts/
nohup bash download_and_preprocess_calvin_debug_data.sh > download_and_preprocess_calvin_debug_data.log 2>&1 &
tail -f download_and_preprocess_calvin_debug_data.log
COMMENT