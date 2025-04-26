#!/bin/bash
#SBATCH -J PuffinD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=128000
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --account=redacted
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH -o /scratch/redacted/redacted/DETAILS/slurm_outputs/%A_%a.out
#SBATCH --error=/scratch/redacted/redacted/DETAILS/slurm_outputs/%A_%a.err

# This script runs steps corresponding to BioQueue job 15123
echo $(hostname);

dest_dir=/scratch/redacted/redacted/workdir/"$SLURM_JOB_ID";
mkdir -p "${dest_dir}";
# running dir
parent_dir=/scratch/redacted/redacted/workdir/"$SLURM_JOB_ID";
mkdir -p "${parent_dir}";
chmod -R 700 "${parent_dir}";
cd "$parent_dir";

test_array=(
  [0]="chr1 chr2"
  [1]="chr3 chr4"
  [2]="chr5 chr6"
  [3]="chr7 chr8"
  [4]="chr9 chr10"
  [5]="chr11 chr12"
  [6]="chr13 chr14"
  [7]="chr15 chr16"
  [8]="chr17 chr18"
  [9]="chr19 chr20"
  [10]="chr21 chr22"
  [11]="chrX"
)
val_array=(
  [0]="chr3"
  [1]="chr5"
  [2]="chr7"
  [3]="chr9"
  [4]="chr11"
  [5]="chr13"
  [6]="chr8"
  [7]="chr6"
  [8]="chr4"
  [9]="chr2"
  [10]="chrX"
  [11]="chr1"
)

# Check if SLURM_ARRAY_TASK_ID is within the range [0, 11]
if [[ $SLURM_ARRAY_TASK_ID -ge 0 && $SLURM_ARRAY_TASK_ID -le 11 ]]; then
    val_chroms=${val_array[$SLURM_ARRAY_TASK_ID]}
    test_chroms=${test_array[$SLURM_ARRAY_TASK_ID]}
else
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of range [0, 11]"
    exit 1
fi

source /u/redacted/miniconda3/etc/profile.d/conda.sh;
conda activate /scratch/redacted/redacted/DETAILS/envs/puffin/;

git clone git@redacted_url/pipeline-helpers.git;

mkdir -p "$parent_dir/models";

python "${parent_dir}"/pipeline-helpers/supervised_signal_prediction/PuffinD/train_puffin_D.py \
  -p /scratch/redacted/redacted/DETAILS/data/14014v0/A673.ds.pl.bw \
     /scratch/redacted/redacted/DETAILS/data/14014v0/Caco2.ds.pl.bw \
     /scratch/redacted/redacted/DETAILS/data/14014v0/HUVEC.ds.pl.bw \
     /scratch/redacted/redacted/DETAILS/data/14014v0/LNCaP.ds.pl.bw \
     /scratch/redacted/redacted/DETAILS/data/14014v0/MCF10A.ds.pl.bw \
  -m /scratch/redacted/redacted/DETAILS/data/14014v0/A673.ds.mn.bw \
     /scratch/redacted/redacted/DETAILS/data/14014v0/Caco2.ds.mn.bw \
     /scratch/redacted/redacted/DETAILS/data/14014v0/HUVEC.ds.mn.bw \
     /scratch/redacted/redacted/DETAILS/data/14014v0/LNCaP.ds.mn.bw \
     /scratch/redacted/redacted/DETAILS/data/14014v0/MCF10A.ds.mn.bw \
  -l A673 Caco2 HUVEC LNCaP MCF10A \
  -f /scratch/redacted/redacted/DETAILS/refs/hg38.22X.fasta --min-delta 0.01 \
  -t $test_chroms -v $val_chroms |\
  tee "train.$SLURM_ARRAY_TASK_ID.log"

job_status=$?

cd "$parent_dir";

mv models models_"$SLURM_ARRAY_TASK_ID";

rm -rf pipeline-helpers;

# synchronize results back to main machine
rsync -rlptD --compress --partial . redacted_ip:/local/storage/redacted/BioQueue/workspace/27/15123v0/;
rsync_status=$?;
echo "Results synchronization status:" $rsync_status

rm -rf "$parent_dir";

# copy slurm logs
cat /scratch/redacted/redacted/DETAILS/slurm_outputs/"${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" >> /scratch/redacted/redacted/DETAILS/slurm_outputs/15123.out
cat /scratch/redacted/redacted/DETAILS/slurm_outputs/"${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >> /scratch/redacted/redacted/DETAILS/slurm_outputs/15123.err

[ $job_status -eq 0 ] && echo "Job finished" || exit 1
