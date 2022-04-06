#!/bin/bash
#SBATCH --job-name=7D35a_tune_pme
#SBATCH --output=7D35a_tune_pme.errout
#SBATCH --partition=gpu
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:1
#SBATCH --exclude exanode-8-18
#SBATCH --time=48:00:0

module use /home/exacloud/software/modules/

module load openmpi/3.1.6
module load gromacs/2020.2+cuda

./run.sh

echo "Starting at `date`"
echo "Running on hosts: $SLURM_JOB_NODELIST"
echo "Running on $SLURM_JOB_NUM_NODES nodes."
echo "Running $SLURM_STEP_NUM_TASKS tasks."
echo "Current working directory is `pwd`"
echo "JobID : $SLURM_JOB_ID"

srun --mpi=pmi2 gmx_mpi tune_pme -s 7D35a_md_gromacs_test1.tpr -so 7D35a_md_tune_test1.tpr -g 7D35a_md_tune_test1.log
srun gmx_mpi mdrun -deffnm 7D35a_md_tune_test1 -v 

echo "Finished 7D35a_md_gromacs_test"	
