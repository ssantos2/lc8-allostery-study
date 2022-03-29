#!/bin/bash 
#SBATCH --job-name=7D35a_100ns_em_energy16-17-18
#SBATCH --output=7D35a_100ns_em_energy16-17-18.errout
#SBATCH --partition=gpu
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --cpus-per-task 6
#SBATCH --gres gpu:1
#SBATCH --time=1:0:0

module use /home/exacloud/software/modules/
#module load gromacs/gromacs-2019
#module load gromacs/gromacs-2019_avx256

module load openmpi/3.1.5
module load mpi/mpich-3.0-x86_64
module load gromacs/2020.2+cuda

./run.sh

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

echo "16 17 18 \n" |srun --mpi=pmi2 -N 1 -n 1 -c 6 gmx_mpi energy -f 7D35a_100ns_em.edr -o 7D35a_100ns_em_energy16-17-18.xvg 

echo "Finished creating 7D35a_100ns_em_energy16-17-18.xvg"