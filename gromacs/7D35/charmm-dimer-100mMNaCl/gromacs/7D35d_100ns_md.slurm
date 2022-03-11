#!/bin/bash
#SBATCH --job-name=7D35d_100ns_md
#SBATCH --output=7D35d_100ns_md.errout
#SBATCH --partition=gpu
#SBATCH --nodes 1
#SBATCH --tasks 2
#SBATCH --cpus-per-task 6
#SBATCH --gres gpu:1
#SBATCH --time=48:0:0

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

srun --mpi=pmi2 gmx_mpi grompp -f 7D35d_100ns_md.mdp -c 7D35d_100ns_em.gro -r 7D35d_100ns_em.gro -t 7D35d_100ns_eq.cpt -p topol.top -o 7D35d_100ns_md.tpr
srun --mpi=pmi2 -N 1 -n 1 -c 6  gmx_mpi mdrun -deffnm 7D35d_100ns_md -v -ntomp 1  -cpt 60  -maxh 24
