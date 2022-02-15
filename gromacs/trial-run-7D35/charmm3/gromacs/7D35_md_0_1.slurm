#!/bin/bash
#SBATCH --tasks-per-node 2
#SBATCH --cpus-per-task 2
#SBATCH --nodes 2
#SBATCH --time 10:00:00
#SBATCH --job-name 7D35_md_0_1
#SBATCH --partition gpu
#SBATCH --gres=gpu:1


export LD_LIBRARY_PATH=/home/exacloud/lustre1/ZuckermanLab/santossh/lc8-allostery-study/gromacs/trial-run-7D35/charmm3/gromacs:$LD_LIBRARY_PATH 

module use /home/exacloud/software/modules 
module load gromacs/2020.2+cuda

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

srun --mpi=pmi2 gmx grompp -f 7D35_md_0_1.mdp -c step3_input.gro -t 7D35_equilib.cpt -p topol.top -o 7D35_md_0_1.tpr
srun --mpi=pmi2 gmx mdrun -deffnm 7D35_md_0_1
echo "Program finished with exit code $? at: `date`"
