#!/bin/bash
#SBATCH --job-name=7D35a_eq_gromacs_test
#SBATCH --output=7D35a_eq_gromacs_test.errout
#SBATCH --partition=gpu
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 1
#SBATCH --gres gpu:1
#SBATCH --exclude exanode-8-18
#SBATCH --time=1:00:0

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

srun --mpi=pmi2 gmx_mpi grompp -f 7D35a_eq_gromacs_test.mdp -c 7D35a_em_gromacs_test.gro -r 7D35a_em_gromacs_test.gro -p topol.top -o 7D35a_eq_gromacs_test.tpr -n index.ndx 
mpirun -np 2 gmx_mpi mdrun -deffnm 7D35a_eq_gromacs_test -ntomp 2 -npme 1 -ntomp_pme 1 -v 
echo "Finished 7D35a_eq_gromacs_test"