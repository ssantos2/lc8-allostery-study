<<<<<<< HEAD
#!/bin/bash
#SBATCH --job-name=7D35_md_0_1
#SBATCH --output=7D35_md_0_1.errout
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

srun --mpi=pmi2 -N 1 -n 1 -c 6 gmx_mpi grompp -f step4.1_equilibration100ns.mdp -c step3_input.gro -r step3_input.gro -t 7D35_em100ns.cpt -p topol.top -o 7D35_equilib100ns.tpr -v 
echo "Finished creating 7D35_equilib100ns.tpr"

srun --mpi=pmi2 -N 1 -n 1 -c 6 gmx_mpi mdrun -deffnm 7D35_equilib100ns -v -ntomp 1  -cpt 60  -maxh 24
=======
#!/bin/bash
#SBATCH --job-name=7D35_equilib100ns
#SBATCH --output=7D35_equilib100ns.errout
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

srun --mpi=pmi2 -N 1 -n 1 -c 6 gmx_mpi grompp -f step4.1_equilibration100ns.mdp -c step3_input.gro -r step3_input.gro -t 7D35_em100ns.cpt -p topol.top -o 7D35_equilib100ns.tpr -v 
echo "Finished creating 7D35_equilib100ns.tpr"

srun --mpi=pmi2 -N 1 -n 1 -c 6 gmx_mpi mdrun -deffnm 7D35_equilib100ns -v -ntomp 1  -cpt 60  -maxh 24
>>>>>>> 03c26ea7e4336609ca587b382faae37f7d77b294
echo "Finished production of 7D35_equilib100ns equilibration"