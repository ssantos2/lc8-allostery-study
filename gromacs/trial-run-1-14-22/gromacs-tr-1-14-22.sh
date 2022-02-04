#!/bin/bash 
#
#SBATCH --job-name=trial-run-1-14-22
#SBATCH --account=zuckermanlab
#SBATCH --nodes=1
#SBATCH --mail-user=santossh@ohsu.edu
#SBATCH --workdir="/home/groups/ZuckermanLab/santossh/lc8-allostery-study/gromacs/trial-run-1-14-22"
#SBATCH --output=trial-results-1-14-22
#SBATCH --error=trial-error-1-14-22
#SBATCH --partition=exacloud

INPUT= 3DV_clean.pdb
OUTPUT=3DVT_processed.gro

module use /home/groups/ZuckermanLab/russojd/modulefiles
module load gromacs/gromacs-2019_avx256
srun --mpi=pmi2 gmx pdb2gmx -f $INPUT -o $OUTPUT -water spce
/bin/echo "Executing gromacs script"
