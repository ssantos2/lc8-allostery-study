#!/bin/bash 
#SBATCH --job-name="trial-run-1-14-22"
#SBATCH --partition=small 
#SBATCH --account=santossh
#SBATCH --nodes=1
#SBATCH --mail-user=santossh@ohsu.edu
#SBATCH --workdir="/home/groups/ZuckermanLab/santossh/lc8-allostery-study/gromacs/trial-run-1-14-22"
#SBATCH --output=trial-results-1-14-22
#SBATCH --error=trial-error-1-14-22

. //home/groups/ZuckermanLab/santossh/lc8-allostery-study/gromacs/trial-run-1-14-22
module load gromacs/gromacs-2019_avx256
INPUT= 3DVT_clean.pdb
OUTPUT= 3DVT_processed.gro

/bin/echo "Executing gromacs script"
gmx pdb2gmx -f $INPUT -o $OUTPUT -water spce