#!/bin/bash

#SBATCH --output=/dev/null

#SBATCH --job-name=boptest_res

#SBATCH --time=5-00:00:00

#SBATCH --ntasks-per-node=2

#SBATCH --partition=skylake,skylake_mpi

#SBATCH --mem-per-cpu=2G

#SBATCH --array=1-60


module load Python/3.9.6-GCCcore-11.2.0

module load BOPTEST/0.1.0

module load matplotlib/3.4.3-foss-2021b

source $VSC_DATA/boptest_env/bin/activate

PORT=$(( SLURM_JOB_ID + 6100 )) 

TESTCASE=bestest_hydronic_heat_pump

BOPTEST_HOME=$EBROOTBOPTEST

IMAGE=/apps/brussel/singularity/boptest/boptest-0.3.0.sif

APP_PATH=/home/developer

mkdir -p $VSC_SCRATCH/boptest/$SLURM_JOB_ID/$TESTCASE

cd $VSC_SCRATCH/boptest/$SLURM_JOB_ID/$TESTCASE

mkdir models

cp $BOPTEST_HOME/testcases/${TESTCASE}/models/wrapped.fmu models/ && \

cp -r $BOPTEST_HOME/testcases/${TESTCASE}/doc/ ./ && \

cp $BOPTEST_HOME/restapi.py ./ && \

cp $BOPTEST_HOME/testcase.py ./ && \

cp $BOPTEST_HOME/version.txt ./ && \

cp -r $BOPTEST_HOME/data ./ && \

cp -r $BOPTEST_HOME/forecast ./ && \

cp -r $BOPTEST_HOME/kpis ./ && \

cp -r $BOPTEST_HOME/examples/ ./



export PYTHONPATH=$PWD:$PYTHONPATH



singularity -s exec --env 'FLASK_APP=restapi' -B $PWD:$APP_PATH $IMAGE flask run --port=$PORT &



#wait a bit to ensure that the container is up and running

sleep 30


cd /user/brussel/102/vsc10250/mammuet/energy-management-learner

# quick fix to allow using a different port than the default 5000

python boptest/paralelle_exec.py $PORT
