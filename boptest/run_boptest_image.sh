PORT=$1

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



singularity exec --env 'FLASK_APP=restapi' -B $PWD:$APP_PATH $IMAGE flask run --port=$PORT &



#wait a bit to ensure that the container is up and running

sleep 30


cd /user/brussel/102/vsc10250/mammuet/energy-management-learner

# quick fix to allow using a different port than the default 5000

python testcase1.py $PORT
