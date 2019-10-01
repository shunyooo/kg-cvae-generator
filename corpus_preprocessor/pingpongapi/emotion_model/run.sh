#!/bin/sh
model_name="[Emotion] Emotion Model"

if [ "$1" = "dev" -o "$1" = "" ]
    then
        echo "RUN $model_name || [dev] server"
        PYTHONPATH="../" python run_server.py -c config_dev.yaml
elif [ "$1" = "service" ]
    then
        echo "RUN $model_name || [service] server"
        PYTHONPATH="../" python run_server.py -c config_service.yaml
fi
