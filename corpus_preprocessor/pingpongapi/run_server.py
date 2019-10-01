import warnings

from flask import Flask
from flask_restful import Api

from emotion_model.utils import load_main_config
from emotion_model.service.models.emotion_execution import EmotionExecution
from emotion_model.service.api import EmotionAPI
from emotion_model.logger import Logger

import time

# python 3.6 과 관련된 tensorflow warning 이 너무 많이 나와서 임시로 한번만 나오도록 설정
warnings.filterwarnings('once')

app = Flask(__name__)
api = Api(app)

# Logging Unit
config = load_main_config()
execution_model = EmotionExecution(config)
logger = Logger(config.get("log_path"), stream=False)


start_time = time.time()
for i in range(10000):
    execution_model.pos_neg_process("오늘 나 야근한다")
elapsed_time = time.time() - start_time
print("elapsed_time", elapsed_time)


start_time = time.time()
result = execution_model.batch_pos_neg_process(["오늘 나 야근한다"]*10000)
elapsed_time = time.time() - start_time
print("batch_elapsed_time", elapsed_time)
print("AAAA")
print(result[0:100])

"""
if __name__ == '__main__':
    emotion_api_args = {"execution_model": execution_model, "logger": logger}
    api.add_resource(EmotionAPI, '/emotion', resource_class_kwargs=emotion_api_args)
    app.run(host='0.0.0.0', port=config['port'], threaded=True)
"""