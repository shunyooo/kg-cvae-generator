from flask_restful import Resource, reqparse, inputs
from emotion_model import __version__
import json


class EmotionAPI(Resource):
    def __init__(self, execution_model, logger):
        self.request_parser = self.load_request_parser()
        self.logger = logger
        self.execution_model = execution_model

    def get(self):
        parsed_args = self.request_parser.parse_args()
        message = parsed_args.get("message")
        messages = parsed_args.get("messages")
        topn = parsed_args['topn']
        sentiment = parsed_args["sentiment"]

        if message:
            response = self.execution_model.process(message, topn=topn, sentiment=sentiment)
            response["model"] = "emotion:%s" % __version__
            logging_payload = {"query": message, "response": response}
            self.logger.log(json.dumps(logging_payload, ensure_ascii=False))

        elif messages:
            response = [
                self.execution_model.process(query, topn=topn, sentiment=sentiment)
                for query in messages.split(',')
            ]
            for res, query in zip(response, messages.split(',')):
                logging_payload = {"query": query, "response": res}
                self.logger.log(json.dumps(logging_payload, ensure_ascii=False))

        else:
            error_description = "message or messages are not include in request_payload"
            return error_description, 400

        return response

    def load_request_parser(self):
        message_get_parser = reqparse.RequestParser()
        message_get_parser.add_argument('message', type=str)
        message_get_parser.add_argument('messages', type=str)
        message_get_parser.add_argument('sentiment', type=inputs.boolean, default=True)
        message_get_parser.add_argument('topn', type=int, default=5)
        return message_get_parser
