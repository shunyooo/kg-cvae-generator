import logging


class Logger:
    def __init__(self, log_file=None, stream=True):
        # logger 인스턴스를 생성 및 로그 레벨 설정
        self.logger = logging.getLogger("crumbs")
        self.logger.setLevel(logging.DEBUG)

        # formatter 생성
        formatter = logging.Formatter('[%(levelname)s|] %(asctime)s > %(message)s')

        # fileHandler와 StreamHandler를 생성
        if log_file:
            self.fileHandler = logging.FileHandler(log_file)
            self.fileHandler.setFormatter(formatter)
            self.logger.addHandler(self.fileHandler)

            if stream is True:
                self.streamHandler = logging.StreamHandler()
                self.streamHandler.setFormatter(formatter)
                self.logger.addHandler(self.streamHandler)
        else:
            self.logger = None

    def log(self, message):
        if self.logger:
            self.logger.info(message)

    def unload(self):
        if self.logger:
            try:
                self.fileHandler.close()
            except AttributeError:
                pass
            try:
                self.streamHandler.close()
            except AttributeError:
                pass
