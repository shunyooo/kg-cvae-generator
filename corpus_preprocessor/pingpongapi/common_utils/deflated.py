import gzip
import zlib
import json

from flask import request


def get_json_data():
    """json 데이터를 request로부터 받아옵니다.

    gzip, deflate를 쓸 가능성이 있는 리퀘스트의 경우 이 함수를 써주세요.

    TODO: 어떻게 abort 할껀가?? OSError
    """
    content_encoding = request.headers.get('Content-Encoding', '')
    if content_encoding == 'deflate':
        return json.loads(zlib.decompress(request.data).decode('utf8'))
    elif content_encoding == 'gzip':
        return json.loads(gzip.decompress(request.data).decode('utf8'))
    else:
        return request.get_json()
