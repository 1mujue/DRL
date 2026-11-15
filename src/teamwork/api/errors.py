"""可被业务后端直接序列化的错误合同。"""


class AlgorithmAPIError(Exception):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details

    def to_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}


def invalid(field, message):
    return AlgorithmAPIError("INVALID_REQUEST", message, field=field)
