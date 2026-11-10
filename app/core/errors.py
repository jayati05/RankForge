class ServiceError(RuntimeError):
    """A client-safe error. Raw filesystem/model exceptions are never returned."""

    def __init__(self, code: str, message: str, status_code: int = 503):
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code
