# pylint: disable=unused-argument
from loguru import logger


def getLogger(  # type: ignore
    module_name: str = None,
    stdout=None,
):
    """_summary_
    Args:
        stdout (_type_, optional): _description_. Defaults to None.
    Returns:
        _type_: _description_
    """
    return logger
