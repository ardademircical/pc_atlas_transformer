import logging
from test2 import test2
logging.basicConfig(level=logging.DEBUG)

def test():
    logging.info("Test started.")
    logging.info("")
    logging.info("Test ended.")
    # return None

if __name__ == "__main__":
    test()
    test2()