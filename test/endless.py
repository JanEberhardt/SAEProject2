def main(x):
    if True:
        return 1
    else:
        return endless()

def endless():
    return endless()

def expected_result():
    return [1]
