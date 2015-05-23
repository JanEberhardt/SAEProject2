def main(x):
    ret = 10
    if x<0:
        ret = 1
    elif x>2:
        ret = 2
    return ret
   
def expected_result():
    return [1,2,10]
