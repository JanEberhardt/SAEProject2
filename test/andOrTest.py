def main(x,y):
    if x<4 and y<4:
        return 0 + orTest(x,y)
    else: 
        return 1 + orTest(x,y)

def orTest(a,b):
    if a<0 or b<0:
        return 10
    else:
        return 11

def expected_result():
    return [10, 11, 12]
