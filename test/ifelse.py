def main(x):
    if x>1:
        if x>2:
            if x>3:
                if x>5:
                    y=5
                elif x>4:
                    y=4
                else:
                    y=3
            else:
                y=2
        else:
            y=1
    else:
        y=0
    return y

def expected_result():
    return [5,4,3,2,1,0]
