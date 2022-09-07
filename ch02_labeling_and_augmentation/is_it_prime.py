from math import sqrt
import numpy as np

def is_prime(n):    
    # this flag maintains status whether the n is prime or not
    prime_flag = 0
    if(n > 1):
        for i in range(2, int(sqrt(n)) + 1):
            if (n % i == 0):
                prime_flag = 1
                break
        if (prime_flag == 0):
            return 1  # True  # print("True")
        else:
            return 0   #False
            # print("False")
    else:
        return 0  # False  # print("False")


def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def array_map(x):
    return np.array(list(map(is_prime, x)))

        
if __name__ == "__main__":
    for n in range(11):
        print(n, is_prime(n), prime_factors(n))
    validation = [22, 11, 7, 2, 32, 101, 102]
    m = array_map(validation)
    print(m)
