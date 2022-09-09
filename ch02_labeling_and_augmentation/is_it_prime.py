from math import sqrt
import numpy as np
import re, sys
import pandas as pd

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
    return np.array(list(map(is_prime_int, x)))


def is_prime(n):
    # see http://www.noulakaz.net/weblog/2007/03/18/a-regular-expression-to-check-for-prime-numbers/
    return re.match(r'^1?$|^(11+?)\1+$', '1' * n) == None


def is_prime_int(n):
    """ Given an integer n returns 1 for prime, 0 for non-prime.  """
    # see http://www.noulakaz.net/weblog/2007/03/18/a-regular-expression-to-check-for-prime-numbers/
    # print(n, re.match(r'^1?$|^(11+?)\1+$', '1' * n) == None)
    return 1 if re.match(r'^1?$|^(11+?)\1+$', '1' * n) == None else 0


def get_n_primes(n):

    # N = int(sys.argv[1]) # number of primes wanted (from command-line)
    N = n  # 200  # int(sys.argv[1]) # number of primes wanted (from command-line)
    M = 100             # upper-bound of search space (we search 100 consecutive integers at a time)
    l = list()           # result list

    while len(l) < N:    
        l += filter(is_prime, range(M - 100, M)) # append prime element of [M - 100, M] to l
        # l += filter(isPrime, range(0, M)) # append prime element of [M - 100, M] to l
        # if we haven't found N primes yet, we shift the search space up by 100;
        # i.e., from 0 - 100 to 100 - 200
        M += 100                                # increment upper-bound

    return l[:N]  # only return the number of primes specified  # print(l)


def get_primes_in_first_n_integers(top):
    l = list()
    for n in range(0, top):
        if is_prime(n):
            l += [n]
    return l      


# if __name__ = "__main__":
def get_prime_to_integer_ratios():
    for n in range(100, 1000, 50):
    #n = 103
        n_prime_list = get_n_primes(n)
        npl_ratio = len(n_prime_list)/ n_prime_list[-1]
        print(n_prime_list[-1], len(n_prime_list), npl_ratio)  #, n_prime_list)

# prime_list_up_to_n = get_primes_in_first_n_integers(100)
# print('len =', len(prime_list_up_to_n), prime_list_up_to_n)

def make_prime_list(num_primes):
    outer_prime_list = list()
    for n in range(0,num_primes):
        inner_list = None
        is_prime = is_prime_int(n)
        if is_prime:
            inner_list = [n, 1]     
        else:
            inner_list = [n, 0] 
        outer_prime_list.append(inner_list)
    return outer_prime_list


def make_primes_df(num_primes):
    prime_list = make_prime_list(num_primes)
    df = pd.DataFrame(prime_list, columns =['Number', 'ground_truth'])     
    return df


def make_list_of_num_and_labels(start_num, end_num):
    nums = list(range(start_num, end_num))
    labels = [0] * end_num
    for num in nums:
        labels[num] = is_prime_int(num)            
        
    return nums, labels


if __name__ == "__main__":
    # for n in range(11):
    #     print(n, is_prime(n), prime_factors(n))
    # validation = [22, 11, 7, 2, 32, 101, 102]
    # m = array_map(validation)
    # print(m)
    # print(make_primes_df(200))
    make_list_of_num_and_labels(0, 200)