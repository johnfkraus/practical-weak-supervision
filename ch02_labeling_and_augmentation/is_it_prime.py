"""
This module contains various utility functions that can be used in Jupyter notebooks to help deal with prime numbers in connection with a data labeling example in Chapter 2 of the Practical Weak Supervision book.  Not all the code here is actually being used, so some pruning might be appropriate as time permits.
"""

import math
#  from math import sqrt
import numpy as np
import re
import pandas as pd
import collections
# import timing
# from decorators import timer
# from gtimer_decorator import timer_func
from time import time

__author__ = "John Kraus"
__email__ = "john.f.kraus19.ctr@mail.mil"
__status__ = "Development"
__version__ = "0.0.1"

  
def timer_func(func):
    # Print the execution time of the passed function object
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func



#@timer_func
def is_prime(n):
    # this flag maintains status whether the n is prime or not
    if n is None:
        raise ValueError("is_prime(n) requires non-null parameter.")
    prime_flag = 0
    try:
        if(n > 1):
            for i in range(2, int(math.sqrt(n)) + 1):
                if (n % i == 0):
                    prime_flag = 1
                    break
            if (prime_flag == 0):
                return 1 
            else:
                return 0
        else:
            return 0
    except ValueError as e:
        print("exception", str(e))


def prime_factors(n):
    """"Given an integer, returns a list of its prime factors."""
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


def get_prime_to_integer_ratios():
    for n in range(100, 1000, 50):
        n_prime_list = get_n_primes(n)
        npl_ratio = len(n_prime_list)/ n_prime_list[-1]
        print(n_prime_list[-1], len(n_prime_list), npl_ratio)  #, n_prime_list)


def prob_rand_int_less_than_n_is_prime(n):
    """According to the prime number theorem for large enough N, the probability that a random integer not greater than N is prime is very close to 1 / log(N).   https://en.wikipedia.org/wiki/Prime_number_theorem"""
    return 1 / math.log(n)


def prime_counting_function(n):
    """ number of primes less than or equal to n; https://en.wikipedia.org/wiki/Prime_number_theorem """
    return n / math.log(n)



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
    """ Return a DataFrame with n=num_primes integers in one column and prime=1 or non-prime=0 in the second column. """
    prime_list = make_prime_list(num_primes)
    df = pd.DataFrame(prime_list, columns =['Number', 'ground_truth'])     
    return df


def make_list_of_ints_and_prime_labels(start_num, end_num):
    """ return two lists: nums = [0,0,1,3,...] and labels = [0,0,1,1,...] where label 0 means non-prime and 1 means prime. """
    nums = list(range(start_num, end_num))
    labels = [0] * end_num
    for num in nums:
        labels[num] = is_prime_int(num)            
        
    return nums, labels


def test_return_prime_factors(n=360):
    pfs = prime_factors(n)
    # No integer less than 2 should be a prime factor.
    assert any(x < 2 for x in pfs) == False
    # All factors should be prime numbers.
    assert (is_prime(x) for x in pfs)
    # The product of all the prime factors should equal n.
    assert np.prod(pfs) == n
    if n == 360:
        expected = list([2, 5, 2, 3, 3, 2])
        print(expected)
        assert len(pfs) == len(expected)
        # Do the two lists contain the same prime factors regardless of order?
        assert collections.Counter(pfs) == collections.Counter(expected)

    print("PASS test_return_prime_factors(", n, ")")


def test():
    test_return_prime_factors()
    test_return_prime_factors(111)


def demo():
    print(make_list_of_ints_and_prime_labels(0, 200))
    # get_prime_to_integer_ratios()
    print(prob_rand_int_less_than_n_is_prime(10000000))
    print(prime_counting_function(10000000))
    print(make_primes_df(11))


if __name__ == "__main__":
    """ Run some tests. """
    test()
    demo()
    is_prime_with_timer = timer_func(is_prime)
    
    print(is_prime_with_timer(83))  #  True, Function 'is_prime' executed in 0.0002s

    # print(is_prime_with_timer(998077))  #  True, Function 'is_prime' executed in 246.2196s
    
    
    big_int = 797421915493860356140028996109826257509243069764668430517233998159752495771749751540427498177503816731916148527883659572687040623351311710753167207791930031079587803266286682404595676276042945084541365179468552061417684453547762289380306912014285651033639519236303675743058069222633073153462634860981
    
    
    print(is_prime_with_timer(big_int))  # OverflowError: cannot fit 'int' into an index-sized integer
    