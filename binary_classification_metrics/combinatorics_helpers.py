

import numpy as np
import random

class Combinations:

    def __init__(self):
        self.cache = {}

    def C(self, n, k):
        assert n >= k, repr((n,k))
        cache = self.cache
        key = (n, k)
        if key in cache:
            return cache[key]
        if k == 0 or n == k:
            result = 1
        else:
            accept = self.C(n-1, k-1)
            reject = self.C(n-1, k)
            result = accept + reject
        cache[key] = result
        return result

    def all_combinations(self, n, k):
        "Combinations array with combinations as columns"
        cc = self.C(n, k)
        result = np.zeros((n, cc))
        for i in range(cc):
            result[:, i] = self.indexed_combination_array(i, n, k)
        return result

    def indexed_combination_array(self, index, n, k, i=0, array=None):
        "Get a combination by index"
        #print ("indexed index, n, k, i, array", index, n, k, i, array)
        if array is None:
            assert i == 0
            array = np.zeros((n,), dtype=np.bool)
        assert n > 0
        assert 0 <= k <= n
        ln = len(array)
        remaining = ln - i
        assert k <= remaining <= n, repr((k, remaining, n))
        # sanity check
        maxindex = self.C(n, k)
        assert index < maxindex, "bad index " + repr((index, maxindex))
        if k == 0:
            #print("k is zero, select nothing else")
            array[i:] = 0
        elif k == remaining:
            #print("k is remaining: select everything (else)")
            array[i:] = 1
        else:
            accept = self.C(remaining-1, k-1)
            if index < accept:
                #print("accept position ", i, index, accept)
                array[i] = 1
                return self.indexed_combination_array(index, n-1, k-1, i+1, array)
            else:
                #print("reject position ", i, index, accept)
                array[i] = 0
                return self.indexed_combination_array(index-accept, n-1, k, i+1, array)
        return array

    def random_combinations_no_replacement(self, n, k, number):
        cc = self.C(n, k)
        assert number <= cc, "not enough possible combinations " + repr((cc, number))
        result = np.zeros((n, number))
        indices = list(range(cc))
        for i in range(number):
            choice = random.randrange(len(indices))
            index = indices[choice]
            result[:, i] = self.indexed_combination_array(index, n, k)
            del indices[choice]
        return result

    def random_combinations_with_replacement(self, n, k, number):
        cc = self.C(n, k)
        result = np.zeros((n, number))
        for i in range(number):
            index = random.randrange(cc)
            result[:, i] = self.indexed_combination_array(index, n, k)
        return result

    def limited_combinations(self, n, k, all_limit=3000, replace_limit=10e6):
        "Return all combinations or a random selection if there are too many"
        assert all_limit < replace_limit
        cc = self.C(n, k)
        if cc > replace_limit:
            return self.random_combinations_with_replacement(n, k, all_limit)
        if cc > all_limit:
            return self.random_combinations_no_replacement(n, k, all_limit)
        # otherwise return all
        return self.all_combinations(n, k)

# singleton
COMBOS = Combinations()

# method abbreviations
C = COMBOS.C
all_combinations = COMBOS.all_combinations
random_combinations_no_replacement = COMBOS.random_combinations_no_replacement
random_combinations_with_replacement = COMBOS.random_combinations_with_replacement
limited_combinations = COMBOS.limited_combinations

def binary_array(size, n):
    result = np.zeros((size,), dtype=np.int)
    for i in range(size):
        if (1 & n) > 0:
            result[i] = 1
        n = (n >> 1)
    return result

def all_subsets(nelts):
    assert nelts > 0
    size = 1 << nelts
    # combinations are columns again
    result = np.zeros((nelts, size), dtype=np.int)
    for i in range(size):
        result[:, i] = binary_array(nelts, i)
    return result

def limited_subsets_no_replacement(nelts, size):
    result = np.zeros((nelts, size), dtype=np.int)
    limit = 1 << nelts
    assert limit >= size
    choices = list(range(limit))
    for i in range(size):
        j = random.randrange(len(choices))
        choice = choices[j]
        del choices[j]
        result[:, i] = binary_array(nelts, choice)
    return result

def limited_subsets_with_replacement(nelts, size):
    result = np.zeros((nelts, size), dtype=np.int)
    limit = 1 << nelts
    assert limit >= size
    for i in range(size):
        j = random.randrange(limit)
        result[:, i] = binary_array(nelts, j)
    return result

def limited_subsets(nelts, all_limit=3000, replace_limit=10e6):
    limit = 1 << nelts
    if limit > replace_limit:
        return limited_subsets_with_replacement(nelts, all_limit)
    if limit > all_limit:
        return limited_subsets_no_replacement(nelts, all_limit)
    return all_subsets(nelts)

def test():
    for (n, k, expect) in [(1,1,1), (2,1,2), (3,3,1), (3,2,3), (5,3,10)]:
        cc = C(n,k)
        #pr (n, k, cc, expect)
        assert cc == expect, repr((n, k, cc, expect))
    #for index in range(11):
    #    cc = COMBOS.indexed_combination_array(index, 3,2)
    #    print ("   FINAL: ", index, tuple(cc))
    for (n, k) in [(1,1), (2,1), (2,2), (6,3), (9,4)]:
        limit = C(n,k)
        print ("testing", n, k, limit)
        collector = set()
        for index in range(limit):
            cc = COMBOS.indexed_combination_array(index, n, k)
            collector.add(tuple(cc))
        assert len(collector) == limit
        ac = all_combinations(n, k)
        assert ac.shape == (n, limit)
        collector = set()
        for index in range(limit):
            cc = ac[:, index]
            collector.add(tuple(cc))
        assert len(collector) == limit
        ac = random_combinations_no_replacement(n, k, limit)
        assert ac.shape == (n, limit)
        collector = set()
        for index in range(limit):
            cc = ac[:, index]
            collector.add(tuple(cc))
        assert len(collector) == limit
        rc = random_combinations_with_replacement(n, k, limit + 3)
        for index in range(limit + 3):
            cc =rc[:, index]
            assert tuple(cc) in collector
        if limit < 8:
            print (list(collector))
        try:
            cc = COMBOS.indexed_combination_array(limit + 1, n, k)
        except AssertionError:
            print ("got expected count and exception")
        else:
            raise ValueError("no error? " + repr((n, k, limit, cc)))
    print("==== bigger tests ====")
    for (k, n) in [(5,8), (6,13), (10,20), (5, 30), (13, 44)]:
        print("choosing", k, "from", n, "possible", C(n,k))
        lc = limited_combinations(n, k)
        print(lc.shape)
        print(lc[:, 5])
        for index in range(lc.shape[1]):
            cc = lc[:, index]
            assert len(cc) == n
            assert np.count_nonzero(cc) == k


if __name__ == "__main__":
    test()
