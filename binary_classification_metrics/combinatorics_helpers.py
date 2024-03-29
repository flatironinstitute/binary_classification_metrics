

import numpy as np
import random

class Combinations:

    def __init__(self):
        self.cache = {}
        self.ss_cache = {}
        self.jitter_cach = {}

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

    def n_subsets(self, nelts, max_size):
        c = self.ss_cache
        key = (nelts, max_size)
        if key in c:
            return c[key]
        result = 0
        assert max_size <= nelts
        for size in range(max_size+1):
            result += self.C(nelts, size)
        c[key] = result
        return result

    def n_jitter(self, n_hits, max_shift=None):
        if max_shift is None:
            max_shift = nhits
        # nelts is not relevant
        result = 0
        for subset in range(max_shift + 1):
            result += self.C(n_hits, subset) * (2 ** subset)
        return result

    def indexed_jitter(self, index, nhits, max_shift=None):
        "array of 0 for no change or -1 for shift up or 1 for shift down"
        if max_shift is None:
            max_shift = nhits
        if max_shift <= 0:
            # trivial case -- jitter no elements
            assert index == 0
            return np.zeros((nhits,), dtype=np.int)
        max_shift1 = max_shift - 1
        n_jitter1 = self.n_jitter(nhits, max_shift1)
        if (index < n_jitter1):
            # jitter a smaller number of elements
            return self.indexed_jitter(index, nhits, max_shift1)
        # otherwise jitter exactly max_size elts
        offset = index - n_jitter1
        (combination_number, subset_number) = divmod(offset, (2 ** max_shift))
        combo = self.indexed_combination_array(combination_number, nhits, max_shift)
        subset = binary_array(max_shift, subset_number)
        i_subset = 0
        for i in range(nhits):
            if combo[i]:
                if subset[i_subset]:
                    combo[i] = -1
                else:
                    combo[i] = 1 # redundant
                i_subset += 1
        assert i_subset == max_shift
        return combo

    def indexed_subset(self, index, nelts, max_size):
        assert nelts >= max_size >= 0
        assert index >= 0
        this_size = self.C(nelts, max_size)
        # return at current max size if possible
        if index < this_size:
            return self.indexed_combination_array(index, nelts, max_size)
        # otherwise return something smaller
        return self.indexed_subset(index - this_size, nelts, max_size-1)

    def all_subsets_of_max_size(self, nelts, max_size):
        ln = self.n_subsets(nelts, max_size)
        result = np.zeros((nelts, ln), dtype=np.int)
        for i in range(ln):
            result[:, i] = self.indexed_subset(i, nelts, max_size)
        return result

    def all_jitters_of_max_size(self, nelts, max_size):
        # nelts is not relevant
        ln = self.n_jitter(nelts, max_size)
        result = np.zeros((nelts, ln), dtype=np.int)
        for i in range(ln):
            result[:, i] = self.indexed_jitter(i, nelts, max_size)
        return result

    def all_combinations(self, n, k):
        "Combinations array with combinations as columns"
        cc = self.C(n, k)
        result = np.zeros((n, cc), dtype=np.int)
        for i in range(cc):
            result[:, i] = self.indexed_combination_array(i, n, k)
        return result

    def indexed_combination_array(self, index, n, k, i=0, array=None):
        "Get a combination by index"
        #print ("indexed index, n, k, i, array", index, n, k, i, array)
        if array is None:
            assert i == 0
            array = np.zeros((n,), dtype=np.int)
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

    def random_combinations_no_replacement(self, n, k, number, C=None, ind=None):
        if C is None:
            C = self.C
            ind = self.indexed_combination_array
        #cc = self.C(n, k)
        cc = C(n, k)
        assert number <= cc, "not enough possible combinations " + repr((cc, number))
        result = np.zeros((n, number))
        indices = list(range(cc))
        for i in range(number):
            choice = random.randrange(len(indices))
            index = indices[choice]
            #result[:, i] = self.indexed_combination_array(index, n, k)
            result[:, i] = ind(index, n, k)
            del indices[choice]
        return result

    def random_subsets_of_max_size_no_replacement(self, nelts, max_size, number):
        return self.random_combinations_no_replacement(
            n=nelts,
            k=max_size,
            number=number,
            C=self.n_subsets,
            ind=self.indexed_subset,
        )

    def random_combinations_with_replacement(self, n, k, number, C=None, ind=None):
        if C is None:
            C = self.C
            ind = self.indexed_combination_array
        cc = C(n, k)
        result = np.zeros((n, number))
        for i in range(number):
            index = random.randrange(cc)
            result[:, i] = ind(index, n, k)
        return result

    def random_subsets_of_max_size_with_replacement(self, nelts, max_size, number):
        return self.random_combinations_with_replacement(
            n=nelts,
            k=max_size,
            number=number,
            C=self.n_subsets,
            ind=self.indexed_subset,
        )

    def limited_combinations(self, n, k, all_limit=3000, replace_limit=10e6, C=None, replace=None, no_replace=None, all=None):
        if C is None:
            C = self.C
            replace = self.random_combinations_with_replacement
            no_replace = self.random_combinations_no_replacement
            all = self.all_combinations
        "Return all combinations or a random selection if there are too many"
        assert all_limit < replace_limit
        cc = C(n, k)
        if cc > replace_limit:
            return replace(n, k, all_limit)
        if cc > all_limit:
            return no_replace(n, k, all_limit)
        # otherwise return all
        return all(n, k)

    def limited_subsets_of_max_size(self, n, k, all_limit=3000, replace_limit=10e6, C=None, replace=None, no_replace=None, all=None):
        return self.limited_combinations(
            n=n, 
            k=k, 
            all_limit=all_limit, 
            replace_limit=replace_limit, 
            C=self.n_subsets, 
            replace=self.random_combinations_with_replacement, 
            no_replace=self.random_subsets_of_max_size_no_replacement, 
            all=self.all_subsets_of_max_size,
        )

def limited_index_selection(max_index, all_limit=3000, replace_limit=100000):
    # XXXX refactor to use this everywhere and remove redundant lagic
    if max_index <= all_limit:
        return range(max_index)
    r = range(all_limit)
    result = list(r)
    if max_index < replace_limit:
        # selection with replacement
        indices = list(range(max_index))
        for i in r:
            choice = random.randrange(len(indices))
            index = indices[choice]
            del indices[choice]
            result[i] = choice
    else:
        for i in r:
            index = random.randrange(max_index)
            result[i] = index
    return result

# singleton
COMBOS = Combinations()

# method abbreviations
C = COMBOS.C
all_combinations = COMBOS.all_combinations
random_combinations_no_replacement = COMBOS.random_combinations_no_replacement
random_combinations_with_replacement = COMBOS.random_combinations_with_replacement
limited_combinations = COMBOS.limited_combinations
random_subsets_of_max_size_no_replacement = COMBOS.random_subsets_of_max_size_no_replacement
random_subsets_of_max_size_with_replacement = COMBOS.random_subsets_of_max_size_with_replacement
n_subsets = COMBOS.n_subsets
all_subsets_of_max_size = COMBOS.all_subsets_of_max_size
limited_subsets_of_max_size = COMBOS.limited_subsets_of_max_size
indexed_jitter = COMBOS.indexed_jitter
indexed_combination_array = COMBOS.indexed_combination_array
n_jitter = COMBOS.n_jitter

def jitter_array(combination_array):
    n = len(combination_array)
    k = int(sum(combination_array))
    num = n_jitter(k, k)
    result = np.zeros((n, num), dtype=np.int)
    for i in range(num):
        jt = indexed_jitter(i, k)
        result[:, i] = apply_jitter(jt, combination_array)
    return dedup_combos(result)

def flip_combo_bits(array, combo_array):
    s = (n,) = array.shape
    assert combo_array.shape == s
    result = np.array(array, dtype=np.int)
    for i in range(n):
        if combo_array[i]:
            result[i] = 1 - array[i]
    return result

def jitter_and_perturb(combination_array, flip_bits=1):
    n = len(combination_array)
    k = int(sum(combination_array))
    jnum = n_jitter(k, k)
    cnum = C(n, flip_bits)
    num = jnum * cnum
    chosen_indices = limited_index_selection(num)
    ln = len(chosen_indices)
    result = np.zeros((n, ln), dtype=np.int)
    for i in range(ln):
        index = chosen_indices[i]
        (jitter_index, c_index) = divmod(index, cnum)
        jt = indexed_jitter(jitter_index, k)
        jittered = apply_jitter(jt, combination_array)
        #print ("            jt", jt, "jittered", jittered)
        combo = indexed_combination_array(c_index, n, flip_bits)
        jittered_and_flipped = flip_combo_bits(jittered, combo)
        #print ("            combo", combo, "jittered", jittered_and_flipped)
        # sanity chack
        if 1:
            sflipped = sum(jittered_and_flipped)
            assert sflipped - flip_bits <= sum(combination_array) <= sflipped + flip_bits
        result[:, i] = jittered_and_flipped
    return dedup_combos(result)

def combo_rotations(combo):
    (n,) = combo.shape
    result = np.zeros((n,n), dtype=np.int)
    for i in range(n):
        split = n-i
        result[:i, i] = combo[split:]
        result[i:, i] = combo[:split]
    return result

def irreducibles(n, k):
    assert (k >= 2) and (n-k > 2), "Must have gaps of size at least 2 for this logic: " + repr((n, k))
    compact = compact_extrema(n, k)
    spread = spread_extrema(n, k)
    if spread is not None:
        result = np.hstack([compact, spread])
    else:
        result = compact
    return result

def compact_extrema(n, k):
    combo = np.zeros((n,), dtype=np.int)
    positive_combos = (n - k + 1) + (n - k) * (k - 1)
    ncombos = positive_combos
    result = np.zeros((n,ncombos), dtype=np.int)
    index = 0
    for n1 in range(n - k + 1):
        combo[:] = 0
        combo[n1:n1+k] = 1
        result[:, index] = combo
        index += 1
        if n1 < n - k:
            combo[:] = 0
            for k1 in range(1, k):
                combo[:] = 0
                combo[n1:n1+k+1] = 1
                combo[n1+k1] = 0
                result[:, index] = combo
                index += 1
    assert index == ncombos, repr((index, ncombos))
    return result

def spread_extrema(n, k):
    if n < 4 or k < 3 or n < k + 3:
        return None
    L = []
    for split in range(1, k):
        for offset in range(split+2, n - (k - split) + 1):
            combo = np.zeros((n,), dtype=np.int)
            combo[:split] = 1
            shift_end = offset + (k - split)
            combo[offset: shift_end] = 1
            L.append(combo)
    # also add reversals where end is 0
    for c in L[:]:
        if c[-1] == 0:
            reversed = c[::-1]
            L.append(reversed) 
    ncombos = len(L)
    result = np.zeros((n,ncombos), dtype=np.int)
    for (i,c) in enumerate(L):
        result[:, i] = c
    return result

def irreducibles0(n, k):
    # saved version
    assert (k >= 2) and (n-k > 2), "Must have gaps of size at least 2 for this logic: " + repr((n, k))
    combo = np.zeros((n,), dtype=np.int)
    positive_combos = (n - k + 1) + (n - k) * (k - 1)
    negative_combos = (k - 1) + (k - 2) * (n - k - 1)
    ncombos = positive_combos + negative_combos
    result = np.zeros((n,ncombos), dtype=np.int)
    index = 0
    for n1 in range(n - k + 1):
        combo[:] = 0
        combo[n1:n1+k] = 1
        result[:, index] = combo
        index += 1
        if n1 < n - k:
            combo[:] = 0
            for k1 in range(1, k):
                combo[:] = 0
                combo[n1:n1+k+1] = 1
                combo[n1+k1] = 0
                result[:, index] = combo
                index += 1
    for k1 in range(1, k):
        combo[:] = 0
        combo[:k1] = 1
        combo[n-(k-k1):] = 1
        #print ("negative base", combo)
        result[:, index] = combo
        index += 1
        if k1 > 1:
            for shift in range(0, n-k-1):
                combo[:] = 0
                combo[:k1-1] = 1
                combo[n-(k-k1):] = 1
                combo[k1 + shift] = 1
                #print ("    negative shift", combo)
                result[:, index] = combo
                index += 1
    assert index == ncombos, repr((index, ncombos))
    return result

def dedup_combos(combos):
    (n, num) = combos.shape
    S = set()
    for i in range(num):
        S.add(tuple(combos[:, i]))
    result = np.zeros( (n, len(S)))
    for (i, c) in enumerate(S):
        result[:, i] = c
    return result

def limited_jitter(combination_array, all_limit=3000, replace_limit=20000):
    n = len(combination_array)
    k = int(sum(combination_array))
    num = n_jitter(k, k)
    #print ("selecting jitter from", num)
    if num > replace_limit:
        result = np.zeros((n, all_limit), dtype=np.int)
        for i in range(all_limit):
            choice = random.randrange(num)
            jt = indexed_jitter(choice, k)
            result[:, i] = apply_jitter(jt, combination_array)
        return dedup_combos(result)
    all_jitter = jitter_array(combination_array)
    (n, max_jitter) = all_jitter.shape
    if max_jitter <= all_limit:
        return all_jitter
    indices = list(range(max_jitter))
    result = np.zeros((n, all_limit), dtype=np.int)
    for i in range(all_limit):
        choice = random.randrange(len(indices))
        index = indices[choice]
        del indices[choice]
        c = all_jitter[:, index]
        result[:, i] = c
    return result

def apply_jitter(jitter_array, to_array):
    #print ("apply jitter", jitter_array, "to", to_array)
    result = np.array(to_array, dtype=np.int)
    jindex = 0
    for index in range(len(to_array)):
        if to_array[index]:
            jitter = jitter_array[jindex]
            if jitter < 0 and index > 0:
                (result[index], result[index-1]) = (result[index-1], result[index])
            elif jitter > 0 and index < len(to_array) - 1:
                (result[index], result[index+1]) = (result[index+1], result[index])
            jindex += 1
    assert jindex == len(jitter_array)
    return result

def variations_of_max_size(combination, max_size):
    combination = np.array(combination, dtype=np.int)
    nelts = len(combination)
    flips = limited_subsets_of_max_size(nelts, max_size)
    result = np.zeros(flips.shape)
    (M, nflips) = flips.shape
    assert M == nelts
    for i in range(nflips):
        result[:, i] = np.logical_xor(flips[:, i], combination)
    return result

def binary_array(size, n):
    n1 = n
    result = np.zeros((size,), dtype=np.int)
    for i in range(size):
        if (1 & n) > 0:
            result[i] = 1
        n = (n >> 1)
    assert n == 0, "number too large for array size " + repr((n1, size))
    return result 

def all_subsets(nelts):
    assert nelts > 0
    size = 1 << nelts
    # combinations are columns again
    result = np.zeros((nelts, size), dtype=np.int)
    for i in range(size):
        result[:, i] = binary_array(nelts, i)
    return result

def subsets_no_replacement(nelts, size):
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

def subsets_with_replacement(nelts, size):
    result = np.zeros((nelts, size), dtype=np.int)
    limit = 1 << nelts
    assert limit >= size
    for i in range(size):
        j = random.randrange(limit)
        result[:, i] = binary_array(nelts, j)
    return result

def all_limited_subsets(nelts, all_limit=3000, replace_limit=10e6):
    limit = 1 << nelts
    if limit > replace_limit:
        return subsets_with_replacement(nelts, all_limit)
    if limit > all_limit:
        return subsets_no_replacement(nelts, all_limit)
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
    print ("...jitter...")
    for nhits in range(1, 4):
        print()
        test_array = indexed_combination_array(nhits * 2 + 1, n=10, k=nhits)
        max_jitter = n_jitter(nhits, nhits)
        print("Nhits", nhits, test_array)
        try:
            for index in range(1000000):
                jt = indexed_jitter(index, nhits)
                applied = apply_jitter(jt, test_array)
                print ("   ", index, jt, applied)
        except AssertionError:
            print("  assertion error")
            assert index == max_jitter
        print()
    test_array = indexed_combination_array(155, 12, 4)
    jittered = jitter_array(test_array)
    print (test_array, "jitterred")
    for i in range(jittered.shape[1]):
        print("   ", i, " :: ", jittered[:, i])
    print (test_array, "limited jitterred", 3)
    jittered = limited_jitter(test_array, all_limit=3, replace_limit=12)
    for i in range(jittered.shape[1]):
        print("   ", i, " :: ", jittered[:, i])
    print ("...jitter and perturb...")
    test_array = indexed_combination_array(50, 15, 2)
    flip = 1
    print("   testing with", test_array, "with", flip, "flips")
    jf = jitter_and_perturb(test_array, flip)
    print("    got", jf.shape)
    jf = jf[:, :10]  # just look at the first 10
    for i in range(jf.shape[1]):
        print("        ", jf[:, i])
    for (n, k) in [(7,3),(5,2),(8,2),(9,3),(10,4)]:
        print()
        print("... irreducible", n, k)
        #ir = spread_extrema(n, k)
        #ir = compact_extrema(n, k)
        ir = irreducibles(n, k)
        if ir is None:
            print ("NONE")
            continue
        for i in range(ir.shape[1]):
            print("        ", ir[:, i])
        dd = dedup_combos(ir)
        assert ir.shape == dd.shape, "DUPLICATES!"
    print ("done.")

if __name__ == "__main__":
    test()
