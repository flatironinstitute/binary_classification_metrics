
import numpy as np
from numpy.linalg import norm
import c_stats
import combinatorics_helpers

class Ranker:

    def __init__(self, combinations_array):
        # combinations are columns of the array
        (self.combo_length, self.ncombos) = combinations_array.shape
        self.array = combinations_array

    def rank(self, metric1, metric2):
        array = self.array
        pairs = np.zeros((self.ncombos, 2), dtype=np.float)
        rank_index = []
        ln = self.combo_length
        for i in range(self.ncombos):
            combo = array[:, i]
            r1 = metric1(combo, ln)
            r2 = metric2(combo, ln)
            pairs[i] = [r1, r2]
            rank_index.append((r1, r2, i))
        rank_index.sort()
        ranked_pairs = np.zeros((self.ncombos, 2), dtype=np.float)
        ranked_array = np.zeros(array.shape, dtype=np.int)
        # rank from highest to lowest
        for (destination_index, (r1, r2, source_index)) in enumerate(reversed(rank_index)):
            ranked_pairs[destination_index] = (r1, r2)
            ranked_array[:, destination_index] = array[:, source_index]
        return Ranking(ranked_array, ranked_pairs, metric1, metric2)

def get_ranker(n, k):
    "get default ranker -- restrict size if needed"
    combo_array = combinatorics_helpers.limited_combinations(n, k)
    return Ranker(combo_array)

class Ranking:

    def __init__(self, sorted_array, sorted_pairs, metric1, metric2):
        (self.combo_length, self.ncombos) = sorted_array.shape
        self.metric1 = metric1
        self.metric2 = metric2
        self.array = sorted_array
        self.pairs = sorted_pairs

    def nearest_pair(self, xy):
        # optimized
        xy = np.array(xy)
        pairs = self.pairs
        diffs = pairs - xy
        norms = norm(diffs, axis=1)
        index = np.argmin(norms)
        nearest = pairs[index]
        return (index, nearest)

    def combination(self, column):
        column = int(column)
        if column < 0 or column >= self.ncombos:
            return None
        combo_array = self.array[:, column]
        (r1, r2) = self.pairs[column]
        return ComboStat(column, combo_array, r1, r2, self.metric1, self.metric2)

class ComboStat:

    def __init__(self, column, combo_array, r1, r2, metric1, metric2):
        (self.column, self.combo_array, self.r1, self.r2, self.metric1, self.metric2) = (
            column, combo_array, r1, r2, metric1, metric2)

    def curve_info(self, index=1):
        ms = [self.metric1, self.metric2]
        m = ms[index - 1]
        points = m.curve_points(self.combo_array)
        area = m.curve_area(self.combo_array, points=points)
        return (points, area)

def test():
    ranker = get_ranker(4,2)
    ranking = ranker.rank(c_stats.AUPR, c_stats.AUROC)
    assert ranking.nearest_pair((1,1)) is not None
    print (ranking.array.shape)
    combo_stat = ranking.combination(3)
    c1 = combo_stat.curve_info(1)
    print(c1)
    assert c1 is not None
    c2 = combo_stat.curve_info(2)
    print(c2)
    assert c2 is not None
    print ("all ok")

if __name__ == "__main__":
    test()
