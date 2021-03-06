"""
Basic statistics calculations on binary classification rank order arrays.
Following https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
"""

import numpy as np
#combinatorics_helpers as ch

class Stat:

    "abstract superclass for shared behavior"

    def default_curve(self):
        return AreaUnderCurve(TH, self)

    def curve_points(self, array, close=True):
        return self.default_curve().curve_points(array, close)

    def curve_area(self, array, points=None):
        return self.default_curve().curve_area(array, points)

class Length(Stat):
    "data size"

    abbreviation = "L"

    def __call__(self, array, threshold=None):
        return len(array)

L = Length()

class Selected(Stat):
    "number of entries before the threshold"

    abbreviation = "TH"

    def __call__(self, array, threshold=None):
        return threshold

TH = Selected()

class ConditionalPositive(Stat):
    "The number of real positive cases in the data"

    abbreviation = "P"

    def __call__(self, array, threshold=None):
        return np.count_nonzero(array)

P = ConditionalPositive()

class ConditionalNegative(Stat):
    "The number of real negative cases in the data"

    abbreviation = "N"

    def __call__(self, array, threshold=None):
        return len(array) - np.count_nonzero(array)

N = ConditionalNegative()

class TruePositive(Stat):
    "The number of trues before the threshold"

    abbreviation = "TP"

    def __call__(self, array, threshold):
        return P(array[:threshold])

TP = TruePositive()

class TrueNegative(Stat):
    "The number of falses after the threshold"

    abbreviation = "TN"

    def __call__(self, array, threshold):
        return N(array[threshold:])

TN = TrueNegative()

class FalsePositive(Stat):
    "The number of trues after the threshold"

    abbreviation = "FP"

    def __call__(self, array, threshold):
        return P(array[threshold:])

FP = FalsePositive()

class FalseNegative(Stat):
    "The number of falses before the threshold"

    abbreviation = "FN"

    def __call__(self, array, threshold):
        return N(array[:threshold])

FN = FalseNegative()

class Recall(Stat):
    "proportion of trues before the threshold of all trues"

    abbreviation = "recall"

    def __call__(self, array, threshold):
        Pa = P(array, threshold)
        if Pa > 0:
            return TP(array, threshold) / Pa
        return 1.0  # default

TPR = Recall()
recall = TPR

class FalsePositiveRate(Stat):
    "proportion of trues before the threshold of all trues"

    abbreviation = "FPR"

    def __call__(self, array, threshold):
        fn = FN(array, threshold)
        if fn > 0:
            n = N(array)
            return fn / n
        return 0.0   # default

FPR = FalsePositiveRate()

# skip specificity TNR for now

class Precision(Stat):
    "proportion of trues before the threshold of all results before the threshold"

    abbreviation = "precision"

    def __call__(self, array, threshold):
        if threshold <= 0:
            return 1.0
        return TP(array, threshold) / threshold

PPV = Precision()
precision = PPV

# skip NPV for now
# skip FNR for now
# skip FPR for now

class F1score(Stat):

    abbreviation = 'F1'

    def __call__(self, array, threshold, epsilon=1e-10):
        pr = precision(array, threshold)
        re = recall(array, threshold)
        #("pr, re", pr, re)
        denominator = pr + re
        if abs(denominator) < epsilon:
            return 0.0
        return (pr * re) / denominator

F1 = F1score()

class PhiCoefficient(Stat):

    """
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    """

    abbreviation = "PHI"

    def __call__(self, array, threshold, epsilon=1e-10):
        tp = TP(array, threshold)
        tn = TN(array, threshold)
        fp = FP(array, threshold)
        fn = FN(array, threshold)
        den2 = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if den2 < epsilon:
            return 0.0  # ?????
        numerator = tp * tn - fp * fn
        den = np.sqrt(den2)
        return numerator / den

PHI = PhiCoefficient()

# Specials
class AveragePreference(Stat):

    "Average distance of true values before threshold to the threshold"

    abbreviation = "ATP"

    def __call__(self, array, threshold=None):
        if threshold is None:
            threshold = len(array)
        sum = 0.0
        count = 0
        for i in range(threshold):
            if array[i]:
                sum += (threshold - i)
                count += 1
        if count == 0:
            #print("count of 0", (count, threshold, sum))
            return 0.0  # default
        if count == threshold:
            #print ("count == threshold", (count, threshold, sum))
            return 1.0
        # normalize to [0..1]
        unnormalized = sum / count
        minimum = (count - 1) * 0.5
        maximum = threshold - minimum
        numerator = unnormalized - minimum
        denominator = maximum - minimum
        stat = numerator / denominator
        #print ("normalized", (count, threshold, sum, unnormalized, minimum, maximum, numerator, denominator, stat))
        return stat

ATP = AveragePreference()


class AverageLogPreference(Stat):

    abbreviation = "ALP"

    # use caching for mins and maxes
    mins = {}
    maxes = {}

    def get_max(self, for_length, count):
        assert for_length >= count
        maxes = self.maxes
        key = (for_length, count)
        if key in maxes:
            return maxes[key]
        test_array = np.zeros((for_length,))
        test_array[:count] = 1
        (result, count1) = self.summation(test_array)
        assert count1 == count
        #print ("   max", test_array, count, result)
        maxes[key] = result
        return result

    def get_min(self, for_length, count):
        assert for_length >= count
        mins = self.mins
        key = (for_length, count)
        if key in mins:
            return mins[key]
        test_array = np.zeros((for_length,))
        test_array[-count:] = 1
        (result, count1) = self.summation(test_array)
        assert count1 == count
        mins[key] = result
        #print ("   min", test_array, count, result)
        return result

    def summation(self, array):
        sum = 0.0
        count = 0
        ln = len(array)
        for i in range(len(array)):
            if array[i]:
                sum += np.log(ln - i)
                count += 1
        # ("summation", array, count, sum)
        return (sum, count)

    def __call__(self, array, threshold):
        truncated = array[:threshold]
        (unnormalized, count) = self.summation(truncated)
        if count < 1:
            return 0.0
        if count == threshold:
            return 1.0
        #unnormalized = logs / count
        minimum = self.get_min(threshold, count)
        maximum = self.get_max(threshold, count)
        assert minimum <= unnormalized <= maximum, repr((minimum, unnormalized, maximum))
        numerator = unnormalized - minimum
        denominator = maximum - minimum
        stat = numerator / denominator
        #print ("normalized", (count, threshold, minimum, maximum, numerator, denominator, stat))
        return stat

ALP = AverageLogPreference()


class AverageSquaredPreference(AverageLogPreference):

    abbreviation = "ASP"

    # use caching for mins and maxes
    mins = {}
    maxes = {}

    def summation(self, array):
        sum = 0.0
        count = 0
        ln = len(array)
        for i in range(len(array)):
            if array[i]:
                sum += (ln - i) ** 2
                count += 1
        # ("summation", array, count, sum)
        return (sum, count)

ASP = AverageSquaredPreference()


class AverageExponentialPreference(AverageLogPreference):

    abbreviation = "AEP"

    # use caching for mins and maxes
    mins = {}
    maxes = {}

    def summation(self, array):
        sum = 0.0
        count = 0
        ln = len(array)
        for i in range(len(array)):
            if array[i]:
                sum += np.exp(ln - i)
                count += 1
        # ("summation", array, count, sum)
        return (sum, count)

AEP = AverageExponentialPreference()


class ReversedLogPreference(AverageLogPreference):

    abbreviation = "RLP"

    # use caching for mins and maxes
    mins = {}
    maxes = {}

    def summation(self, array):
        sum = 0.0
        count = 0
        ln = len(array)
        M = np.log(ln + 1)
        for i in range(ln):
            if array[i]:
                sum += M - np.log(i+1)
                count += 1
        # ("summation", array, count, sum)
        return (sum, count)


RLP = ReversedLogPreference()

class AreaUnderCurve(Stat):
    
    "area under the curve for two statistics"

    def __init__(self, x_stat, y_stat, abbreviation=None):
        if abbreviation is None:
            abbreviation = "AUC(%s, %s)" % (x_stat.abbreviation, y_stat.abbreviation)
        self.abbreviation = abbreviation
        self.x_stat = x_stat
        self.y_stat = y_stat

    def curve_points(self, array, close=True):
        points = []
        x_stat = self.x_stat
        y_stat = self.y_stat
        for threshold in range(len(array) + 1):
            x = x_stat(array, threshold)
            y = y_stat(array, threshold)
            points.append([x, y])
        if close:
            # drop verticals to y=0
            [x, y] = points[-1]
            points.append([x, 0])
            [x, y] = points[0]
            points.append([x, 0])
        return points

    def curve_area(self, array, points=None):
        if points is None:
            points = self.curve_points(array, close=True)
        #(points)
        result = 0.0
        [last_x, last_y] = points[-1]
        for point in points:
            [x, y] = point
            base = x - last_x
            height = 0.5 * (y + last_y)
            result += base * height
            [last_x, last_y] = point
        return result

    def __call__(self, array, threshold=None):
        return self.curve_area(array)

AUPR = AreaUnderCurve(recall, precision, "AUPR")

AUROC = AreaUnderCurve(FPR, recall, "AUROC")

ALL_METRICS = [
    L,
    TH,
    P,
    N,
    TP,
    TN,
    FP,
    FN,
    TPR,
    PPV,
    F1,
    PHI,
    ATP,
    ALP,
    ASP,
    RLP,
    AEP,
    AUPR,
    AUROC,
]

ABBREVIATION_TO_STATISTIC = {}
for statistic in ALL_METRICS:
    ABBREVIATION_TO_STATISTIC[statistic.abbreviation] = statistic

def RankOrder(*values):
    "convenience"
    return np.array(values, dtype=np.int)

def test():
    # threshold          1 2 3 4 5 6 7 8 9
    example = RankOrder(1,0,1,1,0,1,0,0,0,1)
    assert L(example) == 10
    ca = L.curve_area(example)
    assert ca == 100, repr(ca)
    assert P(example) == 5
    assert N(example) == 5
    assert TH(example, 3) == 3
    assert TP(example, 3) == 2
    assert TN(example, 3) == 4
    assert FP(example, 3) == 3
    assert FN(example, 3) == 1
    assert TPR(example, 3) == 2/5.0
    assert PPV(example, 3) == 2/3.0
    assert recall(RankOrder(1,0), 1) == 1.0
    p = precision(RankOrder(1,0), 1)
    assert p == 1.0, repr(p)
    f1 = F1(example, 3)
    assert f1 == 1.0/4.0, repr(f1)
    phi = PHI(example, 3)
    assert phi != 0, repr(phi)  #  smoke test
    aupr = AUPR(RankOrder(1,0))
    assert aupr == 1.0, repr(aupr)
    aupr = AUPR(RankOrder(0,1,0,1,0,1,0))
    assert int(aupr * 100) == 37, repr(aupr)
    print()
    auroc = AUROC(RankOrder(1,0))
    assert auroc == 1.0, repr(auroc)
    auroc = AUROC(RankOrder(0,1,0,1,0,1,0))
    assert int(auroc * 100) == 50, repr(auroc)
    auroc = AUROC(RankOrder(0,0,0,1,1))
    assert int(auroc * 100) == 0, repr(auroc)
    # more smoke tests
    for metric in ALL_METRICS:
        for threshold in range(len(example) + 1):
            try:
                assert metric(example, threshold) is not None, repr((None, example, threshold))
            except:
                print("exception at", threshold, "for", metric.abbreviation, metric.__doc__)
                raise
    """
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                for m in (0,1):
                    test = RankOrder(i, j, k, m)
                    print()
                    for threshold in range(5):
                        aa = PHI(test, threshold)
                        print(test, "at", threshold, "gives", aa)
    #print("mins", ALP.mins)
    #print("maxes", ALP.maxes)
    return
    """
    test = ATP(RankOrder(1, 1, 0, 0, 0), 4)
    assert test == 1.0, repr(test)
    test = ATP(RankOrder(1, 0, 1, 0, 0), 4)
    assert test == 2.5 / 3.0, repr(test)
    test = ATP(RankOrder(1, 0, 1, 0, 0), 3)
    assert test == 0.75, repr(test)
    test = ATP(RankOrder(1, 0, 1, 0, 0), 2)
    assert test == 1.0, repr(test)
    test = ATP(RankOrder(1, 0, 1, 0, 0), 1)
    assert test == 1.0, repr(test)
    test = ATP(RankOrder(1, 0, 1, 0, 0), 0)
    assert test == 0, repr(test)

    test = ALP(RankOrder(1, 1, 0, 0, 0), 4)
    assert test == 1.0, repr(test)
    test = ALP(RankOrder(1, 0, 1, 0, 0), 4)
    assert 0 < test < 1
    test = ALP(RankOrder(1, 0, 1, 0, 0), 3)
    assert 0 < test < 1
    test = ATP(RankOrder(1, 0, 1, 0, 0), 2)
    assert test == 1.0, repr(test)
    test = ALP(RankOrder(1, 0, 1, 0, 0), 1)
    assert test == 1.0, repr(test)
    test = ALP(RankOrder(1, 0, 1, 0, 0), 0)
    assert test == 0, repr(test)
    print ("all okay")

if __name__ == "__main__":
    test()