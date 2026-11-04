from enum import Enum

# communication parameters
class COParams(Enum):

    def __init__(self, name):
        self.n = name

    CONVERGED = 'Converged'
    KMIN = 'kmin'
    KMAX = 'kmax'
    CENTROIDS = 'centroids'
    FINISHED = 'finished'


    ROW_NAMES = 'Row names'
    SUMS = 'Sums'
    NAN = 'Nans'
    MEANS = 'Means'
    SUM_OF_SQUARES = 'Sum of Squares'
    STDS = 'Stds'
    SAMPLE_COUNT = 'Sample count'
