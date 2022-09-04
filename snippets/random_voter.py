import numpy as np
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import RandomVoter

L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
random_voter = RandomVoter()
predictions = random_voter.predict_proba(L)
