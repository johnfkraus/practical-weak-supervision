import pandas as pd
from snorkel.labeling import labeling_function

from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter

@labeling_function()
def is_big_num(x):
    return 1 if x.num > 42 else 0


applier = PandasLFApplier([is_big_num])
result = applier.apply(pd.DataFrame(dict(num=[10, 100], text=["hello", "hi"])))

#  array([[0], [1]])