from ltn_log.core import Variable, Predicate, Constant, Function, Connective, diag, undiag, Quantifier, \
    LTNObject, process_ltn_objects, LambdaModel
import torch
from ltn_log import fuzzy_ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")