from models.DynamicTransolverV1Pre import Model as DynamicTransolverV1PreModel
from models.DynamicTransolverV1_1 import Model as DynamicTransolverV1_1Model
from models import DynamicTransolver, Transolver
import models.TemporalTransolver as TemporalTransolver


def get_model(args):
    model_dict = {
        "Transolver": Transolver,
        "DynamicTransolver": DynamicTransolver,
        "DynamicTransolverV1Pre": DynamicTransolverV1PreModel,
        "DynamicTransolverV1_1": DynamicTransolverV1_1Model,
        "TemporalTransolver": TemporalTransolver,
        # Paper / release alias for TemporalTransolver.
        "DynaSolver": TemporalTransolver,
    }
    return model_dict[args.model].Model(args)
