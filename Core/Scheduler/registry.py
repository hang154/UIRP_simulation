from Core.Scheduler.task_selector.fifo import FIFOTaskSelector
from Core.Scheduler.metric_evaluator.baseline import BaselineEvaluator
from Core.Scheduler.combo_generator.brute_force import BruteForceGenerator
from Core.Scheduler.dispatcher.sequential import SequentialDispatcher

COMBO_REG = {"bf": BruteForceGenerator}

DISP_REG = {"bf": SequentialDispatcher}

try:
    from Core.Scheduler.combo_generator.cpsat import CPSatComboGenerator
    from Core.Scheduler.dispatcher.sequential import CPSatDispatcher
    if CPSatComboGenerator:
        COMBO_REG["cp"] = CPSatComboGenerator
        DISP_REG["cp"] = CPSatDispatcher
except ImportError:
    print("ERROR")
    pass

