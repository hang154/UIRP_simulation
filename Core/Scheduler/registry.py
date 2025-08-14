from Core.Scheduler.task_selector.fifo import FIFOTaskSelector
from Core.Scheduler.metric_evaluator.baseline import BaselineEvaluator
from Core.Scheduler.combo_generator.brute_force import BruteForceGenerator
from Core.Scheduler.combo_generator.greedy import GreedyComboGenerator
from Core.Scheduler.dispatcher.sequential import SequentialDispatcher

COMBO_REG = {"bf": BruteForceGenerator, "greedy": GreedyComboGenerator}

DISP_REG = {"bf": SequentialDispatcher, "greedy": SequentialDispatcher}

try:
    from Core.Scheduler.combo_generator.cpsat import CPSatComboGenerator
    from Core.Scheduler.combo_generator.hybrid_cp import HybridCPComboGenerator
    from Core.Scheduler.dispatcher.sequential import CPSatDispatcher
    if CPSatComboGenerator:
        COMBO_REG["cp"] = CPSatComboGenerator
        DISP_REG["cp"] = CPSatDispatcher
        COMBO_REG["hybrid_cp"] = HybridCPComboGenerator
        DISP_REG["hybrid_cp"] = CPSatDispatcher
except ImportError:
    print("ERROR")
    pass

