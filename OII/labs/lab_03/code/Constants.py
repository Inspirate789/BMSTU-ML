class HelperConstants(object):
    """
    General helper constants
    """
    DIVIDER = "-----" * 10
    CARRIAGE_RETURN = "\n"
    N_EPOCHS_FOR_DISPLAY = 50
    RANDOM_SEED = 100

class AcoConstants(object):
    """
    ACO Constant specifications
    """    
    MAX_ITERATIONS = 150                   # Maximum Number of Iterations
    N_POP = 10                              # Population Size (Archive Size)
    N_ANTS = 40                             # Number of ants
    Q = 0.5                                 # Intensification Factor (Selection Pressure)
    ZETA = 1                                # Deviation-Distance Ratio
