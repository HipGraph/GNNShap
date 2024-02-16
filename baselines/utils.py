import numpy as np

def result2dict(node_id: int, scores: np.array, comp_time: float) -> dict:
    """Converts an explanation result to a dictionary

    Args:
        node_id (int): node id
        scores (np.array): importance scores
        comp_time (float): computation time

    Returns:
        dict: result as dictionary
    """
    return {'node_id': node_id, 'scores': scores, 'num_players': len(scores), 'time': comp_time}
