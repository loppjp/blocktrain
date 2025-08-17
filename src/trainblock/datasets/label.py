from dataclasses import dataclass


@dataclass
class Label:
    """
    A label class
    """

    exist: bool = False
    x: int = -1
    y: int = -1
    width: int = -1
    height: int = -1


def from_ir_json_labels(labels: dict, idx: int):
    """
    Load a label instance from a dictionary of labels and a given index

    The exist, x,y,width,hight schema is consistent with a provided IR dataset

    Args:
        labels: a dictionary of label annotations
        idx: the
    """
    return Label(
        **{
            "exist": labels["exist"][idx],
            **dict(zip(["x", "y", "width", "height"], labels["gt_rect"][idx])),
        }
    )
