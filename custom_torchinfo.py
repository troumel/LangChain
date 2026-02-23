"""Custom wrapper around torchinfo for displaying model summaries."""

from torchinfo import summary


def custom_summary(model, input_size=None, **kwargs):
    """
    Custom summary function that wraps torchinfo.summary.

    Args:
        model: PyTorch model to summarize
        input_size: Size of input tensor (e.g., (batch_size, channels, height, width))
        **kwargs: Additional arguments to pass to torchinfo.summary

    Returns:
        ModelStatistics object from torchinfo
    """
    default_kwargs = {
        "col_names": ["input_size", "output_size", "num_params", "trainable"],
        "verbose": 1,
    }
    default_kwargs.update(kwargs)

    if input_size is not None:
        return summary(model, input_size=input_size, **default_kwargs)
    else:
        return summary(model, **default_kwargs)
