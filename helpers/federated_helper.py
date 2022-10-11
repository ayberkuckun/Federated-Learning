import tensorflow_federated as tff


def format_size(size):
    """A helper function for creating a human-readable size."""
    size = float(size)
    for unit in ['bit', 'Kibit', 'Mibit', 'Gibit']:
        if size < 1024.0:
            return "{size:3.2f}{unit}".format(size=size, unit=unit)
        size /= 1024.0
    return "{size:.2f}{unit}".format(size=size, unit='TiB')


def set_sizing_environment():
    """Creates an environment that contains sizing information."""
    # Creates a sizing executor factory to output communication cost
    # after the training finishes. Note that sizing executor only provides an
    # estimate (not exact) of communication cost, and doesn't capture cases like
    # compression of over-the-wire representations. However, it's perfect for
    # demonstrating the effect of compression in this tutorial.
    sizing_factory = tff.framework.sizing_executor_factory()

    # TFF has a modular runtime you can configure yourself for various
    # environments and purposes, and this example just shows how to configure one
    # part of it to report the size of things.
    context = tff.framework.ExecutionContext(executor_fn=sizing_factory)
    tff.framework.set_default_context(context)

    return sizing_factory
