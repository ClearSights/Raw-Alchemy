import click
from . import core

@click.command()
@click.argument("input_raw", type=click.Path(exists=True))
@click.argument("output_tiff", type=click.Path())
@click.option(
    "--log-space",
    required=True,
    type=click.Choice(list(core.LOG_TO_WORKING_SPACE.keys()), case_sensitive=False),
    help="The log space to convert to.",
)
@click.option(
    "--lut",
    "lut_path",
    type=click.Path(exists=True),
    help="Path to a .cube LUT file to apply.",
)
@click.option(
    "--exposure",
    type=float,
    default=None,
    help="Manual exposure adjustment in stops (e.g., -0.5, 1.0). Overrides all auto exposure.",
)
@click.option(
    "--lens-correct",
    default=True,
    help="Enable or disable lens distortion correction. Enabled by default.",
)
@click.option(
    "--metering",
    default="hybrid",
    type=click.Choice(core.METERING_MODES, case_sensitive=False),
    help="Auto exposure metering mode: hybrid (default), average, center-weighted, highlight-safe.",
)
def main(input_raw, output_tiff, log_space, lut_path, exposure, lens_correct, metering):
    """
    Converts a RAW image to a TIFF file through an ACES-based pipeline.
    """
    core.process_image(
        raw_path=input_raw,
        output_path=output_tiff,
        log_space=log_space,
        lut_path=lut_path,
        exposure=exposure,
        lens_correct=lens_correct,
        metering_mode=metering
    )


if __name__ == "__main__":
    main()