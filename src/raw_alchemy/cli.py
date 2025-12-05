import click
from raw_alchemy import core, orchestrator

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
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
    "--custom-lensfun-db",
    "custom_lensfun_db_path",
    type=click.Path(exists=True),
    help="Path to a custom lensfun database XML file.",
)
@click.option(
    "--metering",
    default="hybrid",
    type=click.Choice(core.METERING_MODES, case_sensitive=False),
    help="Auto exposure metering mode: hybrid (default), average, center-weighted, highlight-safe.",
)
@click.option(
    "--jobs",
    type=int,
    default=4,
    help="Number of concurrent jobs for batch processing. Default is 4.",
)
def main(input_path, output_path, log_space, lut_path, exposure, lens_correct, custom_lensfun_db_path, metering, jobs):
    """
    Converts RAW image(s) to TIFF files through an ProPhoto-based pipeline.

    INPUT_PATH: Path to a single RAW file or a directory containing RAW files.
    OUTPUT_PATH: Path to the output TIFF file or a directory for batch processing.
    """
    try:
        orchestrator.process_path(
            input_path=input_path,
            output_path=output_path,
            log_space=log_space,
            lut_path=lut_path,
            exposure=exposure,
            lens_correct=lens_correct,
            custom_db_path=custom_lensfun_db_path,
            metering_mode=metering,
            jobs=jobs,
            logger_func=print, # Pass the print function for logging
        )
    except Exception as e:
        # The orchestrator will log specifics, but we can catch fatal errors here.
        raise click.ClickException(f"A critical error occurred: {e}")


if __name__ == "__main__":
    main()