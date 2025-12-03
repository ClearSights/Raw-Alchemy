import click
import os
import glob
from . import core

# 支持的 RAW 文件扩展名列表 (小写)
SUPPORTED_RAW_EXTENSIONS = [
    '.dng', '.cr2', '.cr3', '.nef', '.arw', '.rw2', '.raf', '.orf', '.pef', '.srw'
]

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
    "--metering",
    default="hybrid",
    type=click.Choice(core.METERING_MODES, case_sensitive=False),
    help="Auto exposure metering mode: hybrid (default), average, center-weighted, highlight-safe.",
)
def main(input_path, output_path, log_space, lut_path, exposure, lens_correct, metering):
    """
    Converts RAW image(s) to TIFF files through an ACES-based pipeline.

    INPUT_PATH: Path to a single RAW file or a directory containing RAW files.
    OUTPUT_PATH: Path to the output TIFF file or a directory for batch processing.
    """
    # 检查输入路径是文件还是目录
    if os.path.isdir(input_path):
        # --- 批量处理 ---
        print(f"🎬 Starting batch processing for directory: {input_path}")
        
        # 确保输出路径是一个目录，如果不存在则创建
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"  📁 Created output directory: {output_path}")
        elif not os.path.isdir(output_path):
            raise click.UsageError("For batch processing, OUTPUT_PATH must be a directory.")

        # 查找所有支持的 RAW 文件
        raw_files = []
        for ext in SUPPORTED_RAW_EXTENSIONS:
            raw_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            raw_files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        
        if not raw_files:
            print("  ⚠️ No supported RAW files found in the input directory.")
            return

        print(f"  🔍 Found {len(raw_files)} RAW files to process.")

        for raw_path in raw_files:
            # 构建输出文件名
            base_name = os.path.basename(raw_path)
            file_name, _ = os.path.splitext(base_name)
            output_tiff = os.path.join(output_path, f"{file_name}.tif")
            
            try:
                core.process_image(
                    raw_path=raw_path,
                    output_path=output_tiff,
                    log_space=log_space,
                    lut_path=lut_path,
                    exposure=exposure,
                    lens_correct=lens_correct,
                    metering_mode=metering
                )
            except Exception as e:
                print(f"  ❌ Error processing {raw_path}: {e}")
                continue # 继续处理下一个文件

        print("\n🎉 Batch processing complete.")

    else:
        # --- 单文件处理 ---
        # 如果输出路径是目录，则在其中创建文件
        final_output_path = output_path
        if os.path.isdir(output_path):
            base_name = os.path.basename(input_path)
            file_name, _ = os.path.splitext(base_name)
            final_output_path = os.path.join(output_path, f"{file_name}.tif")

        core.process_image(
            raw_path=input_path,
            output_path=final_output_path,
            log_space=log_space,
            lut_path=lut_path,
            exposure=exposure,
            lens_correct=lens_correct,
            metering_mode=metering
        )


if __name__ == "__main__":
    main()