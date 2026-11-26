"""Export the publication rendering of a Matplotlib figure at 300 dpi."""

from io import BytesIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def save_png(figure, destination):
    with BytesIO() as vector, TemporaryDirectory(dir=destination.parent) as directory:
        figure.savefig(vector, format="pdf", metadata={"CreationDate": None, "ModDate": None})
        prefix = Path(directory) / "figure"
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-cropbox", "-r", "300", "-", prefix],
                       input=vector.getvalue(), check=True)
        prefix.with_suffix(".png").replace(destination)
