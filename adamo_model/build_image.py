from pathlib import Path
import subprocess
import tempfile

import modal


repository = Path(__file__).parents[1]
context = Path(tempfile.mkdtemp())
(context / "molmoact2").mkdir()
archive = subprocess.run(
    ["git", "-C", repository, "archive", "HEAD"], capture_output=True, check=True
).stdout
subprocess.run(["tar", "-x", "-C", context / "molmoact2"], input=archive, check=True)
subprocess.run(
    [
        "git",
        "clone",
        "-q",
        "--depth",
        "1",
        "https://github.com/adamo-tech/adamo-model-runtime.git",
        context / "adamo-model-runtime",
    ],
    check=True,
)
image = modal.Image.from_dockerfile(
    context / "molmoact2/adamo_model/Dockerfile", context_dir=context
)
app = modal.App.lookup("adamo-workers", create_if_missing=True)
built_image = image.build(app)
built_image.publish("molmoact2")
