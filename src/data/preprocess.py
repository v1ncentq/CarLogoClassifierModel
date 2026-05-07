from pathlib import Path
from PIL import Image


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path)


def convert_to_grayscale(pil_image: Image.Image) -> Image.Image:
    return pil_image.convert("L")


def resize_image(
    pil_image: Image.Image,
    image_size: tuple[int, int]
) -> Image.Image:
    return pil_image.resize(image_size)


def process_and_save_image(
    input_path: str,
    output_path: str,
    image_size: tuple[int, int],
    grayscale: bool = True
) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pil_image = load_image(str(input_file))

    if grayscale:
        pil_image = convert_to_grayscale(pil_image)

    processed_image = resize_image(pil_image, image_size)
    processed_image.save(str(output_file))