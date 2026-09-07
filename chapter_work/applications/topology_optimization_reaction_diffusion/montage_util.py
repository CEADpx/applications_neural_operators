"""
Image-montage helpers, copied verbatim from the established convention
already repeated in survey_work/problems/{poisson,linear_elasticity,
hyperelasticity}/compare_nops/compareNeuralOperators.ipynb (each of those
notebooks defines the same functions locally rather than importing from
src/ -- this file follows that same repeated pattern for reaction-diffusion,
rather than inventing a new montage style).
"""
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image


def format_pct_error(err, precision=2, symbol="e"):
    """precision=2 matches the established compare_nops convention; pass a
    higher value (or use scientific notation via the 'e' format below) for
    correction-step errors, which can be small enough that 2 decimals round
    to 0.00% and hide the actual (nonzero) improvement. symbol: the LaTeX
    error symbol, e.g. "e" for state error or "\\varepsilon" for design/
    minimizer error, matching the notation in eq:optimizationDesignError /
    eq:optimizationStateError."""
    return rf"${symbol} = {100 * err:.{precision}f}\,\%$"


def format_pct_error_sci(err, precision=2, symbol="e"):
    """Scientific-notation variant for very small errors (e.g. after
    residual correction), where fixed-point precision would round to 0.00%."""
    return rf"${symbol} = {100 * err:.{precision}e}\,\%$"


def render_cell_title(text, width, color, fontsize=20):
    dpi = 150
    fig, ax = plt.subplots(figsize=(width / dpi, 0.35), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize, color=color)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_row_label(text, label_w, cell_h, fontsize=22, color="black"):
    dpi = 150
    fig, ax = plt.subplots(figsize=(label_w / dpi, cell_h / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5, 0.55, text,
        rotation=90, ha="center", va="center",
        fontsize=fontsize, color=color,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_image_montage(
    row_paths,
    save_path,
    pad=4,
    row_labels=None,
    row_label_colors=None,
    cell_titles=None,
    cell_title_colors=None,
    label_w=72,
    label_pad=4,
    label_fontsize=40,
    title_fontsize=28,
    title_x_shift=-60,
    title_y_shift=0,
):
    """Uniform a×a grid; each image centered at native size (no scaling)."""
    def load_rgb(path):
        im = Image.open(path)
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            return bg
        return im.convert("RGB")

    all_imgs = [load_rgb(p) for row in row_paths for p in row]
    a = max(max(im.size) for im in all_imgs)

    nrow = len(row_paths)
    ncol = len(row_paths[0])
    left_margin = (label_w + label_pad) if row_labels else 0
    montage = Image.new(
        "RGB",
        (left_margin + ncol * a + pad * (ncol - 1), nrow * a + pad * (nrow - 1)),
        (255, 255, 255),
    )

    if row_labels:
        if len(row_labels) != nrow:
            raise ValueError("row_labels must match number of rows")
        if row_label_colors is None:
            row_label_colors = ["black"] * nrow
        elif len(row_label_colors) != nrow:
            raise ValueError("row_label_colors must match number of rows")
        for r, text in enumerate(row_labels):
            label_im = render_row_label(
                text, label_w, a, fontsize=label_fontsize, color=row_label_colors[r],
            )
            montage.paste(label_im, (0, r * (a + pad)))

    for r, paths in enumerate(row_paths):
        for c, path in enumerate(paths):
            im = load_rgb(path)
            cell = Image.new("RGB", (a, a), (255, 255, 255))
            x_off = (a - im.width) // 2
            y_off = (a - im.height) // 2
            cell.paste(im, (x_off, y_off))

            if cell_titles is not None and cell_titles[r][c] is not None:
                color = "black"
                if cell_title_colors is not None and cell_title_colors[r][c] is not None:
                    color = cell_title_colors[r][c]
                title_im = render_cell_title(
                    cell_titles[r][c], im.width, color, fontsize=title_fontsize,
                )
                title_y = max(0, y_off - title_im.height - 2) + title_y_shift
                title_x = x_off + (im.width - title_im.width) // 2 + title_x_shift
                cell.paste(title_im, (title_x, title_y))

            montage.paste(cell, (left_margin + c * (a + pad), r * (a + pad)))

    montage.save(save_path)
