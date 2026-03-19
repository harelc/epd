"""Gradio demo for Edge-Preserving Multi-Scale Tone Manipulation."""

import gradio as gr
import numpy as np

from epd.decompose import PyramidMode
from epd.manipulate import multi_scale_tone_manipulate
from epd.smoothing import SmoothingMethod

METHOD_DESCRIPTIONS = {
    "wls": (
        "**WLS** (Farbman et al. 2008): Solves a weighted least-squares system. "
        "Best edge preservation, slowest.\n\n"
        "**User controls:** α (gradient sensitivity, 1.2–2.0), λ (smoothing strength). "
        "Both α and λ are interpolated per level between the fine/coarse slider values.\n\n"
        "**How λ works:** The system `(I + λ·L_g) u = g` is solved directly — higher λ = smoother result. "
        "Doubling λ roughly halves the frequency cutoff (Eq. 10 in the paper)."
    ),
    "bilateral": (
        "**Bilateral Filter** (Tomasi & Manduchi 1998): Weighted average using spatial + range kernels. "
        "Fast but may produce halos at coarse scales.\n\n"
        "**User controls:** λ fine/coarse (controls spatial support σ_s).\n\n"
        "**Auto-adapted:** σ_s = λ × 30 × √(megapixels), so larger images get proportionally larger kernels. "
        "σ_r (range sensitivity) = 10% of image value range × (0.5 + λ) — adapts to image contrast. "
        "α sliders have no effect."
    ),
    "guided": (
        "**Guided Filter** (He et al. 2010): O(N) box-filter based. Very fast, good edge preservation.\n\n"
        "**User controls:** λ fine/coarse (controls filter radius).\n\n"
        "**Auto-adapted:** radius = λ × 40 × √(megapixels), so larger images get proportionally larger radii. "
        "ε (regularization) = 1% of image variance × (0.5 + 5λ) — adapts to image contrast; "
        "higher ε smooths more edges, lower ε preserves more. "
        "α sliders have no effect."
    ),
}


def process(
    image: np.ndarray,
    method: str,
    pyramid_mode: str,
    exposure: float,
    base_boost: float,
    fine_boost: float,
    medium_boost: float,
    coarse_boost: float,
    alpha_fine: float,
    alpha_coarse: float,
    lam_fine: float,
    lam_coarse: float,
) -> np.ndarray | None:
    if image is None:
        return None

    img = image.astype(np.float64) / 255.0

    result = multi_scale_tone_manipulate(
        img,
        exposure=exposure,
        base_boost=base_boost,
        detail_boosts=[fine_boost, medium_boost, coarse_boost],
        num_levels=3,
        alpha_fine=alpha_fine,
        alpha_coarse=alpha_coarse,
        lam_fine=lam_fine,
        lam_coarse=lam_coarse,
        method=SmoothingMethod(method),
        mode=PyramidMode(pyramid_mode),
    )

    return (result * 255).astype(np.uint8)


def update_method_info(method: str) -> str:
    return METHOD_DESCRIPTIONS.get(method, "")


with gr.Blocks(title="Edge-Preserving Multi-Scale Tone Manipulation") as demo:
    gr.Markdown("# Edge-Preserving Multi-Scale Tone Manipulation")
    gr.Markdown(
        "Based on [Farbman et al. 2008](https://doi.org/10.1145/1360612.1360666). "
        "Upload an image, then adjust sliders — output updates automatically."
    )

    # Top row: images side by side
    with gr.Row():
        input_image = gr.Image(label="Input Image", type="numpy")
        output_image = gr.Image(label="Result", type="numpy")

    # Controls below images
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Tone Manipulation")
            exposure = gr.Slider(0.2, 3.0, value=1.0, step=0.05, label="Exposure (η)")
            base_boost = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Base Contrast (δ₀)")
            fine_boost = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Fine Detail Boost (δ₁)")
            medium_boost = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Medium Detail Boost (δ₂)")
            coarse_boost = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Coarse Detail Boost (δ₃)")

        with gr.Column():
            gr.Markdown("### Method & Filter Parameters")
            with gr.Row():
                method = gr.Radio(
                    choices=["wls", "bilateral", "guided"],
                    value="wls",
                    label="Smoothing Method",
                )
                pyramid_mode = gr.Radio(
                    choices=["non_iterative", "iterative"],
                    value="non_iterative",
                    label="Pyramid Mode",
                    info="Non-iterative: each level from original. Iterative: each level from previous.",
                )
            with gr.Row():
                alpha_fine = gr.Slider(0.5, 3.0, value=1.2, step=0.1, label="α fine")
                alpha_coarse = gr.Slider(0.5, 3.0, value=1.4, step=0.1, label="α coarse")
            with gr.Row():
                lam_fine = gr.Slider(0.01, 2.0, value=0.1, step=0.01, label="λ fine")
                lam_coarse = gr.Slider(0.1, 10.0, value=0.4, step=0.1, label="λ coarse")
            method_info = gr.Markdown(value=METHOD_DESCRIPTIONS["wls"])

    # All inputs that trigger auto-processing
    all_inputs = [
        input_image,
        method,
        pyramid_mode,
        exposure,
        base_boost,
        fine_boost,
        medium_boost,
        coarse_boost,
        alpha_fine,
        alpha_coarse,
        lam_fine,
        lam_coarse,
    ]

    # Auto-update on every change
    for component in all_inputs:
        component.change(fn=process, inputs=all_inputs, outputs=output_image)

    method.change(fn=update_method_info, inputs=[method], outputs=[method_info])

if __name__ == "__main__":
    demo.launch()
