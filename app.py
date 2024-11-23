import os
import gradio as gr
import os.path as osp
from src.gradio_pipeline import GradioPipeline, GradioPipelineAnimal
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig

args = ArgumentConfig(
    gradio_temp_dir="gradio_temp",
    server_port=7860,
    share=True,
    server_name="127.0.0.1",
)

inference_cfg = InferenceConfig()
crop_cfg = CropConfig()

gradio_pipeline_human = GradioPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)
gradio_pipeline_animal = GradioPipelineAnimal(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)

def gpu_wrapped_execute_a2v(*args, **kwargs):
    if args[5] == "animal":
        return gradio_pipeline_animal.execute_a2v(*args, **kwargs)
    else:
        return gradio_pipeline_human.execute_a2v(*args, **kwargs)

with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")])) as demo:
    with gr.Row():
        with gr.Accordion(open=True, label="🖼️ Reference Image"):
            input_image = gr.Image(type="filepath", width=512, label="Reference Image")
        with gr.Accordion(open=True, label="🎵 Input Audio"):
            input_audio = gr.Audio(type="filepath", label="Input Audio")
        with gr.Accordion(open=True, label="🎬 Output Video",):
            output_video = gr.Video(autoplay=False, interactive=False, width=512)     
    with gr.Column():
        with gr.Accordion(open=True, label="Key Animation Options"):
            with gr.Row():
                animation_mode = gr.Radio(['human', 'animal'], value="human", label="Animation Mode") 
                flag_do_crop_input = gr.Checkbox(value=True, label="do crop (image)")
                cfg_scale = gr.Number(value=4.0, label="cfg_scale", minimum=0.0, maximum=10.0, step=0.5)
        with gr.Accordion(open=False, label="Optional Animation Options"):
            with gr.Row():
                driving_option_input = gr.Radio(['expression-friendly', 'pose-friendly'], value="expression-friendly", label="driving option")
                driving_multiplier = gr.Number(value=1.0, label="driving multiplier", minimum=0.0, maximum=2.0, step=0.02)
            with gr.Row():
                flag_normalize_lip = gr.Checkbox(value=True, label="normalize lip")
                flag_relative_motion = gr.Checkbox(value=True, label="relative motion")
                flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                flag_stitching_input = gr.Checkbox(value=True, label="stitching")
        with gr.Accordion(open=False, label="Optional Options for Reference Image"):
            with gr.Row():
                scale = gr.Number(value=2.3, label="image crop scale", minimum=1.8, maximum=4.0, step=0.05)
                vx_ratio = gr.Number(value=0.0, label="image crop x", minimum=-0.5, maximum=0.5, step=0.01)
                vy_ratio = gr.Number(value=-0.125, label="image crop y", minimum=-0.5, maximum=0.5, step=0.01)

    with gr.Row():
        process_button_generate = gr.Button("🚀 Generate", variant="primary")
    generation_func = gpu_wrapped_execute_a2v
    process_button_generate.click(
        fn=generation_func,
        inputs=[
            input_image,
            input_audio,
            flag_normalize_lip,
            flag_relative_motion,
            driving_multiplier,
            animation_mode,
            driving_option_input,
            flag_do_crop_input,
            scale,
            vx_ratio,
            vy_ratio,
            flag_stitching_input,
            flag_remap_input,
            cfg_scale,
        ],
        outputs=[
            output_video,
        ],
        show_progress=True
    )

demo.launch(
    server_port=args.server_port,
    share=args.share,
    server_name=args.server_name,
    inline=False
)
