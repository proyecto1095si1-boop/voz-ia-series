"""
Interfaz gráfica de VoxSync (Gradio).
Pensada para usuarios no técnicos: canales de YouTube, productoras, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import tempfile
import shutil

from voxsync import __version__, __product_name__
from voxsync.pipeline import run_pipeline


def _process(
    input_file,
    youtube_url: str,
    source_lang: str,
    target_lang: str,
    model_size: str,
    device: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    Ejecuta el pipeline y devuelve:
    (ruta_srt, ruta_txt, ruta_json, mensaje_estado)
    """
    # Determinar entrada
    if youtube_url and youtube_url.strip():
        input_path = youtube_url.strip()
    elif input_file is not None:
        input_path = input_file
    else:
        return None, None, None, "❌ Subí un video/audio o pegá una URL de YouTube."

    output_dir = Path(tempfile.mkdtemp(prefix="voxsync_"))

    try:
        result = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            source_lang=source_lang,
            target_lang=target_lang,
            model_size=model_size,
            device=device,
            keep_audio=False,
        )

        srt = result.get("srt_path")
        txt = result.get("txt_path")
        js = result.get("json_path")

        msg = (
            f"✅ Procesamiento completado\n"
            f"Segmentos: {result.get('segments_count', '?')}\n"
            f"Salida: {result.get('output_dir')}"
        )
        return srt, txt, js, msg

    except Exception as e:
        return None, None, None, f"❌ Error: {e}"


def launch(share: bool = False, server_port: int = 7860):
    """Lanza la interfaz Gradio."""
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError(
            "Gradio no está instalado. Ejecutá:\n  pip install gradio"
        ) from e

    with gr.Blocks(
        title=f"{__product_name__} v{__version__}",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 960px !important; }",
    ) as demo:
        gr.Markdown(
            f"""
            # {__product_name__}
            **Transcripción · Diarización · Traducción · 100% local**

            Subí un video/audio o pegá un enlace de YouTube.
            VoxSync transcribe, separa hablantes, detecta género/edad y traduce.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Video o audio local",
                    file_types=["video", "audio"],
                    type="filepath",
                )
                youtube_url = gr.Textbox(
                    label="O URL de YouTube",
                    placeholder="https://www.youtube.com/watch?v=...",
                )

                with gr.Row():
                    source_lang = gr.Dropdown(
                        choices=["en", "es", "fr", "pt", "de", "it", "ja", "zh", "ko", "ru"],
                        value="en",
                        label="Idioma origen",
                    )
                    target_lang = gr.Dropdown(
                        choices=["es", "en", "fr", "pt", "de", "it", "ja", "zh", "ko", "ru"],
                        value="es",
                        label="Idioma destino",
                    )

                with gr.Row():
                    model_size = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        value="large-v3",
                        label="Modelo Whisper",
                        info="large-v3 = máxima calidad (más lento)",
                    )
                    device = gr.Dropdown(
                        choices=["auto", "cuda", "cpu"],
                        value="auto",
                        label="Dispositivo",
                    )

                btn = gr.Button("Procesar", variant="primary", size="lg")

            with gr.Column(scale=1):
                status = gr.Textbox(label="Estado", lines=4, interactive=False)
                srt_out = gr.File(label="Subtítulos SRT")
                txt_out = gr.File(label="Guion legible (TXT)")
                json_out = gr.File(label="Datos completos (JSON)")

        btn.click(
            fn=_process,
            inputs=[input_file, youtube_url, source_lang, target_lang, model_size, device],
            outputs=[srt_out, txt_out, json_out, status],
        )

        gr.Markdown(
            """
            ---
            **Tips**
            - Primera ejecución: se descargan los modelos (puede tardar).
            - GPU (CUDA) acelera mucho la transcripción.
            - Para series enteras usá la CLI: `voxsync batch ./carpeta`
            """
        )

    demo.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch()
