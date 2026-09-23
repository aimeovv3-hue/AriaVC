import os
import sys
import time
import subprocess
import shutil
import gradio as gr

# ==========================================
# GESTIÓN DE SERVICIOS EN SEGUNDO PLANO
# ==========================================
class BackgroundServices:
    def __init__(self):
        self.processes = {}

    def start_filebrowser(self, port=8080, root_dir="/kaggle/working"):
        if "filebrowser" in self.processes and self.processes["filebrowser"].poll() is None:
            return "Filebrowser ya está en ejecución."
        cmd = f"filebrowser -p {port} -r {root_dir} --noauth"
        try:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes["filebrowser"] = p
            return f"Filebrowser iniciado en puerto {port}"
        except Exception as e:
            return f"Error: {str(e)}"

    def start_tensorboard(self, logdir="/kaggle/working/logs", port=6006):
        if "tensorboard" in self.processes and self.processes["tensorboard"].poll() is None:
            return "Tensorboard ya está en ejecución."
        os.makedirs(logdir, exist_ok=True)
        cmd = f"tensorboard --logdir={logdir} --port={port} --bind_all"
        try:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes["tensorboard"] = p
            return f"Tensorboard iniciado en puerto {port}"
        except Exception as e:
            return f"Error: {str(e)}"

    def start_tunnel(self, service_port, tunnel_type="localtunnel"):
        key = f"tunnel_{service_port}"
        if key in self.processes and self.processes[key].poll() is None:
            return f"Túnel para puerto {service_port} ya activo."
        
        cmd = f"lt --port {service_port}" if tunnel_type == "localtunnel" else f"cloudflared tunnel --url http://localhost:{service_port}"
        try:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes[key] = p
            return f"Túnel ({tunnel_type}) lanzado para el puerto {service_port}."
        except Exception as e:
            return f"Error: {str(e)}"

    def stop_all(self):
        for name, p in self.processes.items():
            if p.poll() is None:
                p.terminate()
        self.processes.clear()
        return "Todos los servicios han sido detenidos."

services = BackgroundServices()

# ==========================================
# FUNCIONES DE PROCESAMIENTO (MOCKS BACKEND)
# ==========================================
def process_dataset(files, dataset_name, sr_target, slice_method, max_silence, normalize_lufs, denoise, pitch_aug, aug_range):
    if not files:
        gr.Warning("⚠️ No hay archivos de audio.")
        return "Error: Dataset vacío."
    target_dir = f"/kaggle/working/datasets/{dataset_name}"
    os.makedirs(target_dir, exist_ok=True)
    gr.Info(f"📁 Copiando {len(files)} archivos...")
    for f in files: shutil.copy(f.name, os.path.join(target_dir, os.path.basename(f.name)))
    gr.Info(f"✂️ Segmentando ({slice_method}) y Normalizando...")
    time.sleep(1)
    if pitch_aug: gr.Info("🎵 Aplicando Data Augmentation (Pitch Shift)...")
    return f"✅ Dataset procesado en {target_dir}."

def extract_features(dataset_name, embedder, f0_method, f0_min, f0_max, gpu_extract, cpu_threads):
    gr.Info(f"🔍 Extrayendo Embeddings con '{embedder}'...")
    time.sleep(1)
    gr.Info(f"🎼 Calculando Pitch con '{f0_method}'...")
    return "✅ Características extraídas correctamente."

def run_training_pipeline(model_name, sr_train, vocoder_arch, total_epochs, batch_size, lr_g, lr_d, precision, grad_accum, save_every, save_latest, save_small, cache_vram, use_pretrain, custom_g, custom_d, f0_guided, pitch_loss_w, progress=gr.Progress()):
    gr.Info(f"🚀 Iniciando entrenamiento: {model_name} | Vocoder: {vocoder_arch} | Precisión: {precision}")
    for epoch in progress.tqdm(range(1, total_epochs + 1), desc="Entrenando..."):
        time.sleep(0.01)
        if epoch % save_every == 0:
            gr.Info(f"💾 Checkpoint G/D guardado (Época {epoch})")
    if save_small: gr.Info("✂️ Guardando peso de inferencia optimizado (.pth)")
    return "🎉 Entrenamiento finalizado con éxito."

def build_index(model_name, index_algo, metric_choice, use_pca, pca_dim, nlist):
    gr.Info(f"📊 Entrenando índice {index_algo}...")
    time.sleep(1.5)
    return f"✅ Índice FAISS creado para {model_name}."

def run_inference(model_path, index_path, audio_input, pitch_shift, f0_method, index_rate, protect_consonants, rms_mix, resample_sr, envelop_ratio):
    if not audio_input:
        gr.Warning("Sube un audio primero.")
        return None, "Error: Falta audio input"
    gr.Info(f"🎙️ Convirtiendo audio usando {f0_method} con tono modificado {pitch_shift} semitonos...")
    time.sleep(2)
    # Devuelve el mismo audio de entrada como simulación de salida y un mensaje
    return audio_input, "✅ Inferencia completada."

# ==========================================
# INTERFAZ GRADIO
# ==========================================
theme = gr.themes.Soft(primary_hue="violet", secondary_hue="indigo")

with gr.Blocks(theme=theme, title="AriaVC Studio Pro") as aria_ui:
    gr.Markdown("# 🎵 AriaVC Studio Pro - Training & Inference")
    
    with gr.Tabs():
        
        # TAB 1: PREPROCESAMIENTO
        with gr.TabItem("🎙️ 1. Dataset"):
            with gr.Row():
                with gr.Column():
                    dataset_name = gr.Textbox(value="Cantante_Principal", label="Nombre del Dataset")
                    dataset_files = gr.File(file_count="multiple", label="Sube tus Audios (WAV/FLAC)", file_types=["audio"])
                    sr_target = gr.Radio(choices=["32k", "40k", "48k"], value="40k", label="Sample Rate")
                with gr.Column():
                    slice_method = gr.Dropdown(choices=["Silero-VAD", "Librosa-RMS"], value="Silero-VAD", label="Segmentación")
                    max_silence = gr.Slider(minimum=200, maximum=2000, value=800, label="Silencio Máximo (ms)")
                    normalize_lufs = gr.Slider(10, 30, 23, label="Normalización (-LUFS)")
                    # Denoise apagado por defecto: para canto, limpiar ruido suele destruir frecuencias agudas de la voz.
                    denoise = gr.Checkbox(value=False, label="Denoise (⚠️ Desactivar para canto de estudio)")
                    pitch_aug = gr.Checkbox(value=True, label="Data Augmentation (Pitch Shift)")
                    aug_range = gr.Slider(1, 6, 2, label="Rango de Semitonos (±N)")

            btn_process = gr.Button("⚙️ Procesar Dataset", variant="primary")
            out_process = gr.Textbox(label="Estado")
            btn_process.click(process_dataset, [dataset_files, dataset_name, sr_target, slice_method, max_silence, normalize_lufs, denoise, pitch_aug, aug_range], out_process)

        # TAB 2: EXTRACCIÓN
        with gr.TabItem("🧠 2. Extracción (F0 & Embeddings)"):
            with gr.Row():
                with gr.Column():
                    dataset_ref = gr.Textbox(value="Cantante_Principal", label="Nombre del Dataset")
                    # Whisper-large-v3 es la mejor opción actual para capturar detalles, seguido de ContentVec
                    embedder = gr.Dropdown(choices=["Whisper-large-v3", "ContentVec-500k", "Hubert-Soft", "XLSR-53"], value="Whisper-large-v3", label="Modelo Extractor (Embedder)")
                with gr.Column():
                    # FCPE es el estándar de oro actual para canto sin errores de tono.
                    f0_method = gr.Dropdown(choices=["FCPE", "RMVPE", "Crepe-Full", "Harvest", "Híbrido (FCPE+RMVPE)"], value="FCPE", label="Algoritmo Pitch (F0)")
                    f0_min = gr.Number(value=50, label="F0 Min (Hz)")
                    f0_max = gr.Number(value=1100, label="F0 Max (Hz)")

            with gr.Accordion("⚙️ Rendimiento", open=False):
                gpu_extract = gr.Checkbox(value=True, label="Aceleración CUDA")
                cpu_threads = gr.Slider(1, 16, 4, label="Hilos CPU")

            btn_extract = gr.Button("🔍 Extraer Características", variant="primary")
            out_extract = gr.Textbox(label="Estado")
            btn_extract.click(extract_features, [dataset_ref, embedder, f0_method, f0_min, f0_max, gpu_extract, cpu_threads], out_extract)

        # TAB 3: ENTRENAMIENTO
        with gr.TabItem("🚀 3. Entrenamiento"):
            with gr.Row():
                with gr.Column():
                    model_name = gr.Textbox(value="Mi_Modelo_Vocal", label="Nombre del Modelo")
                    # BigVGAN-v2 previene artefactos metálicos en agudos extremos.
                    vocoder_arch = gr.Dropdown(choices=["BigVGAN-v2", "HiFi-GAN", "RefineGAN"], value="BigVGAN-v2", label="Vocoder")
                    sr_train = gr.Radio(choices=["32k", "40k", "48k"], value="40k", label="Sample Rate")
                with gr.Column():
                    epochs = gr.Slider(10, 2000, 300, label="Épocas")
                    batch_size = gr.Slider(1, 64, 8, label="Batch Size (Bajar si da Error de Memoria)")
                    # bf16/fp16 son los recomendados. fp16 es más compatible universalmente.
                    precision = gr.Dropdown(choices=["fp16", "bf16", "tf32", "fp32"], value="fp16", label="Precisión")

            with gr.Accordion("🎛️ Hiperparámetros Avanzados", open=False):
                lr_g = gr.Textbox(value="0.0001", label="Learning Rate G")
                lr_d = gr.Textbox(value="0.0001", label="Learning Rate D")
                grad_accum = gr.Slider(1, 8, 1, label="Gradient Accumulation")
                f0_guided = gr.Checkbox(value=True, label="Pitch Guidance (Obligatorio para Canto)")
                pitch_loss_w = gr.Slider(0.1, 5.0, 1.0, label="Pitch Loss Weight")
                cache_vram = gr.Checkbox(value=False, label="Cachear a VRAM (Activar solo si tienes > 16GB VRAM)")

            with gr.Accordion("📦 Guardado y Pretrains (Recomendado)", open=True):
                save_every = gr.Slider(1, 100, 10, label="Guardar cada X Épocas")
                save_latest = gr.Checkbox(value=True, label="Guardar SOLO el último (Evita llenar disco)")
                save_small = gr.Checkbox(value=True, label="Extraer peso final liviano")
                use_pretrain = gr.Checkbox(value=True, label="Usar Pretrains Base")
                custom_g = gr.Textbox(placeholder="Ruta/URL Pretrain G...", label="Custom G")
                custom_d = gr.Textbox(placeholder="Ruta/URL Pretrain D...", label="Custom D")

            btn_train = gr.Button("🚀 Iniciar Entrenamiento", variant="primary")
            out_train = gr.Textbox(label="Consola")
            btn_train.click(run_training_pipeline, [model_name, sr_train, vocoder_arch, epochs, batch_size, lr_g, lr_d, precision, grad_accum, save_every, save_latest, save_small, cache_vram, use_pretrain, custom_g, custom_d, f0_guided, pitch_loss_w], out_train)

        # TAB 4: ÍNDICE FAISS
        with gr.TabItem("📊 4. Índice (FAISS)"):
            with gr.Row():
                with gr.Column():
                    faiss_model = gr.Textbox(value="Mi_Modelo_Vocal", label="Modelo")
                    index_algo = gr.Dropdown(choices=["IVFFlat", "Flat", "HNSW"], value="IVFFlat", label="Algoritmo")
                with gr.Column():
                    use_pca = gr.Checkbox(value=False, label="Usar PCA")
                    pca_dim = gr.Slider(64, 256, 128, label="Dimensiones PCA")
                    nlist = gr.Slider(1, 512, 256, label="Clusters K-Means (nlist)")
            
            btn_index = gr.Button("📊 Crear Índice", variant="primary")
            out_index = gr.Textbox(label="Estado")
            btn_index.click(build_index, [faiss_model, index_algo, gr.State("L2"), use_pca, pca_dim, nlist], out_index)

        # TAB 5: INFERENCIA (NUEVO)
        with gr.TabItem("🎧 5. Inferencia / Probar Modelo"):
            gr.Markdown("### Convierte un audio usando tu modelo entrenado")
            with gr.Row():
                with gr.Column(scale=1):
                    inf_model = gr.Textbox(placeholder="Ej: /kaggle/working/logs/Mi_Modelo/peso.pth", label="Ruta del Modelo (.pth)")
                    inf_index = gr.Textbox(placeholder="Ej: /kaggle/working/logs/Mi_Modelo/added.index", label="Ruta del Índice (.index)")
                    inf_audio = gr.Audio(type="filepath", label="Audio de Entrada (Voz a convertir)")
                
                with gr.Column(scale=1):
                    inf_pitch = gr.Slider(-24, 24, 0, step=1, label="Transposición (Semitonos)", info="Mujer a Hombre: -12 | Hombre a Mujer: +12")
                    inf_f0 = gr.Dropdown(choices=["FCPE", "RMVPE", "Crepe", "Harvest"], value="FCPE", label="Algoritmo Pitch Inferencia")
                    
                    with gr.Accordion("⚙️ Ajustes Avanzados de Calidad", open=True):
                        # Index Rate en 0.70 da un balance perfecto entre acento de la voz original y el timbre clonado.
                        inf_index_rate = gr.Slider(0.0, 1.0, 0.70, label="Feature Retrieval Rate (Fuerza del Índice)")
                        # Proteger consonantes es clave para que las eses ("s") y tes ("t") no suenen a robot.
                        inf_protect = gr.Slider(0.0, 0.5, 0.33, label="Proteger Consonantes Sordas", info="Menos valor = Más protección")
                        inf_rms = gr.Slider(0.0, 1.0, 0.25, label="RMS Mix Rate (Mezcla de Volumen)")
                        inf_env = gr.Slider(0.0, 1.0, 1.0, label="Volume Envelope Ratio")
                        inf_resample = gr.Slider(0, 48000, 0, step=1000, label="Resamplear Salida (0 = No Resamplear)")

            btn_infer = gr.Button("🎧 Generar Audio", variant="primary")
            out_infer_audio = gr.Audio(label="Resultado Final", interactive=False)
            out_infer_log = gr.Textbox(label="Log Inferencia")
            
            btn_infer.click(run_inference, [inf_model, inf_index, inf_audio, inf_pitch, inf_f0, inf_index_rate, inf_protect, inf_rms, inf_resample, inf_env], [out_infer_audio, out_infer_log])

        # TAB 6: SERVICIOS
        with gr.TabItem("🌐 6. Servicios y Túneles"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Filebrowser (Archivos)")
                    btn_fb = gr.Button("Abrir 8080")
                    out_fb = gr.Textbox(label="Log")
                    btn_fb.click(lambda: services.start_filebrowser(), outputs=out_fb)
                with gr.Column():
                    gr.Markdown("#### Tensorboard (Gráficas)")
                    btn_tb = gr.Button("Abrir 6006")
                    out_tb = gr.Textbox(label="Log")
                    btn_tb.click(lambda: services.start_tensorboard(), outputs=out_tb)
            with gr.Row():
                btn_tunnel = gr.Button("🔗 Generar Túnel Público")
                out_tunnel = gr.Textbox(label="Log Túnel")
                btn_tunnel.click(lambda: services.start_tunnel(8080), outputs=out_tunnel)
            btn_stop = gr.Button("🛑 Detener Todo", variant="stop")
            btn_stop.click(lambda: services.stop_all(), outputs=gr.Textbox(label="Estado General"))

if __name__ == "__main__":
    aria_ui.queue().launch(share=True, show_error=True)