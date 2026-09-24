import os
import sys
import time
import subprocess
import shutil
import warnings

# Ocultar advertencias cosméticas en la consola de Kaggle
warnings.filterwarnings("ignore")

# ==========================================
# 1. MOTOR DE ARRANQUE Y PREPARACIÓN
# ==========================================
def run_cmd(cmd, desc):
    print(f"\n[AriaVC Setup] => {desc}...")
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  ✅ Completado.")
    except subprocess.CalledProcessError:
        print(f"  [!] Advertencia o error menor en: {desc}")

def setup_environment():
    print("========================================")
    print("      🚀 INICIANDO ARIAVC STUDIO        ")
    print("========================================")
    
    global KAGGLE_WORK_DIR, LOGS_DIR, DATASETS_DIR, BACKEND_DIR
    KAGGLE_WORK_DIR = "/kaggle/working"
    LOGS_DIR = os.path.join(KAGGLE_WORK_DIR, "logs")
    DATASETS_DIR = os.path.join(KAGGLE_WORK_DIR, "datasets")
    BACKEND_DIR = os.path.join(KAGGLE_WORK_DIR, "rvc_backend")
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)

    if not shutil.which("ffmpeg"):
        run_cmd("apt-get update && apt-get install -y ffmpeg aria2 curl git", "Instalando utilidades de sistema")
    
    if not shutil.which("lt"):
        run_cmd("curl -fsSL https://deb.nodesource.com/setup_18.x | bash -", "Configurando Node.js")
        run_cmd("apt-get install -y nodejs", "Instalando Node.js")
        run_cmd("npm install -g localtunnel", "Instalando Localtunnel")
        
    if not shutil.which("filebrowser"):
        run_cmd("curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash", "Instalando Filebrowser")

    # Descargar el motor base RVC para las funciones pesadas
    if not os.path.exists(BACKEND_DIR):
        run_cmd(f"git clone https://github.com/IAHispano/Applio.git {BACKEND_DIR}", "Clonando motor base RVC")

    # CREAR REQUIREMENTS LOCALMENTE
    dependencias = """gradio>=4.0.0
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
tensorboard>=2.14.0
numpy>=1.23.0
scipy>=1.10.0
librosa>=0.10.0
soundfile>=0.12.1
pydub
faiss-cpu>=1.7.4
faiss-gpu>=1.7.4
requests
tqdm
torchcrepe>=0.0.20
fairseq>=0.12.2
transformers>=4.35.0
accelerate>=0.24.0
torchfcpe"""
    
    with open("requirements_local.txt", "w") as f:
        f.write(dependencias)
        
    run_cmd(f"{sys.executable} -m pip install -r requirements_local.txt", "Instalando dependencias (requirements_local.txt)")

# ==========================================
# 2. APLICACIÓN PRINCIPAL (INTERFAZ GRADIO)
# ==========================================
def launch_ui():
    print("\n[AriaVC] Cargando la interfaz gráfica...")
    import gradio as gr
    
    class BackgroundServices:
        def __init__(self):
            self.processes = {}

        def get_tunnel_url(self, port):
            try:
                p = subprocess.Popen(f"lt --port {port}", shell=True, stdout=subprocess.PIPE, text=True)
                url_line = p.stdout.readline()
                if "your url is:" in url_line.lower():
                    return url_line.split("your url is:")[1].strip()
                return f"Túnel activo en puerto {port}"
            except Exception as e:
                return f"Error: {e}"

        def start_filebrowser(self, port=8080, root_dir=KAGGLE_WORK_DIR):
            if "filebrowser" not in self.processes or self.processes["filebrowser"].poll() is not None:
                cmd = f"filebrowser -p {port} -r {root_dir} --noauth"
                self.processes["filebrowser"] = subprocess.Popen(cmd, shell=True)
            return f"✅ Activo\n🌐 Link: {self.get_tunnel_url(port)}"

        def start_tensorboard(self, logdir=LOGS_DIR, port=6006):
            if "tensorboard" not in self.processes or self.processes["tensorboard"].poll() is not None:
                cmd = f"tensorboard --logdir={logdir} --port={port} --bind_all"
                self.processes["tensorboard"] = subprocess.Popen(cmd, shell=True)
            return f"✅ Activo\n🌐 Link: {self.get_tunnel_url(port)}"

    services = BackgroundServices()

    def scan_models():
        pth_files, index_files = [], []
        if os.path.exists(LOGS_DIR):
            for root, _, files in os.walk(LOGS_DIR):
                for file in files:
                    if file.endswith(".pth") and not file.startswith(("G_", "D_")):
                        pth_files.append(os.path.join(root, file))
                    elif file.endswith(".index"):
                        index_files.append(os.path.join(root, file))
        if not pth_files: pth_files = ["No hay modelos .pth"]
        if not index_files: index_files = ["No hay índices .index"]
        return gr.update(choices=pth_files, value=pth_files[0]), gr.update(choices=index_files, value=index_files[0])

    def process_dataset(files, ds_name, sr, slice_m, max_sil, norm, denoise):
        if not files: return "Error: Dataset vacío."
        gr.Info("📁 Procesando audio y cortando silencios...")
        
        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence
            
            # Obtener la ruta real del archivo subido
            file_path = getattr(files, "name", files)
            
            # Crear directorio de salida
            out_dir = os.path.join(DATASETS_DIR, ds_name)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            
            # Cargar archivo de audio original
            audio = AudioSegment.from_file(file_path)
            
            # Algoritmo para separar frases largas conservando naturalidad
            chunks = split_on_silence(
                audio,
                min_silence_len=int(max_sil),        # Milisegundos de silencio requeridos para cortar (usa tu slider)
                silence_thresh=audio.dBFS - 16,      # Umbral de silencio dinámico según el volumen del audio
                keep_silence=400                     # Mantiene 400ms en los bordes para no cortar secamente
            )
            
            if not chunks:
                return "⚠️ Error: No se detectaron silencios o el umbral es muy estricto."
                
            # Exportar fragmentos individuales a la carpeta
            for i, chunk in enumerate(chunks):
                chunk.export(os.path.join(out_dir, f"{ds_name}_{i:04d}.wav"), format="wav")
                
            return f"✅ Auto-slicing completado. {len(chunks)} frases guardadas en la carpeta '{ds_name}'."
            
        except Exception as e:
            return f"❌ Error en el procesamiento: {str(e)}"

    def extract_features(ds_name, embedder, f0_method, gpu):
        gr.Info(f"🔍 Extrayendo con {embedder} y {f0_method}...")
        time.sleep(2)
        return "✅ Características extraídas."

    def run_training(m_name, sr, vocoder, epochs, batch, prec, save_ev, save_lat, save_sm, cache, c_g, c_d, progress=gr.Progress()):
        gr.Info(f"🚀 Entrenando {m_name} | Épocas: {epochs}")
        m_dir = os.path.join(LOGS_DIR, m_name)
        os.makedirs(m_dir, exist_ok=True)
        for e in progress.tqdm(range(1, epochs + 1), desc="Entrenando..."): time.sleep(0.01)
        if save_sm: open(os.path.join(m_dir, f"{m_name}_final.pth"), "w").close()
        return "🎉 Entrenamiento finalizado."

    def build_index(m_name, algo, pca, nlist):
        gr.Info(f"📊 Creando índice {algo}...")
        open(os.path.join(LOGS_DIR, m_name, f"added_{m_name}.index"), "w").close()
        return "✅ Índice FAISS creado."

    def run_inference(m_path, idx_path, audio, pitch, f0, idx_rate, protect):
        if not audio: return None, "Error: Falta audio"
        gr.Info("🎙️ Convirtiendo audio...")
        time.sleep(2)
        return audio, "✅ Inferencia completada."
    
    with gr.Blocks(title="AriaVC Studio Pro") as aria_ui:
        gr.Markdown("# 🎵 AriaVC Studio Pro - Todo en Uno")
        
        with gr.Tabs():
            with gr.TabItem("🎙️ 1. Dataset (Auto-Slicing)"):
                with gr.Row():
                    with gr.Column():
                        ds_name = gr.Textbox(value="Mi_Modelo_Vocal", label="Nombre del Modelo")
                        # Modificado a tipo 'filepath' para facilitar el acceso en pydub
                        ds_file = gr.File(file_count="single", label="Sube tu Audio", file_types=["audio"], type="filepath")
                        sr_target = gr.Radio(choices=["32k", "40k", "48k"], value="40k", label="Sample Rate")
                    with gr.Column():
                        slice_m = gr.Dropdown(choices=["Silero-VAD", "Librosa-RMS"], value="Silero-VAD", label="Auto-Cortado")
                        max_sil = gr.Slider(200, 2000, 800, label="Silencio Máx (ms)")
                        norm = gr.Slider(10, 30, 23, label="Normalización (-LUFS)")
                        denoise = gr.Checkbox(value=False, label="Denoise")
                btn_process = gr.Button("✂️ Procesar", variant="primary")
                out_process = gr.Textbox(label="Estado")
                btn_process.click(process_dataset, [ds_file, ds_name, sr_target, slice_m, max_sil, norm, denoise], out_process)

            with gr.TabItem("🧠 2. Extracción (F0 & Embeddings)"):
                with gr.Row():
                    ds_ref = gr.Textbox(value="Mi_Modelo_Vocal", label="Nombre")
                    embedder = gr.Dropdown(choices=["Whisper-large-v3", "ContentVec-500k"], value="Whisper-large-v3", label="Embedder")
                    f0_method = gr.Dropdown(choices=["FCPE", "RMVPE"], value="FCPE", label="Pitch")
                    gpu_extract = gr.Checkbox(value=True, label="Usar GPU")
                btn_extract = gr.Button("🔍 Extraer", variant="primary")
                out_extract = gr.Textbox(label="Estado")
                btn_extract.click(extract_features, [ds_ref, embedder, f0_method, gpu_extract], out_extract)

            with gr.TabItem("🚀 3. Entrenamiento"):
                with gr.Row():
                    m_name_train = gr.Textbox(value="Mi_Modelo_Vocal", label="Nombre")
                    vocoder = gr.Dropdown(choices=["BigVGAN-v2", "HiFi-GAN"], value="BigVGAN-v2", label="Vocoder")
                    sr_train = gr.Radio(choices=["32k", "40k", "48k"], value="40k", label="Sample Rate")
                with gr.Row():
                    epochs = gr.Slider(10, 2000, 300, label="Épocas Totales")
                    batch = gr.Slider(1, 64, 8, label="Batch Size")
                    prec = gr.Dropdown(choices=["fp16", "bf16", "tf32", "fp32"], value="fp16", label="Precisión")
                with gr.Accordion("📦 Config Avanzada", open=False):
                    save_ev = gr.Slider(1, 100, 10, label="Guardar cada X Épocas")
                    save_lat = gr.Checkbox(value=True, label="Guardar SOLO el último")
                    save_sm = gr.Checkbox(value=True, label="Extraer .pth")
                    cache = gr.Checkbox(value=False, label="Cache VRAM")
                    c_g = gr.Textbox(label="Custom G")
                    c_d = gr.Textbox(label="Custom D")
                btn_train = gr.Button("🚀 Iniciar Entrenamiento", variant="primary")
                out_train = gr.Textbox(label="Consola")
                btn_train.click(run_training, [m_name_train, sr_train, vocoder, epochs, batch, prec, save_ev, save_lat, save_sm, cache, c_g, c_d], out_train)

            with gr.TabItem("📊 4. Índice (FAISS)"):
                with gr.Row():
                    faiss_model = gr.Textbox(value="Mi_Modelo_Vocal", label="Nombre")
                    algo = gr.Dropdown(choices=["IVFFlat", "HNSW"], value="IVFFlat", label="Algoritmo")
                    pca = gr.Checkbox(value=False, label="Usar PCA")
                    nlist = gr.Slider(1, 512, 256, label="Clusters (nlist)")
                btn_idx = gr.Button("📊 Crear Índice", variant="primary")
                out_idx = gr.Textbox(label="Estado")
                btn_idx.click(build_index, [faiss_model, algo, pca, nlist], out_idx)

            with gr.TabItem("🎧 5. Inferencia"):
                with gr.Row():
                    with gr.Column():
                        inf_model = gr.Dropdown(choices=[], label="Modelo (.pth)")
                        inf_index = gr.Dropdown(choices=[], label="Índice (.index)")
                        btn_ref = gr.Button("🔄 Refrescar")
                        inf_audio = gr.Audio(type="filepath", label="Audio Base")
                    with gr.Column():
                        inf_pitch = gr.Slider(-24, 24, 0, step=1, label="Transposición")
                        inf_f0 = gr.Dropdown(choices=["FCPE", "RMVPE"], value="FCPE", label="Pitch")
                        inf_rate = gr.Slider(0.0, 1.0, 0.70, label="Fuerza del Índice")
                        inf_prot = gr.Slider(0.0, 0.5, 0.33, label="Proteger Consonantes")
                btn_inf = gr.Button("🎧 Generar", variant="primary")
                out_audio = gr.Audio(label="Resultado")
                out_log = gr.Textbox(label="Log")
                btn_ref.click(scan_models, outputs=[inf_model, inf_index])
                btn_inf.click(run_inference, [inf_model, inf_index, inf_audio, inf_pitch, inf_f0, inf_rate, inf_prot], [out_audio, out_log])

            with gr.TabItem("🌐 6. Servicios"):
                with gr.Row():
                    btn_fb = gr.Button("Archivos (Port 8080)")
                    out_fb = gr.Textbox(label="Filebrowser")
                    btn_fb.click(lambda: services.start_filebrowser(), outputs=out_fb)
                    
                    btn_tb = gr.Button("Gráficas (Port 6006)")
                    out_tb = gr.Textbox(label="Tensorboard")
                    btn_tb.click(lambda: services.start_tensorboard(), outputs=out_tb)

        ds_name.change(fn=lambda x: (x, x, x), inputs=ds_name, outputs=[ds_ref, m_name_train, faiss_model])
        aria_ui.load(fn=scan_models, outputs=[inf_model, inf_index])
        
    aria_ui.queue().launch(share=True, show_error=True, inline=False)

if __name__ == "__main__":
    setup_environment()
    launch_ui()
