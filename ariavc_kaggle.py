import os
import sys
import time
import subprocess
import shutil
import warnings

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
        print(f"  [!] Advertencia en: {desc}")

def setup_environment():
    print("========================================")
    print("      🚀 INICIANDO ARIAVC STUDIO V6      ")
    print("========================================")
    
    global KAGGLE_WORK_DIR, LOGS_DIR, DATASETS_DIR, BACKEND_DIR
    KAGGLE_WORK_DIR = "/kaggle/working"
    LOGS_DIR = os.path.join(KAGGLE_WORK_DIR, "logs")
    DATASETS_DIR = os.path.join(KAGGLE_WORK_DIR, "datasets")
    BACKEND_DIR = os.path.join(KAGGLE_WORK_DIR, "rvc_backend")
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)

    if not shutil.which("ffmpeg"):
        run_cmd("apt-get update && apt-get install -y ffmpeg aria2 curl git", "Instalando utilidades del sistema")
    
    if not shutil.which("lt"):
        run_cmd("curl -fsSL https://deb.nodesource.com/setup_18.x | bash -", "Configurando Node.js")
        run_cmd("apt-get install -y nodejs && npm install -g localtunnel", "Instalando Localtunnel")
        
    if not shutil.which("filebrowser"):
        run_cmd("curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash", "Instalando Filebrowser")

    if not os.path.exists(BACKEND_DIR):
        run_cmd(f"git clone https://github.com/IAHispano/Applio.git {BACKEND_DIR}", "Clonando motor base RVC")

    # INSTALACIÓN MASIVA Y OFICIAL: Descarga todo lo que pide Applio por defecto
    run_cmd(f"{sys.executable} -m pip install -r {BACKEND_DIR}/requirements.txt", "Instalando dependencias COMPLETAS oficiales de Applio")
    
    # Instalamos solo lo extra necesario para nuestra interfaz
    dependencias_ui = "gradio>=4.0.0\npydub\ntensorboardX\n"
    with open("requirements_ui.txt", "w") as f: f.write(dependencias_ui)
    run_cmd(f"{sys.executable} -m pip install -q -r requirements_ui.txt", "Instalando complementos de la Interfaz Custom")

# ==========================================
# 2. CONSOLA EN TIEMPO REAL (ESPEJO KAGGLE + UI)
# ==========================================
def stream_cmd_realtime(cmd):
    msg_inicio = f"\n[AriaVC Backend] 🚀 Ejecutando:\n{cmd}\n"
    print(msg_inicio)
    sys.stdout.flush() 
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    log_output = msg_inicio + "\n"
    yield log_output
    
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush() 
        log_output += line
        yield log_output 
        
    process.stdout.close()
    process.wait()
    msg_fin = f"\n✅ Proceso finalizado (Código: {process.returncode})\n"
    sys.stdout.write(msg_fin)
    sys.stdout.flush()
    log_output += msg_fin
    yield log_output

# ==========================================
# 3. INTERFAZ GRÁFICA GRADIO
# ==========================================
def launch_ui():
    import gradio as gr
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    
    class BackgroundServices:
        def __init__(self): 
            self.processes = {}

        def start_service(self, name, cmd_service, port):
            if name not in self.processes or self.processes[name].poll() is not None:
                self.processes[name] = subprocess.Popen(cmd_service, shell=True)
                time.sleep(2)
                
                lt_name = f"lt_{name}"
                print(f"\nGenerando link para {name.upper()}...")
                self.processes[lt_name] = subprocess.Popen(f"lt --port {port}", shell=True, stdout=subprocess.PIPE, text=True)
                
                url_line = self.processes[lt_name].stdout.readline()
                url = url_line.split("is:")[-1].strip() if "is:" in url_line else "URL no encontrada"
                
                print(f"\n" + "="*60)
                print(f" 🌐 LINK DIRECTO PARA {name.upper()}: {url}")
                print("="*60 + "\n")
                sys.stdout.flush()
                
                return f"✅ Activo\n🌐 Link: {url}"
            return "✅ El servicio ya estaba activo. Revisa la consola de Kaggle para el link."

        def start_filebrowser(self, port=8080):
            cmd = f"filebrowser -p {port} -r {KAGGLE_WORK_DIR} --noauth"
            return self.start_service("filebrowser", cmd, port)

        def start_tensorboard(self, port=6006):
            cmd = f"tensorboard --logdir={LOGS_DIR} --port={port} --bind_all"
            return self.start_service("tensorboard", cmd, port)

    services = BackgroundServices()

    def scan_models():
        pth, idx = ["No hay modelos .pth"], ["No hay índices .index"]
        if os.path.exists(LOGS_DIR):
            pth = [os.path.join(r, f) for r, _, fs in os.walk(LOGS_DIR) for f in fs if f.endswith(".pth") and not f.startswith(("G_", "D_"))] or pth
            idx = [os.path.join(r, f) for r, _, fs in os.walk(LOGS_DIR) for f in fs if f.endswith(".index")] or idx
        return gr.update(choices=pth, value=pth[0]), gr.update(choices=idx, value=idx[0])

    def process_dataset(audio_path, ds_name, slice_length):
        if not audio_path: yield "❌ Error: No subiste ningún audio."
        yield f"📁 Cortando audio en fragmentos de {slice_length}s..."
        try:
            out_dir = os.path.join(DATASETS_DIR, ds_name)
            if os.path.exists(out_dir): shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            
            audio = AudioSegment.from_file(audio_path)
            chunks = make_chunks(audio, int(slice_length) * 1000)
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > 1000:
                    chunk.export(os.path.join(out_dir, f"{ds_name}_{i:04d}.wav"), format="wav")
                    
            yield f"✅ Éxito: {len(chunks)} frases guardadas en '{ds_name}'. Listas para extraer."
        except Exception as e:
            yield f"❌ Error: {str(e)}"

    def extract_features(ds_name, embedder, f0_method, gpu):
        emb_name = "whisper-large-v3" if "whisper" in embedder.lower() else "contentvec"
        
        yield "🔄 [Paso 1/2] Preprocesando Dataset...\n"
        cmd_pre = f"cd {BACKEND_DIR} && python core.py preprocess --model_name '{ds_name}' --dataset_path '{os.path.join(DATASETS_DIR, ds_name)}'"
        for log in stream_cmd_realtime(cmd_pre): yield log
        
        yield "\n🔄 [Paso 2/2] Extrayendo Pitch y Features...\n"
        cmd_ex = f"cd {BACKEND_DIR} && python core.py extract --model_name '{ds_name}' --f0method {f0_method.lower()} --embedder_model {emb_name} || python core.py extract --model_name '{ds_name}' --f0method {f0_method.lower()} --embedder {emb_name} || python core.py extract --model_name '{ds_name}' --f0method {f0_method.lower()}"
        for log in stream_cmd_realtime(cmd_ex): yield log

    def run_training(m_name, sr, vocoder, epochs, batch, save_ev, save_lat, save_sm, save_every_weights):
        s_lat = "True" if save_lat else "False"
        s_w = "True" if save_every_weights else "False"
        
        cmd_tr = f"cd {BACKEND_DIR} && python core.py train --model_name '{m_name}' --total_epoch {epochs} --batch_size {batch} --save_every_epoch {save_ev} --save_only_latest {s_lat} --save_every_weights {s_w} || python core.py train --model_name '{m_name}' --total_epoch {epochs} --batch_size {batch} --save_every_epoch {save_ev}"
        yield from stream_cmd_realtime(cmd_tr)

    def build_index(m_name, algo):
        cmd_idx = f"cd {BACKEND_DIR} && python core.py index --model_name '{m_name}'"
        yield from stream_cmd_realtime(cmd_idx)

    def run_inference(m_path, idx_path, audio, pitch, f0, inf_embedder, inf_format):
        if not audio: yield None, "❌ Falta audio base"
        out_path = os.path.join(KAGGLE_WORK_DIR, f"output_{int(time.time())}.{inf_format}")
        
        m_name_only = os.path.basename(m_path).replace(".pth", "")
        
        cmd = f"cd {BACKEND_DIR} && python core.py infer --model_name '{m_name_only}' --index_path '{idx_path}' --audio_path '{audio}' --export_format {inf_format} --f0method {f0.lower()} --pitch {pitch} --output_path '{out_path}' || python core.py infer --model_name '{m_name_only}' --index_path '{idx_path}' --input_path '{audio}' --export_format {inf_format} --f0method {f0.lower()} --pitch {pitch} --output_path '{out_path}'"
        
        for log in stream_cmd_realtime(cmd):
            yield None, log
        
        if os.path.exists(out_path): yield out_path, "✅ Inferencia exitosa"
        else: yield None, "❌ Fallo al generar el audio. Revisa la consola."

    with gr.Blocks(title="AriaVC Studio Pro", theme=gr.themes.Base()) as aria_ui:
        gr.Markdown("# 🎵 AriaVC Studio Pro V6 - Instalación Masiva")
        
        with gr.Tabs():
            with gr.TabItem("🎙️ 1. Dataset (Auto-Cortado Rápido)"):
                with gr.Row():
                    ds_file = gr.Audio(type="filepath", label="Sube tu Audio de Dataset (WAV/FLAC/MP3)")
                    with gr.Column():
                        ds_name = gr.Textbox(value="Mi_Modelo_Vocal", label="Nombre del Modelo")
                        slice_length = gr.Slider(5, 20, 12, step=1, label="Duración por fragmento (Segundos)")
                btn_process = gr.Button("✂️ Procesar Rápido", variant="primary")
                out_process = gr.Textbox(label="Consola de Cortado", lines=3)
                btn_process.click(process_dataset, [ds_file, ds_name, slice_length], out_process)

            with gr.TabItem("🧠 2. Extracción (Real)"):
                with gr.Row():
                    ds_ref = gr.Textbox(value="Mi_Modelo_Vocal", label="Nombre")
                    embedder = gr.Dropdown(choices=["Whisper-large-v3", "ContentVec-500k"], value="Whisper-large-v3", label="Embedder")
                    f0_method = gr.Dropdown(choices=["FCPE", "RMVPE"], value="FCPE", label="Pitch")
                    gpu_extract = gr.Checkbox(value=True, label="Usar GPU")
                btn_extract = gr.Button("🔍 Extraer", variant="primary")
                out_extract = gr.Textbox(label="Consola de Extracción", lines=10)
                btn_extract.click(extract_features, [ds_ref, embedder, f0_method, gpu_extract], out_extract)

            with gr.TabItem("🚀 3. Entrenamiento (Real)"):
                with gr.Row():
                    m_name_train = gr.Textbox(value="Mi_Modelo_Vocal", label="Nombre")
                    sr_train = gr.Radio(choices=["32k", "40k", "48k"], value="40k", label="Sample Rate")
                    vocoder = gr.Dropdown(choices=["BigVGAN-v2", "HiFi-GAN"], value="BigVGAN-v2", label="Vocoder")
                with gr.Row():
                    epochs = gr.Slider(10, 2000, 300, label="Épocas Totales")
                    batch = gr.Slider(1, 64, 8, label="Batch Size")
                with gr.Accordion("📦 Config de Guardado", open=True):
                    with gr.Row():
                        save_ev = gr.Slider(1, 100, 10, label="Guardar cada X Épocas")
                        save_lat = gr.Checkbox(value=True, label="Guardar SOLO el último")
                        save_sm = gr.Checkbox(value=True, label="Extraer .pth pequeño")
                        save_every_weights = gr.Checkbox(value=False, label="Guardar TODOS los pesos (Save Every Weights)")
                btn_train = gr.Button("🚀 Iniciar Entrenamiento", variant="primary")
                out_train = gr.Textbox(label="Consola de Entrenamiento", lines=12)
                btn_train.click(run_training, [m_name_train, sr_train, vocoder, epochs, batch, save_ev, save_lat, save_sm, save_every_weights], out_train)

            with gr.TabItem("📊 4. Índice (FAISS)"):
                with gr.Row():
                    faiss_model = gr.Textbox(value="Mi_Modelo_Vocal", label="Nombre")
                    algo = gr.Dropdown(choices=["IVFFlat", "HNSW"], value="IVFFlat", label="Algoritmo")
                btn_idx = gr.Button("📊 Crear Índice", variant="primary")
                out_idx = gr.Textbox(label="Consola FAISS", lines=4)
                btn_idx.click(build_index, [faiss_model, algo], out_idx)

            with gr.TabItem("🎧 5. Inferencia (Opciones Pro)"):
                with gr.Row():
                    with gr.Column():
                        inf_model = gr.Dropdown(choices=[], label="Modelo (.pth)")
                        inf_index = gr.Dropdown(choices=[], label="Índice (.index)")
                        btn_ref = gr.Button("🔄 Refrescar Listas")
                        inf_audio = gr.Audio(type="filepath", label="Sube Audio Base (WAV/FLAC/MP3)")
                    with gr.Column():
                        inf_pitch = gr.Slider(-24, 24, 0, step=1, label="Transposición de Tono")
                        inf_f0 = gr.Dropdown(choices=["FCPE", "RMVPE"], value="FCPE", label="Algoritmo Pitch")
                        inf_embedder = gr.Dropdown(choices=["Whisper-large-v3", "ContentVec-500k"], value="Whisper-large-v3", label="Embedder")
                        inf_format = gr.Dropdown(choices=["wav", "flac", "mp3"], value="wav", label="Formato de Exportación")
                        
                btn_inf = gr.Button("🎧 Generar Audio", variant="primary")
                out_audio = gr.Audio(label="Resultado Final")
                out_log = gr.Textbox(label="Consola de Generación", lines=6)
                
                btn_ref.click(scan_models, outputs=[inf_model, inf_index])
                btn_inf.click(run_inference, [inf_model, inf_index, inf_audio, inf_pitch, inf_f0, inf_embedder, inf_format], [out_audio, out_log])

            with gr.TabItem("🌐 6. Servicios & Archivos"):
                gr.Markdown("### 🔗 Da clic aquí y revisa la consola de Kaggle para ver el link directo")
                with gr.Row():
                    btn_fb = gr.Button("📂 Generar Link de Filebrowser")
                    out_fb = gr.Textbox(label="Estado")
                    btn_fb.click(lambda: services.start_filebrowser(), outputs=out_fb)
                    
                    btn_tb = gr.Button("📈 Generar Link de Tensorboard")
                    out_tb = gr.Textbox(label="Estado")
                    btn_tb.click(lambda: services.start_tensorboard(), outputs=out_tb)

        ds_name.change(fn=lambda x: (x, x, x), inputs=ds_name, outputs=[ds_ref, m_name_train, faiss_model])
        aria_ui.load(fn=scan_models, outputs=[inf_model, inf_index])
        
    print("\n" + "="*50)
    print(" ✅ SISTEMA LISTO. ABRE EL LINK DE GRADIO ABAJO")
    print("="*50 + "\n")
    sys.stdout.flush()
    aria_ui.queue().launch(share=True, show_error=True, inline=False)

if __name__ == "__main__":
    setup_environment()
    launch_ui()
