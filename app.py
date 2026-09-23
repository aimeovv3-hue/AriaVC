import gradio as gr
import subprocess
import os
import time

# ==========================================
# 1. RUTAS Y CONFIGURACIÓN (Ejemplo Kaggle)
# ==========================================
WORKSPACE = "/kaggle/working"
LOGS_DIR = os.path.join(WORKSPACE, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ==========================================
# 2. INICIAR TÚNELES Y SERVICIOS EN SEGUNDO PLANO
# ==========================================
def start_background_services():
    print("[AriaVC] Instalando dependencias de túneles...")
    os.system("npm install -g localtunnel")
    
    print("[AriaVC] Iniciando Tensorboard en el puerto 6006...")
    subprocess.Popen(["tensorboard", "--logdir", LOGS_DIR, "--port", "6006"])
    subprocess.Popen("lt --port 6006 > tensorboard_url.txt", shell=True)
    
    print("[AriaVC] Descargando e iniciando Filebrowser en el puerto 8080...")
    os.system("curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash")
    subprocess.Popen(["filebrowser", "-r", WORKSPACE, "-p", "8080", "--noauth"])
    subprocess.Popen("lt --port 8080 > filebrowser_url.txt", shell=True)
    
    time.sleep(5) # Esperar a que los túneles se generen
    
    try:
        tb_url = open("tensorboard_url.txt").read().strip()
        fb_url = open("filebrowser_url.txt").read().strip()
        print(f"\n✅ TENSORBOARD (Gráficas): {tb_url}")
        print(f"✅ FILEBROWSER (Archivos): {fb_url}\n")
    except:
        print("Túneles inicializándose, revisa la consola más tarde.")

# ==========================================
# 3. INTERFAZ GRADIO (ENTRENAMIENTO E INFERENCIA)
# ==========================================
def dummy_train(dataset, epochs, precision):
    # Aquí conectarías la función train_aria() del paso anterior
    return f"Entrenamiento iniciado con {dataset} a {epochs} epochs en modo {precision}. Revisa Tensorboard para el progreso."

def dummy_infer(audio_input, model_name, pitch_alg):
    # Aquí conectarías la función de inferencia con FCPE/RMVPE y BigVGAN
    return audio_input, f"Audio procesado exitosamente usando el modelo {model_name} y algoritmo {pitch_alg}."

with gr.Blocks(theme=gr.themes.Base()) as aria_ui:
    gr.Markdown("# 🎙️ AriaVC - Advanced Vocal Conversion (Supera a RVC2)")
    
    with gr.Tabs():
        # PESTAÑA 1: ENTRENAMIENTO
        with gr.TabItem("🚀 Entrenamiento Superior"):
            with gr.Row():
                with gr.Column():
                    dataset_path = gr.Textbox(label="Ruta del Dataset (Acapellas procesadas)", placeholder="/kaggle/input/...")
                    epochs = gr.Slider(minimum=10, maximum=1000, value=200, step=10, label="Epochs")
                    precision = gr.Radio(["fp32", "fp16", "tf32"], value="tf32", label="Precisión (TF32 recomendado)")
                    train_btn = gr.Button("Iniciar Entrenamiento", variant="primary")
                with gr.Column():
                    train_output = gr.Textbox(label="Estado del Entrenamiento")
            
            train_btn.click(dummy_train, inputs=[dataset_path, epochs, precision], outputs=[train_output])
            
        # PESTAÑA 2: INFERENCIA / PRUEBA
        with gr.TabItem("🎧 Inferencia (Probar Modelo)"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(label="Audio Original (Voz limpia)", type="filepath")
                    model_sel = gr.Dropdown(choices=["Aria_Vocal_v1.pth", "Aria_Vocal_v2.pth"], label="Seleccionar Modelo")
                    pitch_ext = gr.Radio(["FCPE", "RMVPE Pro", "Híbrido (FCPE+RMVPE)"], value="Híbrido (FCPE+RMVPE)", label="Extractor de Pitch")
                    infer_btn = gr.Button("Convertir Voz", variant="primary")
                with gr.Column():
                    audio_out = gr.Audio(label="Voz Convertida (48kHz BigVGAN)")
                    infer_log = gr.Textbox(label="Registro de Inferencia")
                    
            infer_btn.click(dummy_infer, inputs=[audio_in, model_sel, pitch_ext], outputs=[audio_out, infer_log])

if __name__ == "__main__":
    start_background_services()
    # share=True crea el túnel público (el enlace de Gradio)
    aria_ui.launch(share=True, server_name="0.0.0.0", server_port=7860)