import gradio as gr
import subprocess
import os
import time
import shutil

# ==========================================
# 1. CONFIGURACIÓN DEL ENTORNO
# ==========================================
WORKSPACE = "/kaggle/working"
LOGS_DIR = os.path.join(WORKSPACE, "logs")
DATASET_DIR = os.path.join(WORKSPACE, "dataset_temp")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ==========================================
# 2. SERVICIOS EN SEGUNDO PLANO (Túneles)
# ==========================================
def start_background_services():
    """Inicia Filebrowser y Tensorboard y crea enlaces públicos."""
    print("[AriaVC] Iniciando servicios...")
    os.system("npm install -g localtunnel")
    
    # Tensorboard
    subprocess.Popen(["tensorboard", "--logdir", LOGS_DIR, "--port", "6006"])
    subprocess.Popen("lt --port 6006 > tensorboard_url.txt", shell=True)
    
    # Filebrowser
    os.system("curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash")
    subprocess.Popen(["filebrowser", "-r", WORKSPACE, "-p", "8080", "--noauth"])
    subprocess.Popen("lt --port 8080 > filebrowser_url.txt", shell=True)
    
    time.sleep(5)

# ==========================================
# 3. LÓGICA DE ENTRENAMIENTO Y MANEJO DE ARCHIVOS
# ==========================================
def process_dataset_and_train(archivos_subidos, epochs, precision):
    """Maneja la subida de archivos, los procesa y simula el entrenamiento."""
    
    # 1. Verificación inicial (Burbuja de advertencia si no hay archivos)
    if not archivos_subidos:
        gr.Warning("⚠️ No has subido ningún dataset. Sube al menos un archivo de audio o ZIP.")
        return "Error: Dataset vacío."
    
    gr.Info("📦 Procesando archivos subidos...")
    
    # 2. Limpiar directorio anterior y guardar nuevos archivos
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    for archivo in archivos_subidos:
        nombre_base = os.path.basename(archivo.name)
        destino = os.path.join(DATASET_DIR, nombre_base)
        shutil.copy(archivo.name, destino)
    
    cantidad_archivos = len(os.listdir(DATASET_DIR))
    gr.Info(f"✅ {cantidad_archivos} archivos guardados correctamente en el entorno.")
    
    # 3. Inicio del entrenamiento
    gr.Info("🚀 Extrayendo Pitch y Embeddings... Esto tomará un momento.")
    time.sleep(2) # Simulación de tiempo de procesamiento
    
    gr.Info(f"⚡ Iniciando entrenamiento en modo {precision} por {epochs} epochs.")
    # Aquí iría el bucle de entrenamiento real (tu train_kaggle.py integrado)
    time.sleep(3) # Simulación de entrenamiento
    
    gr.Info("🎉 ¡Entrenamiento finalizado con éxito! Modelo guardado.")
    return f"Entrenamiento completado usando {cantidad_archivos} archivos a {epochs} epochs en {precision}."

# ==========================================
# 4. LÓGICA DE INFERENCIA
# ==========================================
def process_inference(audio_input, model_name, pitch_alg):
    if audio_input is None:
        gr.Warning("⚠️ Por favor sube un audio o graba con el micrófono primero.")
        return None, "Falta archivo de audio."
    
    gr.Info(f"🔍 Analizando audio con {pitch_alg}...")
    time.sleep(1) # Simulación de extracción de pitch
    
    gr.Info("✨ Pasando por el vocoder BigVGAN (48kHz)...")
    time.sleep(1) # Simulación de renderizado
    
    gr.Info("✅ Conversión completada exitosamente.")
    
    # Retorna el mismo audio como "procesado" por ahora y un mensaje
    return audio_input, "Inferencia exitosa."

# ==========================================
# 5. INTERFAZ GRÁFICA (GRADIO)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as aria_ui:
    gr.Markdown("# 🎙️ AriaVC - Interfaz Avanzada de Entrenamiento e Inferencia")
    
    with gr.Tabs():
        # --- PESTAÑA DE ENTRENAMIENTO ---
        with gr.TabItem("🚀 Entrenamiento Superior"):
            with gr.Row():
                with gr.Column():
                    # Nuevo componente para arrastrar archivos múltiples
                    dataset_files = gr.File(label="Arrastra tus Acapellas (Audios) aquí", file_count="multiple")
                    epochs = gr.Slider(minimum=10, maximum=1000, value=200, step=10, label="Epochs")
                    precision = gr.Radio(["fp32", "fp16", "tf32"], value="tf32", label="Precisión (TF32 para RTX/A100)")
                    train_btn = gr.Button("Iniciar Procesamiento y Entrenamiento", variant="primary")
                
                with gr.Column():
                    train_output = gr.Textbox(label="Estado del Sistema")
                    gr.Markdown("""
                    **Instrucciones:**
                    1. Arrastra tus archivos `.wav` al recuadro.
                    2. Ajusta las epochs.
                    3. Revisa la esquina superior derecha para ver las notificaciones en burbuja.
                    """)
            
            # Conexión del botón con la función y los elementos visuales
            train_btn.click(
                fn=process_dataset_and_train, 
                inputs=[dataset_files, epochs, precision], 
                outputs=[train_output]
            )
            
        # --- PESTAÑA DE INFERENCIA ---
        with gr.TabItem("🎧 Inferencia (Probar)"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(label="Audio Original", type="filepath")
                    model_sel = gr.Dropdown(choices=["Aria_Vocal_v1.pth"], label="Seleccionar Modelo Entrenado")
                    pitch_ext = gr.Radio(["FCPE", "RMVPE Pro", "Híbrido (FCPE+RMVPE)"], value="Híbrido (FCPE+RMVPE)", label="Algoritmo de Pitch")
                    infer_btn = gr.Button("Convertir Voz", variant="primary")
                
                with gr.Column():
                    audio_out = gr.Audio(label="Voz Convertida (48kHz)")
                    infer_log = gr.Textbox(label="Registro")
                    
            infer_btn.click(
                fn=process_inference, 
                inputs=[audio_in, model_sel, pitch_ext], 
                outputs=[audio_out, infer_log]
            )

if __name__ == "__main__":
    start_background_services()
    aria_ui.launch(share=True, server_name="0.0.0.0", server_port=7860)