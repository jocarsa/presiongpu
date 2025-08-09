# -*- coding: utf-8 -*-
"""
Monitor GPU NVIDIA (barras pegadas, última ventana de 60s) + log CSV
- Dos gráficos: PROCESADOR (%) y MEMORIA (%)
- Cada barra = una muestra (últimos 60 segundos, lo más nuevo a la derecha)
- Barras pegadas, sin hueco, color sólido según % (verde->rojo)
- CSV completo: epoch, fecha_completa, carga_procesador, carga_memoria
"""

import time
import os
import csv
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName
    )
except ImportError:
    raise SystemExit("Instala NVML para Python:  pip install nvidia-ml-py3")

# ========= CONFIGURACIÓN =========
GPU_INDEX = 0
INTERVALO_MUESTREO = 0.5   # segundos
VENTANA_SEGUNDOS = 60      # ventana visible
RUTA_CSV = "gpu_log.csv"

# ========= CSV =========
nuevo = not os.path.exists(RUTA_CSV)
f_csv = open(RUTA_CSV, "a", newline="", encoding="utf-8")
w_csv = csv.writer(f_csv)
if nuevo:
    w_csv.writerow(["epoch", "fecha_completa", "carga_procesador", "carga_memoria"])
f_csv.flush()

# ========= NVML =========
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(GPU_INDEX)
gpu_name = nvmlDeviceGetName(handle).decode("utf-8")

# ========= Buffers =========
t_muestras = deque()
proc_vals  = deque()
mem_vals   = deque()

def recorta_antiguas(ahora_epoch):
    """Elimina muestras fuera de la ventana."""
    corte = ahora_epoch - VENTANA_SEGUNDOS
    while t_muestras and t_muestras[0] < corte:
        t_muestras.popleft()
        proc_vals.popleft()
        mem_vals.popleft()

def color_por_porcentaje(pct):
    """0% verde -> 100% rojo"""
    t = max(0.0, min(1.0, pct / 100.0))
    return (t, 1.0 - t, 0.0)  # (R,G,B)

# ========= Matplotlib =========
plt.ion()
fig, (ax_proc, ax_mem) = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle(f"Monitor GPU NVIDIA – {gpu_name} – Últimos {VENTANA_SEGUNDOS}s (barras pegadas)")

for ax, titulo, ylabel in [
    (ax_proc, "Procesador GPU (%)", "Porcentaje"),
    (ax_mem,  "Memoria GPU (%)",    "Porcentaje"),
]:
    ax.set_xlim(0, VENTANA_SEGUNDOS)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

ax_mem.set_xlabel(f"Tiempo (últimos {VENTANA_SEGUNDOS} segundos)")

plt.tight_layout()

try:
    while True:
        ahora = time.time()

        # Lecturas
        util = nvmlDeviceGetUtilizationRates(handle)
        carga_proc = float(util.gpu)
        mem = nvmlDeviceGetMemoryInfo(handle)
        carga_mem = (mem.used / mem.total) * 100.0

        # Guardar CSV
        w_csv.writerow([
            f"{ahora:.3f}",
            datetime.fromtimestamp(ahora).isoformat(sep=" ", timespec="seconds"),
            f"{carga_proc:.2f}",
            f"{carga_mem:.2f}",
        ])
        f_csv.flush()

        # Añadir a buffers y recortar
        t_muestras.append(ahora)
        proc_vals.append(carga_proc)
        mem_vals.append(carga_mem)
        recorta_antiguas(ahora)

        # Eje X relativo
        if t_muestras:
            x = [t - (ahora - VENTANA_SEGUNDOS) for t in t_muestras]
        else:
            x = []

        # Ancho = distancia exacta entre muestras para que no haya hueco
        if len(x) >= 2:
            dx = sum(x[i+1] - x[i] for i in range(len(x)-1)) / (len(x)-1)
            width = dx
        else:
            width = INTERVALO_MUESTREO

        colores_proc = [color_por_porcentaje(v) for v in proc_vals]
        colores_mem  = [color_por_porcentaje(v) for v in mem_vals]

        # Dibujar procesador
        ax_proc.cla()
        ax_proc.bar(x, proc_vals, width=width, align="center", color=colores_proc, edgecolor="none")
        ax_proc.set_xlim(0, VENTANA_SEGUNDOS)
        ax_proc.set_ylim(0, 100)
        ax_proc.set_ylabel("Porcentaje")
        ax_proc.set_title("Procesador GPU (%)", fontsize=11)
        ax_proc.grid(True, axis="y", linestyle="--", alpha=0.4)

        # Dibujar memoria
        ax_mem.cla()
        ax_mem.bar(x, mem_vals, width=width, align="center", color=colores_mem, edgecolor="none")
        ax_mem.set_xlim(0, VENTANA_SEGUNDOS)
        ax_mem.set_ylim(0, 100)
        ax_mem.set_xlabel(f"Tiempo (últimos {VENTANA_SEGUNDOS} segundos)")
        ax_mem.set_ylabel("Porcentaje")
        ax_mem.set_title("Memoria GPU (%)", fontsize=11)
        ax_mem.grid(True, axis="y", linestyle="--", alpha=0.4)

        fig.suptitle(f"Monitor GPU NVIDIA – {gpu_name} – Últimos {VENTANA_SEGUNDOS}s (barras pegadas)")
        plt.tight_layout()
        plt.pause(INTERVALO_MUESTREO)

except KeyboardInterrupt:
    print("Detenido por el usuario.")
finally:
    f_csv.close()
    nvmlShutdown()

